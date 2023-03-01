import glob
import os
import subprocess
import cv2
import numpy as np
import json

from utils.data_processing import load_landmark_openface,compute_crop_radius
from utils.deep_speech import DeepSpeech
from config.config import DataProcessingOptions

def extract_audio(source_video_dir,res_audio_dir):
    '''
    extract audio files from videos
    '''
    if not os.path.exists(source_video_dir):
        raise ('wrong path of video dir')
    if not os.path.exists(res_audio_dir):
        os.mkdir(res_audio_dir)
    video_path_list = glob.glob(os.path.join(source_video_dir, '*.mp4'))
    for video_path in video_path_list:
        print('extract audio from video: {}'.format(os.path.basename(video_path)))
        audio_path = os.path.join(res_audio_dir, os.path.basename(video_path).replace('.mp4', '.wav'))
        cmd = 'ffmpeg -i {} -f wav -ar 16000 {}'.format(video_path, audio_path)
        subprocess.call(cmd, shell=True)

def extract_deep_speech(audio_dir,res_deep_speech_dir,deep_speech_model_path):
    '''
    extract deep speech feature
    '''
    if not os.path.exists(res_deep_speech_dir):
        os.mkdir(res_deep_speech_dir)
    DSModel = DeepSpeech(deep_speech_model_path)
    wav_path_list = glob.glob(os.path.join(audio_dir, '*.wav'))
    for wav_path in wav_path_list:
        video_name = os.path.basename(wav_path).replace('.wav', '')
        res_dp_path = os.path.join(res_deep_speech_dir, video_name + '_deepspeech.txt')
        if os.path.exists(res_dp_path):
            os.remove(res_dp_path)
        print('extract deep speech feature from audio:{}'.format(video_name))
        ds_feature = DSModel.compute_audio_feature(wav_path)
        np.savetxt(res_dp_path, ds_feature)

def extract_video_frame(source_video_dir,res_video_frame_dir):
    '''
        extract video frames from videos
    '''
    if not os.path.exists(source_video_dir):
        raise ('wrong path of video dir')
    if not os.path.exists(res_video_frame_dir):
        os.mkdir(res_video_frame_dir)
    video_path_list = glob.glob(os.path.join(source_video_dir, '*.mp4'))
    for video_path in video_path_list:
        video_name = os.path.basename(video_path)
        frame_dir = os.path.join(res_video_frame_dir, video_name.replace('.mp4', ''))
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        print('extracting frames from {} ...'.format(video_name))
        videoCapture = cv2.VideoCapture(video_path)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        if int(fps) != 25:
            raise ('{} video is not in 25 fps'.format(video_path))
        frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        for i in range(int(frames)):
            ret, frame = videoCapture.read()
            result_path = os.path.join(frame_dir, str(i).zfill(6) + '.jpg')
            cv2.imwrite(result_path, frame)


def crop_face_according_openfaceLM(openface_landmark_dir,video_frame_dir,res_crop_face_dir,clip_length):
    '''
      crop face according to openface landmark
    '''
    if not os.path.exists(openface_landmark_dir):
        raise ('wrong path of openface landmark dir')
    if not os.path.exists(video_frame_dir):
        raise ('wrong path of video frame dir')
    if not os.path.exists(res_crop_face_dir):
        os.mkdir(res_crop_face_dir)
    landmark_openface_path_list = glob.glob(os.path.join(openface_landmark_dir, '*.csv'))
    for landmark_openface_path in landmark_openface_path_list:
        video_name = os.path.basename(landmark_openface_path).replace('.csv', '')
        crop_face_video_dir = os.path.join(res_crop_face_dir, video_name)
        if not os.path.exists(crop_face_video_dir):
            os.makedirs(crop_face_video_dir)
        print('cropping face from video: {} ...'.format(video_name))
        landmark_openface_data = load_landmark_openface(landmark_openface_path).astype(np.int)
        frame_dir = os.path.join(video_frame_dir, video_name)
        if not os.path.exists(frame_dir):
            raise ('run last step to extract video frame')
        if len(glob.glob(os.path.join(frame_dir, '*.jpg'))) != landmark_openface_data.shape[0]:
            raise ('landmark length is different from frame length')
        frame_length = min(len(glob.glob(os.path.join(frame_dir, '*.jpg'))), landmark_openface_data.shape[0])
        end_frame_index = list(range(clip_length, frame_length, clip_length))
        video_clip_num = len(end_frame_index)
        for i in range(video_clip_num):
            first_image = cv2.imread(os.path.join(frame_dir, '000000.jpg'))
            video_h,video_w = first_image.shape[0], first_image.shape[1]
            crop_flag, radius_clip = compute_crop_radius((video_w,video_h),
                                    landmark_openface_data[end_frame_index[i] - clip_length:end_frame_index[i], :,:])
            if not crop_flag:
                continue
            radius_clip_1_4 = radius_clip // 4
            print('cropping {}/{} clip from video:{}'.format(i, video_clip_num, video_name))
            res_face_clip_dir = os.path.join(crop_face_video_dir, str(i).zfill(6))
            if not os.path.exists(res_face_clip_dir):
                os.mkdir(res_face_clip_dir)
            for frame_index in range(end_frame_index[i]- clip_length,end_frame_index[i]):
                source_frame_path = os.path.join(frame_dir,str(frame_index).zfill(6)+'.jpg')
                source_frame_data = cv2.imread(source_frame_path)
                frame_landmark = landmark_openface_data[frame_index, :, :]
                crop_face_data = source_frame_data[
                                    frame_landmark[29, 1] - radius_clip:frame_landmark[
                                                                            29, 1] + radius_clip * 2 + radius_clip_1_4,
                                    frame_landmark[33, 0] - radius_clip - radius_clip_1_4:frame_landmark[
                                                                                              33, 0] + radius_clip + radius_clip_1_4,
                                    :].copy()
                res_crop_face_frame_path = os.path.join(res_face_clip_dir, str(frame_index).zfill(6) + '.jpg')
                if os.path.exists(res_crop_face_frame_path):
                    os.remove(res_crop_face_frame_path)
                cv2.imwrite(res_crop_face_frame_path, crop_face_data)


def generate_training_json(crop_face_dir,deep_speech_dir,clip_length,res_json_path):
    video_name_list = os.listdir(crop_face_dir)
    video_name_list.sort()
    res_data_dic = {}
    for video_index, video_name in enumerate(video_name_list):
        print('generate training json file :{} {}/{}'.format(video_name,video_index,len(video_name_list)))
        tem_dic = {}
        deep_speech_feature_path = os.path.join(deep_speech_dir, video_name + '_deepspeech.txt')
        if not os.path.exists(deep_speech_feature_path):
            raise ('wrong path of deep speech')
        deep_speech_feature = np.loadtxt(deep_speech_feature_path)
        video_clip_dir = os.path.join(crop_face_dir, video_name)
        clip_name_list = os.listdir(video_clip_dir)
        clip_name_list.sort()
        video_clip_num = len(clip_name_list)
        clip_data_list = []
        for clip_index, clip_name in enumerate(clip_name_list):
            tem_tem_dic = {}
            clip_frame_dir = os.path.join(video_clip_dir, clip_name)
            frame_path_list = glob.glob(os.path.join(clip_frame_dir, '*.jpg'))
            frame_path_list.sort()
            assert len(frame_path_list) == clip_length
            start_index = int(float(clip_name) * clip_length)
            assert int(float(os.path.basename(frame_path_list[0]).replace('.jpg', ''))) == start_index
            frame_name_list = [video_name + '/' + clip_name + '/' + os.path.basename(item) for item in frame_path_list]
            deep_speech_list = deep_speech_feature[start_index:start_index + clip_length, :].tolist()
            if len(frame_name_list) != len(deep_speech_list):
                print(' skip video: {}:{}/{}  clip:{}:{}/{} because of different length: {} {}'.format(
                    video_name,video_index,len(video_name_list),clip_name,clip_index,len(clip_name_list),
                     len(frame_name_list),len(deep_speech_list)))
            tem_tem_dic['frame_name_list'] = frame_name_list
            tem_tem_dic['frame_path_list'] = frame_path_list
            tem_tem_dic['deep_speech_list'] = deep_speech_list
            clip_data_list.append(tem_tem_dic)
        tem_dic['video_clip_num'] = video_clip_num
        tem_dic['clip_data_list'] = clip_data_list
        res_data_dic[video_name] = tem_dic
    if os.path.exists(res_json_path):
        os.remove(res_json_path)
    with open(res_json_path,'w') as f:
        json.dump(res_data_dic,f)


if __name__ == '__main__':
    opt = DataProcessingOptions().parse_args()
    ##########  step1: extract video frames
    if opt.extract_video_frame:
        extract_video_frame(opt.source_video_dir, opt.video_frame_dir)
    ##########  step2: extract audio files
    if opt.extract_audio:
        extract_audio(opt.source_video_dir,opt.audio_dir)
    ##########  step3: extract deep speech features
    if opt.extract_deep_speech:
        extract_deep_speech(opt.audio_dir, opt.deep_speech_dir,opt.deep_speech_model)
    ##########  step4: crop face images
    if opt.crop_face:
        crop_face_according_openfaceLM(opt.openface_landmark_dir,opt.video_frame_dir,opt.crop_face_dir,opt.clip_length)
    ##########  step5: generate training json file
    if opt.generate_training_json:
        generate_training_json(opt.crop_face_dir,opt.deep_speech_dir,opt.clip_length,opt.json_path)


