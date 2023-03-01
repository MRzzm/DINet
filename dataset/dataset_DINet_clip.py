import torch
import numpy as np
import json
import random
import cv2

from torch.utils.data import Dataset


def get_data(json_name,augment_num):
    print('start loading data')
    with open(json_name,'r') as f:
        data_dic = json.load(f)
    data_dic_name_list = []
    for augment_index in range(augment_num):
        for video_name in data_dic.keys():
            data_dic_name_list.append(video_name)
    random.shuffle(data_dic_name_list)
    print('finish loading')
    return data_dic_name_list,data_dic


class DINetDataset(Dataset):
    def __init__(self,path_json,augment_num,mouth_region_size):
        super(DINetDataset, self).__init__()
        self.data_dic_name_list,self.data_dic = get_data(path_json,augment_num)
        self.mouth_region_size = mouth_region_size
        self.radius = mouth_region_size//2
        self.radius_1_4 = self.radius//4
        self.img_h = self.radius * 3 + self.radius_1_4
        self.img_w = self.radius * 2 + self.radius_1_4 * 2
        self.length = len(self.data_dic_name_list)

    def __getitem__(self, index):
        video_name = self.data_dic_name_list[index]
        video_clip_num = len(self.data_dic[video_name]['clip_data_list'])
        source_anchor = random.sample(range(video_clip_num), 1)[0]
        source_image_path_list = self.data_dic[video_name]['clip_data_list'][source_anchor]['frame_path_list']
        source_clip_list = []
        source_clip_mask_list = []
        deep_speech_list = []
        reference_clip_list = []
        for source_frame_index in range(2, 2 + 5):
            ## load source clip
            source_image_data = cv2.imread(source_image_path_list[source_frame_index])[:, :, ::-1]
            source_image_data = cv2.resize(source_image_data, (self.img_w, self.img_h)) / 255.0
            source_clip_list.append(source_image_data)
            source_image_mask = source_image_data.copy()
            source_image_mask[self.radius:self.radius + self.mouth_region_size,
            self.radius_1_4:self.radius_1_4 + self.mouth_region_size, :] = 0
            source_clip_mask_list.append(source_image_mask)

            ## load deep speech feature
            deepspeech_array = np.array(self.data_dic[video_name]['clip_data_list'][source_anchor]['deep_speech_list'][
                                       source_frame_index - 2:source_frame_index + 3])
            deep_speech_list.append(deepspeech_array)

            ## ## load reference images
            reference_frame_list = []
            reference_anchor_list = random.sample(range(video_clip_num), 5)
            for reference_anchor in reference_anchor_list:
                reference_frame_path_list = self.data_dic[video_name]['clip_data_list'][reference_anchor][
                    'frame_path_list']
                reference_random_index = random.sample(range(9), 1)[0]
                reference_frame_path = reference_frame_path_list[reference_random_index]
                reference_frame_data = cv2.imread(reference_frame_path)[:, :, ::-1]
                reference_frame_data = cv2.resize(reference_frame_data, (self.img_w, self.img_h)) / 255.0
                reference_frame_list.append(reference_frame_data)
            reference_clip_list.append(np.concatenate(reference_frame_list, 2))

        source_clip = np.stack(source_clip_list, 0)
        source_clip_mask = np.stack(source_clip_mask_list, 0)
        deep_speech_clip = np.stack(deep_speech_list, 0)
        reference_clip = np.stack(reference_clip_list, 0)
        deep_speech_full = np.array(self.data_dic[video_name]['clip_data_list'][source_anchor]['deep_speech_list'])

        # # display data
        # display_source = np.concatenate(source_clip_list,1)
        # display_source_mask = np.concatenate(source_clip_mask_list,1)
        # display_reference0 = np.concatenate([reference_clip_list[0][:,:,:3],reference_clip_list[0][:,:,3:6],reference_clip_list[0][:,:,6:9],
        #                                 reference_clip_list[0][:,:,9:12],reference_clip_list[0][:,:,12:15]],1)
        # display_reference1 = np.concatenate([reference_clip_list[1][:, :, :3], reference_clip_list[1][:, :, 3:6],
        #                                 reference_clip_list[1][:, :, 6:9],
        #                                 reference_clip_list[1][:, :, 9:12], reference_clip_list[1][:, :, 12:15]],1)
        # display_reference2 = np.concatenate([reference_clip_list[2][:, :, :3], reference_clip_list[2][:, :, 3:6],
        #                                 reference_clip_list[2][:, :, 6:9],
        #                                 reference_clip_list[2][:, :, 9:12], reference_clip_list[2][:, :, 12:15]],1)
        # display_reference3 = np.concatenate([reference_clip_list[3][:, :, :3], reference_clip_list[3][:, :, 3:6],
        #                                 reference_clip_list[3][:, :, 6:9],
        #                                 reference_clip_list[3][:, :, 9:12], reference_clip_list[3][:, :, 12:15]],1)
        # display_reference4 = np.concatenate([reference_clip_list[4][:, :, :3], reference_clip_list[4][:, :, 3:6],
        #                                 reference_clip_list[4][:, :, 6:9],
        #                                 reference_clip_list[4][:, :, 9:12], reference_clip_list[4][:, :, 12:15]],1)
        # merge_img = np.concatenate([display_source,display_source_mask,
        #                             display_reference0,display_reference1,display_reference2,display_reference3,
        #                             display_reference4],0)
        # cv2.imshow('test',(merge_img[:,:,::-1] * 255).astype(np.uint8))
        # cv2.waitKey(-1)



        # # 2 tensor
        source_clip = torch.from_numpy(source_clip).float().permute(0, 3, 1, 2)
        source_clip_mask = torch.from_numpy(source_clip_mask).float().permute(0, 3, 1, 2)
        reference_clip = torch.from_numpy(reference_clip).float().permute(0, 3, 1, 2)
        deep_speech_clip = torch.from_numpy(deep_speech_clip).float().permute(0, 2, 1)
        deep_speech_full = torch.from_numpy(deep_speech_full).permute(1, 0)
        return source_clip,source_clip_mask, reference_clip,deep_speech_clip,deep_speech_full

    def __len__(self):
        return self.length
