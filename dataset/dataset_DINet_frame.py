import torch
import numpy as np
import json
import random
import cv2
import os

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
        random_anchor = random.sample(range(video_clip_num), 6)
        source_anchor, reference_anchor_list = random_anchor[0],random_anchor[1:]
        ## load source image
        source_image_path_list = self.data_dic[video_name]['clip_data_list'][source_anchor]['frame_path_list']
        source_random_index = random.sample(range(2, 7), 1)[0]
        ## modify the path
        source_image_path = os.path.join(source_image_path_list[source_random_index].replace('\\','/').split('/')) # fix the path problem
        if not os.path.exists(source_image_path):
            raise FileNotFoundError(f"{source_image_path} does not")
        source_image_data = cv2.imread(source_image_path_list[source_random_index])[:, :, ::-1]
        if source_image_data is None:
            raise FileNotFoundError(f"{source_image_path} does not")
        source_image_data = source_image_data[:, :, ::-1]
        source_image_data = cv2.resize(source_image_data, (self.img_w, self.img_h))/ 255.0
        source_image_mask = source_image_data.copy()
        source_image_mask[self.radius:self.radius+self.mouth_region_size,self.radius_1_4:self.radius_1_4 +self.mouth_region_size ,:] = 0

        ## load deep speech feature
        deepspeech_feature = np.array(self.data_dic[video_name]['clip_data_list'][source_anchor]['deep_speech_list'][source_random_index - 2:source_random_index + 3])

        ## load reference images
        reference_frame_data_list = []
        for reference_anchor in reference_anchor_list:
            reference_frame_path_list = self.data_dic[video_name]['clip_data_list'][reference_anchor]['frame_path_list']
            reference_random_index = random.sample(range(9), 1)[0]
            ## modify the path
            reference_frame_path = os.path.join(reference_frame_path_list[reference_random_index].replace('\\','/').split('/')) # fix the path problem
            if not os.path.exists(reference_frame_path):
                raise FileNotFoundError(f"{reference_frame_path} does not exsit")
            reference_frame_data = cv2.imread(reference_frame_path)
            if reference_frame_data is None:
                raise IOError(f"Failed to open {reference_frame_path}")
            reference_frame_data = reference_frame_data[:, :, ::-1]
            reference_frame_data = cv2.resize(reference_frame_data, (self.img_w, self.img_h))/ 255.0
            reference_frame_data_list.append(reference_frame_data)
        reference_clip_data = np.concatenate(reference_frame_data_list, 2)

        # display the source image and reference images
        # display_img = np.concatenate([source_image_data,source_image_mask]+reference_frame_data_list,1)
        # cv2.imshow('image display',(display_img[:,:,::-1] * 255).astype(np.uint8))
        # cv2.waitKey(-1)

        # # to tensor
        source_image_data = torch.from_numpy(source_image_data).float().permute(2,0,1)
        source_image_mask = torch.from_numpy(source_image_mask).float().permute(2,0,1)
        reference_clip_data = torch.from_numpy(reference_clip_data).float().permute(2,0,1)
        deepspeech_feature = torch.from_numpy(deepspeech_feature).float().permute(1,0)
        return source_image_data,source_image_mask, reference_clip_data,deepspeech_feature

    def __len__(self):
        return self.length


