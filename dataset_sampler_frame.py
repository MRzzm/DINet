"""
This script is used to do random sampling from the DINet training dataset for single frame prediction.
"""
import os
import random

import cv2
import numpy as np

from config.config import DINetTrainingOptions
from dataset.dataset_DINet_frame import DINetDataset


if __name__ == "__main__":
    opt = DINetTrainingOptions().parse_args()

    # create debug dir
    debug_dir = r'./asserts/debug_frame'
    os.makedirs(debug_dir, exist_ok=True)

    # set seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)

    # load training data in memory
    train_data = DINetDataset(opt.train_data, opt.augment_num, opt.mouth_region_size)

    print(f'Length of training data: {len(train_data)}')

    for i in range(len(train_data)):
        source_image_data, source_image_mask, reference_clip_data, deepspeech_feature = train_data[i]

        # source_image_data: [3, H, W] RGB float32 [0..1]
        # source_image_mask: [3, H, W] RGB float32 [0..1] with black mouth region
        # reference_clip_data: [3*k, H, W] RGB float32 [0..1] with k frames concatenated
        # deepspeech_feature: [29, t] float32, t features centered around the source frame

        k = reference_clip_data.shape[0] // 3

        # visualizing images and saving to debug dir
        source_image_data = np.clip(source_image_data.permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
        source_image_mask = np.clip(source_image_mask.permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)

        # stacking reference images horizontally
        reference_clip_data = reference_clip_data.view(k, 3, *reference_clip_data.shape[1:])  # [k, 3, H, W]
        reference_clip_data = reference_clip_data.permute(2, 0, 3, 1)  # [H, k, W, 3]
        reference_clip_data = reference_clip_data.reshape(reference_clip_data.shape[0], -1, 3)  # [H, k*W, 3]
        reference_clip_data = np.clip(reference_clip_data.numpy() * 255, 0, 255).astype(np.uint8)

        # stacking source, mask and reference images horizontally
        display_img = np.concatenate([source_image_data, source_image_mask, reference_clip_data], 1)
        display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)

        # saving to debug dir
        cv2.imwrite(os.path.join(debug_dir, f'{i}.jpg'), display_img)
