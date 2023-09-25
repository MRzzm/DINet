"""
This script is used to do random sampling from the DINet training dataset for 5-frame prediction.
"""
import os
import random

import cv2
import numpy as np

from config.config import DINetTrainingOptions
from dataset.dataset_DINet_clip import DINetDataset


if __name__ == "__main__":
    opt = DINetTrainingOptions().parse_args()

    # create debug dir
    debug_dir = r'./asserts/debug_clip'
    os.makedirs(debug_dir, exist_ok=True)

    # set seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)

    # load training data in memory
    train_data = DINetDataset(opt.train_data, opt.augment_num, opt.mouth_region_size)

    print(f'Length of training data: {len(train_data)}')

    for i in range(len(train_data)):
        source_clip, source_clip_mask, reference_clip, deep_speech_clip, deep_speech_full = train_data[i]

        # source_clip: [n, 3, H, W] RGB float32 [0..1]
        # source_clip_mask: [n, 3, H, W] RGB float32 [0..1] with black mouth region
        # reference_clip: [n, 3*k, H, W] RGB float32 [0..1] with k frames concatenated
        # deep_speech_clip: [n, 29, t] float32, t features centered around the source frame
        # deep_speech_full: [29, T] float32, T features for the whole video

        n = source_clip.shape[0]  # number of frames
        k = reference_clip.shape[1] // 3  # number of reference frames for each frame
        h, w = source_clip.shape[2:]  # height and width of the frames

        source_clip = source_clip.permute(0, 2, 3, 1)  # [n, H, W, 3]
        source_clip = source_clip.view(-1, *source_clip.shape[2:])  # [n*H, W, 3] - stacking time vertically
        source_clip = np.clip(source_clip.numpy() * 255, 0, 255).astype(np.uint8)

        source_clip_mask = source_clip_mask.permute(0, 2, 3, 1)  # [n, H, W, 3]
        source_clip_mask = source_clip_mask.view(n * h, w, 3)  # [n*H, W, 3] - stacking time vertically
        source_clip_mask = np.clip(source_clip_mask.numpy() * 255, 0, 255).astype(np.uint8)

        reference_clip = reference_clip.view(n, k, 3, h, w)  # [n, k, 3, H, W]
        reference_clip = reference_clip.permute(0, 3, 1, 4, 2)  # [n, H, k, W, 3]
        reference_clip = reference_clip.reshape(n * h, k * w, 3)  # [n*H, k*W, 3] - stacking time vertically and frames horizontally
        reference_clip = np.clip(reference_clip.numpy() * 255, 0, 255).astype(np.uint8)

        # stacking source, mask and reference images horizontally
        display_img = np.concatenate([source_clip, source_clip_mask, reference_clip], 1)
        display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)

        # saving to debug dir
        cv2.imwrite(os.path.join(debug_dir, f'{i}.jpg'), display_img)
