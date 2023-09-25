from typing import Tuple

import cv2
import numpy as np


class Augmenter:
    """
    This class is used to do data augmentation. Initializes random state on creation.
    """

    def __init__(
            self,
            gaussian_noise_std: float = 0.01,
            brightness_range: float = 0.1,
            contrast_range: float = 0.1,
            brightness_jitter: float = 0.02,
            contrast_jitter: float = 0.02,
            gamma_range: Tuple[float, float] = (0.8, 1.2),
            hue_range: float = 0.02,
    ):
        """
        :param gaussian_noise_std: standard deviation of gaussian noise to add to image
        :param brightness_range: range of brightness change to apply to image
        :param contrast_range: range of contrast change to apply to image
        :param brightness_jitter: jitter of brightness change to apply to image (std of normal distribution)
        :param contrast_jitter: jitter of contrast change to apply to image (std of normal distribution)
        :param gamma_range: range of gamma change to apply to image
        :param hue_range: range of hue change to apply to image
        """

        self.gaussion_noise_std = gaussian_noise_std
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter
        self.generator = np.random.default_rng()
        self.do_hflip = self.generator.uniform(0.0, 1.0) > 0.5  # random horizontal flip
        self.brightness = self.generator.uniform(1 - self.brightness_range * 0.5, 1 + self.brightness_range * 0.5)
        self.contrast = self.generator.uniform(1 - self.contrast_range * 0.5, 1 + self.contrast_range * 0.5)
        self.gamma = self.generator.uniform(gamma_range[0], gamma_range[1])
        self.hue = self.generator.standard_normal() * hue_range

    def transform_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to image.

        :param image: [H, W, 3] RGB image, float32, [0..1]
        :return: [H, W, 3] RGB image
        """
        if self.do_hflip:
            image = cv2.flip(image, 1)

        # apply random brightness and contrast
        brightness = self.brightness + self.generator.standard_normal() * self.brightness_jitter
        contrast = self.contrast + self.generator.standard_normal() * self.contrast_jitter
        image = image * brightness
        image = image + (image.mean() - image) * (1 - contrast)

        # apply random noise
        noise = self.generator.standard_normal(image.shape, dtype=image.dtype) * self.gaussion_noise_std
        image = image + noise

        # clip image to [0..1] range
        image = np.clip(image, 0, 1)

        # apply gamma
        image = np.power(image, self.gamma)

        # apply hue
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:, :, 0] = hsv[:, :, 0] + self.hue * 255
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return image
