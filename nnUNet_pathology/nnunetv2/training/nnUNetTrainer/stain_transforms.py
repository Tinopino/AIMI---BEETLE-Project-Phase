from typing import Optional

import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform


class StainJitterTransform(AbstractTransform):
    def __init__(self, sigma: float = 0.05, bias: float = 0.05,
                 p_per_sample: float = 0.8, data_key: str = 'data'):
        self.sigma = sigma
        self.bias = bias
        self.p_per_sample = p_per_sample
        self.data_key = data_key

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        if data is None:
            return data_dict

        B, C = data.shape[0], data.shape[1]
        for b in range(B):
            if np.random.rand() < self.p_per_sample:
                shape = (C,) + (1,) * (data.ndim - 2)
                alpha = 1.0 + np.random.uniform(-self.sigma, self.sigma, size=shape).astype(data.dtype)
                beta = np.random.uniform(-self.bias, self.bias, size=shape).astype(data.dtype)
                data[b] = data[b] * alpha + beta

        data_dict[self.data_key] = data
        return data_dict


class HEDStainAugmentation(AbstractTransform):
    """
    Stain augmentation
    """

    def __init__(self, sigma: float = 0.02, bias: float = 0.02,
                 p_per_sample: float = 0.8, data_key: str = 'data',
                 assume_uint8_range: Optional[bool] = None):
        try:
            from skimage.color import rgb2hed, hed2rgb  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "HEDStainAugmentation requires scikit-image. "
                "Install with `pip install scikit-image`."
            ) from e
        self.sigma = sigma
        self.bias = bias
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.assume_uint8_range = assume_uint8_range

    def __call__(self, **data_dict):
        from skimage.color import rgb2hed, hed2rgb
        data = data_dict[self.data_key]
        if data is None or data.ndim != 4 or data.shape[1] != 3:
            return data_dict  # only valid for 2D RGB

        B = data.shape[0]
        # Decide range once based on first sample
        if self.assume_uint8_range is None:
            uint8_range = float(data.max()) > 1.5
        else:
            uint8_range = self.assume_uint8_range

        for b in range(B):
            if np.random.rand() >= self.p_per_sample:
                continue
            img = data[b]
            img_hwc = np.transpose(img, (1, 2, 0))
            if uint8_range:
                img_hwc = img_hwc / 255.0
            img_hwc = np.clip(img_hwc, 1e-6, 1.0)

            hed = rgb2hed(img_hwc)
            alpha = 1.0 + np.random.uniform(-self.sigma, self.sigma, size=(1, 1, 3))
            beta = np.random.uniform(-self.bias, self.bias, size=(1, 1, 3))
            hed = hed * alpha + beta
            rgb = hed2rgb(hed)
            rgb = np.clip(rgb, 0.0, 1.0)

            if uint8_range:
                rgb = rgb * 255.0
            data[b] = np.transpose(rgb, (2, 0, 1)).astype(data.dtype)

        data_dict[self.data_key] = data
        return data_dict
