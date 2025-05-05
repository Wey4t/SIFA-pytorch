import torch
import numpy as np
import random
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Union, Tuple, List



class BaseTransformation(object):
    def __init__(self, keys: List[str]):
        if len(keys) < 1:
            raise ValueError('The number of data keys must be at least one.')

        self.keys = keys

    def __call__(self, input_dict: Dict[str, np.ndarray]):
        pass

class Resize(BaseTransformation):
    """
    Resize 2D or 3D image tensors to a target size.
    """

    def __init__(self, size: Union[Tuple[int, int], Tuple[int, int, int]], mode: str = 'bilinear', keys: List[str] = ["image"]):
        super().__init__(keys)
        self.size = size
        self.mode = mode

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for key in self.keys:
            if key in sample:
                image = sample[key]
                if image.dim() == 3:
                    image = F.interpolate(image.unsqueeze(0), size=self.size, mode=self.mode, align_corners=False)
                    sample[key] = image.squeeze(0)
                elif image.dim() == 4:
                    C, D, H, W = image.shape
                    image = image.view(C * D, H, W).unsqueeze(1)
                    image = F.interpolate(image, size=self.size[1:], mode=self.mode, align_corners=False)
                    image = image.squeeze(1).view(C, *self.size)
                    sample[key] = image
                else:
                    raise ValueError(f"Unsupported image shape for Resize: {image.shape}.")
        return sample






class Compose(object):
    """
    Compose multiple transformations sequentially.
    """

    def __init__(self, transforms: List[BaseTransformation]):
        super().__init__()
        
        self.transforms = transforms

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for transform in self.transforms:
            sample = transform(sample)
        return sample
class ToTensor(BaseTransformation):
    """
    Convert a NumPy array or PyTorch Tensor to a float32 PyTorch Tensor.
    For NumPy input, it adds a channel dimension and converts to (C, H, W).
    """

    def __init__(self, keys: List[str] = ["image"]):
        super().__init__(keys)

    def __call__(self, sample: Dict[str, Union[np.ndarray, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        for key in self.keys:
            if key in sample:
                image = sample[key]
                if isinstance(image, np.ndarray):
                    if image.ndim == 2:
                        image = image[np.newaxis, :, :]  # (C, H, W)
                    elif image.ndim == 3 and image.shape[0] != 1:
                        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
                    image = torch.from_numpy(image).float()
                elif isinstance(image, torch.Tensor):
                    image = image.float()
                else:
                    raise TypeError("Input image must be a NumPy array or PyTorch tensor.")
                sample[key] = image
        return sample



class Normalization(BaseTransformation):
    def __init__(self, keys: List[str], contiguous=False, exemption: List[str] = ()):
        self.contiguous = contiguous
        self.exemption = exemption
        super().__init__(keys)

    def __call__(self, input_dict: Dict[str, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]):
        for k in self.keys:
            if k in self.exemption:
                continue
            if k in input_dict and input_dict[k] is not None:
                arr = input_dict[k]
                min_val = arr.min()
                max_val = arr.max()
                normed = (arr - min_val) / (max_val - min_val + 1e-8)
                input_dict[k] = normed * 2.0 - 1.0
        return input_dict
