import numpy as np

from paddle.vision.transforms import RandomHorizontalFlip, RandomResizedCrop, Compose, BrightnessTransform, \
    ContrastTransform, RandomCrop, Normalize, RandomRotation


class ToArray(object):
    """Convert a ``PIL.Image`` to ``numpy.ndarray``.
    Converts a PIL.Image or numpy.ndarray (H x W x C) to a paddle.Tensor of shape (C x H x W).
    If input is a grayscale image (H x W), it will be converted to a image of shape (H x W x 1). 
    And the shape of output tensor will be (1 x H x W).
    If you want to keep the shape of output tensor as (H x W x C), you can set data_format = ``HWC`` .
    Converts a PIL.Image or numpy.ndarray in the range [0, 255] to a paddle.Tensor in the 
    range [0.0, 1.0] if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, 
    RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8.
    In the other cases, tensors are returned without scaling.
    """
    def __call__(self, img):
        img = np.array(img)
        img = np.transpose(img, [2, 0, 1])
        img = img / 255.
        return img.astype('float32')


class RandomApply(object):
    """Random apply a transform"""
    def __init__(self, transform, p=0.5):
        super().__init__()
        self.p = p
        self.transform = transform

    def __call__(self, img):
        if self.p < np.random.rand():
            return img
        img = self.transform(img)
        return img


def build_transform():
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.1942, 0.1918, 0.1958]
    train_transforms = Compose([
        RandomCrop(32, padding=4),
        ContrastTransform(0.1),
        BrightnessTransform(0.1),
        RandomHorizontalFlip(),
        RandomRotation(15),
        ToArray(),
        Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_transforms = Compose([ToArray(), Normalize(CIFAR_MEAN, CIFAR_STD)])
    return train_transforms, test_transforms