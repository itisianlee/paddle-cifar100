import warnings

import paddle.nn as nn
from .builder import classifier

# Paper suggests 0.9997 momentum, for TensorFlow. Equivalent PyTorch momentum is
# 1.0 - tensorflow.
_BN_MOMENTUM = 1 - 0.9997


class _InvertedResidual(nn.Layer):

    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size,
        stride,
        expansion_factor,
        bn_momentum=0.1
    ):
        super(_InvertedResidual, self).__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        mid_ch = in_ch * expansion_factor
        self.apply_residual = (in_ch == out_ch and stride == 1)
        self.layers = nn.Sequential(
            # Pointwise
            nn.Conv2D(in_ch, mid_ch, 1, bias_attr=False),
            nn.BatchNorm2D(mid_ch, momentum=bn_momentum),
            nn.ReLU(),
            # Depthwise
            nn.Conv2D(mid_ch, mid_ch, kernel_size, padding=kernel_size // 2, stride=stride, 
                      groups=mid_ch, bias_attr=False),
            nn.BatchNorm2D(mid_ch, momentum=bn_momentum),
            nn.ReLU(),
            # Linear pointwise. Note that there's no activation.
            nn.Conv2D(mid_ch, out_ch, 1, bias_attr=False),
            nn.BatchNorm2D(out_ch, momentum=bn_momentum))

    def forward(self, input):
        if self.apply_residual:
            return self.layers(input) + input
        else:
            return self.layers(input)


def _stack(in_ch, out_ch, kernel_size, stride, exp_factor, repeats, bn_momentum):
    """ Creates a stack of inverted residuals. """
    assert repeats >= 1
    # First one has no skip, because feature map size changes.
    first = _InvertedResidual(in_ch, out_ch, kernel_size, stride, exp_factor, bn_momentum=bn_momentum)
    remaining = []
    for _ in range(1, repeats):
        remaining.append(_InvertedResidual(out_ch, out_ch, kernel_size, 1, exp_factor, bn_momentum=bn_momentum))
    return nn.Sequential(first, *remaining)


def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(alpha):
    """ Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down. """
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


class MNASNet(nn.Layer):
    """ MNASNet, as described in https://arxiv.org/pdf/1807.11626.pdf. This
    implements the B1 variant of the model.
    >>> model = MNASNet(1.0, num_classes=1000)
    >>> x = paddle.rand(1, 3, 224, 224)
    >>> y = model(x)
    >>> y.dim()
    2
    >>> y.nelement()
    1000
    """
    # Version 2 adds depth scaling in the initial stages of the network.
    _version = 2

    def __init__(self, alpha, num_classes=100, dropout=0.2):
        super(MNASNet, self).__init__()
        assert alpha > 0.0
        self.alpha = alpha
        self.num_classes = num_classes
        depths = _get_depths(alpha)
        layers = [
            # First layer: regular conv.
            nn.Conv2D(3, depths[0], 3, padding=1, stride=1, bias_attr=False),
            nn.BatchNorm2D(depths[0], momentum=_BN_MOMENTUM),
            nn.ReLU(),
            # Depthwise separable, no skip.
            nn.Conv2D(depths[0], depths[0], 3, padding=1, stride=1, groups=depths[0], bias_attr=False),
            nn.BatchNorm2D(depths[0], momentum=_BN_MOMENTUM),
            nn.ReLU(),
            nn.Conv2D(depths[0], depths[1], 1, padding=0, stride=1, bias_attr=False),
            nn.BatchNorm2D(depths[1], momentum=_BN_MOMENTUM),
            # MNASNet blocks: stacks of inverted residuals.
            _stack(depths[1], depths[2], 3, 2, 3, 3, _BN_MOMENTUM),
            _stack(depths[2], depths[3], 5, 1, 3, 3, _BN_MOMENTUM),
            _stack(depths[3], depths[4], 5, 2, 6, 3, _BN_MOMENTUM),
            _stack(depths[4], depths[5], 3, 1, 6, 2, _BN_MOMENTUM),
            _stack(depths[5], depths[6], 5, 2, 6, 4, _BN_MOMENTUM),
            _stack(depths[6], depths[7], 3, 1, 6, 1, _BN_MOMENTUM),
            # Final mapping to classifier input.
            nn.Conv2D(depths[7], 1280, 1, padding=0, stride=1, bias_attr=False),
            nn.BatchNorm2D(1280, momentum=_BN_MOMENTUM),
            nn.ReLU(),
        ]
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(1280, num_classes))

    def forward(self, x):
        x = self.layers(x)
        # Equivalent to global avgpool and removing H and W dimensions.
        x = x.mean([2, 3])
        return self.classifier(x)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, 
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get("version", None)
        assert version in [1, 2]

        if version == 1 and not self.alpha == 1.0:
            # In the initial version of the model (v1), stem was fixed-size.
            # All other layer configurations were the same. This will patch
            # the model so that it's identical to v1. Model with alpha 1.0 is
            # unaffected.
            depths = _get_depths(self.alpha)
            v1_stem = [
                nn.Conv2D(3, 32, 3, padding=1, stride=2, bias_attr=False),
                nn.BatchNorm2D(32, momentum=_BN_MOMENTUM),
                nn.ReLU(),
                nn.Conv2D(32, 32, 3, padding=1, stride=1, groups=32, bias_attr=False),
                nn.BatchNorm2D(32, momentum=_BN_MOMENTUM),
                nn.ReLU(),
                nn.Conv2D(32, 16, 1, padding=0, stride=1, bias_attr=False),
                nn.BatchNorm2D(16, momentum=_BN_MOMENTUM),
                _stack(16, depths[2], 3, 2, 3, 3, _BN_MOMENTUM),
            ]
            for idx, layer in enumerate(v1_stem):
                self.layers[idx] = layer

            # The model is now identical to v1, and must be saved as such.
            self._version = 1
            warnings.warn(
                "A new version of MNASNet model has been implemented. "
                "Your checkpoint was saved using the previous version. "
                "This checkpoint will load and work as before, but "
                "you may want to upgrade by training a newer model or "
                "transfer learning from an updated ImageNet checkpoint.",
                UserWarning)

        super(MNASNet, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


@classifier.register_module()
def mnasnet(alpha=1.0, **kwargs):
    r"""MNASNet with depth multiplier of alpha from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    <https://arxiv.org/pdf/1807.11626.pdf>`_.
    """
    model = MNASNet(alpha=alpha, **kwargs)
    return model