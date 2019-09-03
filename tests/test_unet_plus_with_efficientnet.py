import torch

import segmentation_models_pytorch as smp


def test_efficient_net_with_unetplus():
    ENCODER = 'efficientnet-b0'

    ACTIVATION = None
    model = smp.UnetPlus(
        encoder_name=ENCODER,
        encoder_weights=None,
        classes=4,
        activation=ACTIVATION,
    )
    x = torch.rand((10, 3, 256, 256))
    assert model(x).shape == (10, 4, 256, 256)


if __name__ == '__main__':
    test_efficient_net_with_unetplus()
