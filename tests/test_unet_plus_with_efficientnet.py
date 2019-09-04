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


def test_cuda():
    ENCODER = 'efficientnet-b0'
    ACTIVATION = None

    model = smp.UnetPlus(
        encoder_name=ENCODER,
        encoder_weights=None,
        classes=4,
        activation=ACTIVATION,
    ).cuda()
    x = torch.rand((10, 3, 256, 256)).cuda()
    assert model(x).shape == (10, 4, 256, 256)


def test_fit():
    ENCODER = 'efficientnet-b0'
    ACTIVATION = None

    model = smp.UnetPlus(
        encoder_name=ENCODER,
        encoder_weights=None,
        classes=4,
        activation=ACTIVATION,
    ).cuda()
    x = torch.rand((10, 3, 256, 256)).cuda()

    loss = torch.nn.L1Loss()
    y_pred = model(x)
    y_true = torch.rand(10, 4, 256, 256).cuda()
    output = loss(y_true, y_pred)
    output.backward()


if __name__ == '__main__':
    test_efficient_net_with_unetplus()
    test_cuda()
    test_fit()
