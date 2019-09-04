import torch
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import load_pretrained_weights, get_model_params, relu_fn


class EfficientNetEncoder(EfficientNet):

    def __init__(self, model_name='efficientnet-b0'):
        blocks_args, global_params = get_model_params(model_name, override_params={})
        super().__init__(blocks_args, global_params)
        load_pretrained_weights(self, model_name=model_name, load_fc=True)
        self.model_name = model_name

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = relu_fn(self._bn0(self._conv_stem(inputs)))

        # Blocks
        self.output_blocks = []
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            self.output_blocks.append(x)
        # Head
        x = relu_fn(self._bn1(self._conv_head(x)))

        return self.output_blocks

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """

        # Convolution layers
        connecting_layers = self.extract_features(inputs)
        connecting_layers = self._get_correct_connecting_layers(connecting_layers)
        return connecting_layers

    def _get_correct_connecting_layers(self, features):
        idxs = {'efficientnet-b0': [0, 2, 4, 10, -1],
                'efficientnet-b1': [1, 4, 7, 15, -1],
                'efficientnet-b2': [1, 4, 7, 15, -1],
                'efficientnet-b3': [1, 4, 7, 17, -1]
                }[self.model_name]
        connecting_layers = [features[idx] for idx in idxs][::-1]
        return connecting_layers



efficientnet_encoders = {
    'efficientnet-b0': {
        'encoder': EfficientNetEncoder,
        'pretrained_settings': {'imagenet': {}},
        'out_shapes': (320, 112, 40, 24, 16),
        'params': {'model_name': 'efficientnet-b0'}  # these are called by instantiation of EfficientNetEncoder
    },
    'efficientnet-b1': {
        'encoder': EfficientNetEncoder,
        'pretrained_settings': {'imagenet': {}},
        'out_shapes': (320, 112, 40, 24, 16),
        'params': {'model_name': 'efficientnet-b1'}  # these are called by instantiation of EfficientNetEncoder
    },
    'efficientnet-b2': {
        'encoder': EfficientNetEncoder,
        'pretrained_settings': {'imagenet': {}},
        'out_shapes': (352, 120, 48, 24, 16),
        'params': {'model_name': 'efficientnet-b1'}  # these are called by instantiation of EfficientNetEncoder
    },
    'efficientnet-b3': {
        'encoder': EfficientNetEncoder,
        'pretrained_settings': {'imagenet': {}},
        'out_shapes': (384, 136, 48, 32, 24),
        'params': {'model_name': 'efficientnet-b3'}  # these are called by instantiation of EfficientNetEncoder
    }
}

if __name__ == '__main__':
    model = EfficientNetEncoder(model_name='efficientnet-b0')

    x = torch.rand((11, 3, 256, 256))
    print([block.shape for block in model(x)])
