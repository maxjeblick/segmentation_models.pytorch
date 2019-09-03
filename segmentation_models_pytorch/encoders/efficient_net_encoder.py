import torch
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import load_pretrained_weights, get_model_params, relu_fn


class EfficientNetEncoder(EfficientNet):

    def __init__(self, model_name='efficientnet-b0'):
        blocks_args, global_params = get_model_params(model_name, override_params={})
        super().__init__(blocks_args, global_params)
        load_pretrained_weights(self, model_name=model_name, load_fc=True)

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
        features = self.extract_features(inputs)
        idxs = [0, 2, 4, 10, -1]
        return [features[idx] for idx in idxs][::-1]


if __name__ == '__main__':
    model = EfficientNetEncoder()

    x = torch.rand((11, 3, 256, 256))
    print([block.shape for block in model(x)])
"""
0: torch.Size([11, 16, 128, 128]),  <-
torch.Size([11, 24, 64, 64]),
2: torch.Size([11, 24, 64, 64]), <-
torch.Size([11, 40, 32, 32]), 
4: torch.Size([11, 40, 32, 32]),   <-
torch.Size([11, 80, 16, 16]), 
torch.Size([11, 80, 16, 16]), 
torch.Size([11, 80, 16, 16]), 
torch.Size([11, 112, 16, 16]), 
torch.Size([11, 112, 16, 16]), 
10: torch.Size([11, 112, 16, 16]),   <-
torch.Size([11, 192, 8, 8]), 
torch.Size([11, 192, 8, 8]), 
torch.Size([11, 192, 8, 8]), 
torch.Size([11, 192, 8, 8]), 
-1: torch.Size([11, 320, 8, 8])   <-
"""

efficientnet_encoders = {
    'efficientnet-b0': {
        'encoder': EfficientNetEncoder,
        'pretrained_settings': {'imagenet': {}},
        'out_shapes': (320, 112, 40, 24, 16),
        'params': dict()  # these are called by instantiation of EfficientNetEncoder
    }
}
