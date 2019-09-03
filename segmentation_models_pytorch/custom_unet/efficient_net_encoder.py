import torch
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import load_pretrained_weights, get_model_params, relu_fn


class EfficientNetEncoder(EfficientNet):

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
        return self.extract_features(inputs)


if __name__ == '__main__':
    model_name = 'efficientnet-b0'

    blocks_args, global_params = get_model_params(model_name, override_params={})

    model = EfficientNetEncoder(blocks_args, global_params)
    load_pretrained_weights(model, model_name=model_name, load_fc=True)

    x = torch.rand((11, 3, 128, 128))
    print([block.shape for block in model(x)])
