import torch.nn as nn


class CNN2DAudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = []

        def build_conv_layer(in_d, out_d, ks, stride, pad):
            conv1 = nn.Conv2d(in_d, out_d,
                              kernel_size=(ks, ks),
                              stride=(stride, stride),
                              padding=(pad, pad))
            relu1 = nn.ReLU()
            bn1 = nn.BatchNorm2d(out_d)
            nn.init.kaiming_normal_(conv1.weight, a=0.1)
            conv1.bias.data.zero_()
            return [conv1, relu1, bn1]

        self.conv_layers += build_conv_layer(1, 8, 5, 2, 2)
        self.conv_layers += build_conv_layer(8, 16, 3, 2, 1)
        self.conv_layers += build_conv_layer(16, 32, 3, 2, 1)
        self.conv_layers += build_conv_layer(32, 64, 3, 2, 1)

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=10)

        self.conv = nn.Sequential(*self.conv_layers)

    def forward(self, x):
        x = self.conv(x)

        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        return self.lin(x)
