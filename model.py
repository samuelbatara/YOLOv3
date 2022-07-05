import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not use_bn, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels) if use_bn else None
        self.leaky = nn.LeakyReLU(0.1) 
    
    def forward(self, x):
        if(self.batchnorm):
            return self.leaky(self.batchnorm(self.conv(x)))
        
        return self.leaky(self.conv(x))

class DarknetResidual(nn.Module):
    def __init__(self, in_channels, N, **kwargs):
        super().__init__()
        self.num_repeats = N
        self.layers = nn.ModuleList()
        
        for _ in range(N):
            self.layers += [
                nn.Sequential(
                    CNNBlock(in_channels, in_channels//2, kernel_size=1),
                    CNNBlock(in_channels//2, in_channels, kernel_size=3, padding=1),
                )
            ]

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)

        return x

class Prediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.layer = nn.Sequential(
                CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
                CNNBlock(2 * in_channels, 3 * (num_classes + 5), kernel_size=1, use_bn=False),
            ) 

    def forward(self, x):
        return self.layer(x)\
            .reshape(x.shape[0], 3, 5 + self.num_classes, x.shape[2], x.shape[3])\
                .permute(0, 1, 3, 4, 2)

configs = [
    (32, 3, 1), # (filters, kernel_size, stride)
    (64, 3, 2), # Dowmsample
    ['R', 1],
    (128, 3, 2), # Downsample
    ['R', 2],
    (256, 3, 2), # Downsample
    ['R', 8],
    (512, 3, 2), # Downsample
    ['R', 8],
    (1024, 3, 2), # Downsample
    ['R', 4],
    (512, 1, 1),
    (1024, 3, 1),
    (512, 1, 1), #P
    (1024, 3, 1),
    (512, 1, 1),
    'P', # output pertama
    (256, 1, 1),
    'U',
    (256, 1, 1),
    (512, 3, 1),
    (256, 1, 1), #P
    (512, 3, 1),
    (256, 1, 1),
    'P', # output kedua
    (128, 1, 1),
    'U',
    (128, 1, 1),
    (256, 3, 1),
    (128, 1, 1), #P
    (256, 3, 1),
    (128, 1, 1),
    'P', # output ketiga
]

class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layers = self._create_layers()

    def forward(self, x):
        layers = self.layers 
        residual_outputs = []
        outputs = []
        for layer in layers:
            if(isinstance(layer, Prediction)):
                outputs.append(
                    layer(x)
                )
                continue

            x = layer(x)

            if(isinstance(layer, DarknetResidual) and layer.num_repeats==8):
                residual_outputs.append(x)
            
            elif(isinstance(layer, nn.Upsample)):
                x = torch.cat([x, residual_outputs[-1]], dim=1)
                residual_outputs.pop(-1)
        
        return outputs

    def _create_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for config in configs:
            if(isinstance(config, tuple)):
                layers.append(
                    CNNBlock(
                        in_channels, 
                        config[0], 
                        kernel_size=config[1],
                        stride=config[2],
                        padding=config[1]//2,
                    )
                )
                in_channels = config[0]

            elif(isinstance(config, list)):
                layers.append(
                    DarknetResidual(in_channels, config[1])
                )

            elif(isinstance(config, str) and config=='P'):
                layers.append(
                    Prediction(in_channels, self.num_classes)
                )

            elif(isinstance(config, str) and config=='U'):
                layers.append(nn.Upsample(scale_factor=2))
                in_channels *= 3
        
        return layers