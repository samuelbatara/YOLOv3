import torch
import torch.nn as nn

"""
Tuple: (filters, kernel_size, stride)
Jika filters = -1, maka filters = 3 * (num_classes + 5)
"""
config = [ # input: 416 x 416
    (32, 3, 1), # output: 416 x 416
    (64, 3, 2), # output: 208 x 208
    [
        "Residual",
        [(32, 1, 1), (64, 3, 1)],
        1
    ], # output: 208 x 208
    (128, 3, 2), # output: 104 x 104
    [
        "Residual",
        [(64, 1, 1), (128, 3, 1)],
        2
    ], # output: 104 x 104
    (256, 3, 2), # output: 52 x 52
    [
        "Residual",
        [(128, 1, 1), (256, 3, 1)],
        8
    ], # output: 52 x 52
    (512, 3, 2), # output: 26 x 26
    [
        "Residual",
        [(256, 1, 1), (512, 3, 1)],
        8
    ], # output: 26 x 26
    (1024, 3, 2), # output: 13 x 13
    [
        "Residual",
        [(512, 1, 1), (1024, 3, 1)],
        4
    ], # output: 13 x 13
    (512, 1, 1), # output: 13 x 13
    (1024, 3, 1), # output: 13 x 13
    (512, 1, 1), # output: 13 x 13
    (1024, 3, 1), # output: 13 x 13
    (512, 1, 1), # output: 13 x 13
    [
        "Output",
        [(1024, 3, 1),(-1, 1, 1)] 
    ],
    (256, 1, 1), # output: 13 x 13
    "Upsample", # output: 26 x 26
    (256, 1, 1), # output: 26 x 26
    (512, 3, 1), # output: 26 x 26
    (256, 1, 1), # output: 26 x 26
    (512, 3, 1), # output: 26 x 26
    (256, 1, 1), # output: 26 x 26
    [
        "Output",
        [(512, 3, 1), (-1, 1, 1)]
    ],
    (128, 1, 1), # output: 26 x 26
    "Upsample", # output: 52 x 52
    (128, 1, 1), # output: 52 x 52
    (256, 3, 1), # output: 52 x 52
    (128, 1, 1), # output: 52 x 52
    (256, 3, 1), # output: 52 x 52
    (128, 1, 1), # output: 52 x 52
    [
        "Output",
        [(256, 3, 1), (-1, 1, 1)]
    ]
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels,
            bias=False,
            **kwargs,
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky(self.batchnorm(self.conv(x)))

class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self._layers = self._create_layers()

    def forward(self, x):
        layers = self._layers
        outputs = []
        save_outputs = []

        print("Input:", x.shape)
        for layer in layers:
            tmp_x = x
            if(isinstance(layer, CNNBlock)):
                print("After Conv:", end="\t")
                x = layer(x)

            elif(isinstance(layer, list) and layer[0] == "Residual"):
                print("After Residual:", end="\t")
                for _ in range(layer[2]):
                    xx = x 
                    for conv in layer[1]:
                        x = conv(x)
                    x += xx 
                if(int(layer[2]) == 8):
                    save_outputs.append(x)
            
            elif(isinstance(layer, list) and layer[0] == "Output"):
                print("Output: ", end="\t")
                res = x
                for conv in layer[1]:
                    res = conv(res)

                res = res.reshape(
                    res.shape[0], 3, self.num_classes + 5, res.shape[2], res.shape[3] 
                ).permute(0, 1, 3, 4, 2)

                outputs.append(res)
            
            elif(isinstance(layer, nn.Upsample)):
                print("After upsample:", end="\t")
                x = layer(x)
                x = torch.cat((x, save_outputs.pop()), dim=1)

            print(x.shape)

        return outputs

    def _create_layers(self) -> list:
        layers = []
        in_channels = self.in_channels

        for module in config:
            if(isinstance(module, tuple)):
                layers.append(
                    CNNBlock(
                        in_channels=in_channels,
                        out_channels=module[0],
                        kernel_size=module[1],
                        stride=module[2],
                        padding=module[1]//2,
                    )
                )
                in_channels = module[0]

            elif(isinstance(module, list) and module[0] == "Residual"):
                residuals = []
                for _ in range(module[2]):
                    conv1 = module[1][0]
                    conv2 = module[1][1] 
                    residuals.append(
                        CNNBlock(
                            in_channels=in_channels,
                            out_channels=conv1[0],
                            kernel_size=conv1[1],
                            stride=conv1[2],
                            padding=conv1[1]//2,
                        )
                    )
                    residuals.append(
                        CNNBlock(
                            in_channels=conv1[0],
                            out_channels=conv2[0],
                            kernel_size=conv2[1],
                            stride=conv2[2],
                            padding=conv2[1]//2,
                        )
                    )
                    in_channels = conv2[0]
                    
                layers.append(["Residual", residuals, module[2]])

            elif(isinstance(module, list) and module[0] == "Output"):
                outputs = []
                conv1 = module[1][0]
                conv2 = module[1][1]
                outputs.append(
                    CNNBlock(
                        in_channels=in_channels,
                        out_channels=conv1[0],
                        kernel_size=conv1[1],
                        stride=conv1[2],
                        padding=conv1[1]//2,
                    )
                )
                outputs.append(
                    CNNBlock(
                        in_channels=conv1[0],
                        out_channels=3 * (self.num_classes + 5),
                        kernel_size=conv2[1],
                        stride=conv2[2],
                        padding=conv2[1]//2,
                    )
                )

                layers.append(["Output", outputs])

            elif(isinstance(module, str)):
                layers.append(
                    nn.Upsample(scale_factor=2),
                )
                in_channels *= 3

        return layers

if __name__ == "__main__": 
    X = torch.rand(5, 3, 416, 416)
    model = YOLOv3(in_channels=3, num_classes=20)

    outputs = model(X)
    assert outputs[0].shape == (5, 3, 13, 13, 25)
    assert outputs[1].shape == (5, 3, 26, 26, 25)
    assert outputs[2].shape == (5, 3, 52, 52, 25)
    print("Berhasil!")
