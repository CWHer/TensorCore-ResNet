import os

import torch
import torch.nn as nn
from utils import randomInt, randomTensor, writeTensor


def makeConv2DTests(directory: str):
    print("Making Conv2D tests")

    seed = 0
    generator = randomInt(5, 25, seed=seed)

    # fix torch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    batch_size = 23
    num_features = 7
    height = 117
    width = 191

    x_tensor = randomTensor(
        (batch_size, num_features, height, width), generator) - 1
    writeTensor(x_tensor, os.path.join(directory, "test_conv2d_x.bin"))

    net = nn.Conv2d(in_channels=num_features, out_channels=3,
                    kernel_size=5, stride=2, padding=3, bias=False)
    state_dict = net.state_dict()
    for name, tensor in state_dict.items():
        print("generate {}".format(name))
        new_tensor = randomTensor(tensor.shape, generator) - 1
        state_dict[name] = new_tensor
        writeTensor(new_tensor, os.path.join(
            directory, f"test_conv_conv2d_{name}.bin"))
    net.load_state_dict(state_dict)

    net.eval()
    with torch.no_grad():
        y = net(x_tensor)
    # print(y)
    writeTensor(y.detach(), os.path.join(directory, "test_conv2d_y.bin"))
