import os

import torch
import torch.nn as nn
from utils import randomInt, randomTensor, writeTensor


def makeIm2colTests(directory: str):
    print("Making im2col tests")

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
    writeTensor(x_tensor, os.path.join(directory, "test_im2col_x.bin"))

    net = nn.Unfold(kernel_size=3, stride=2, padding=1)

    net.eval()
    with torch.no_grad():
        y = net(x_tensor)
    # print(y)
    writeTensor(y.detach(), os.path.join(directory, "test_im2col_y.bin"))
