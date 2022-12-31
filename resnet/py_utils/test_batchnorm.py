import os

import torch
import torch.nn as nn
from utils import randomInt, randomTensor, writeTensor


def makeBatchNormTests(directory: str):
    print("Making batchnorm tests")

    seed = 0
    generator = randomInt(5, 25, seed=seed)

    # fix torch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    batch_size = 23
    num_features = 3
    height = 117
    width = 16

    x_tensor = randomTensor(
        (batch_size, num_features, height, width), generator)
    writeTensor(x_tensor, os.path.join(directory, "test_batchnorm_x.bin"))

    net = nn.BatchNorm2d(num_features=num_features)
    state_dict = net.state_dict()
    for name, tensor in state_dict.items():
        if name not in ["num_batches_tracked"]:
            print("generate {}".format(name))
            new_tensor = randomTensor(tensor.shape, generator)
            state_dict[name] = new_tensor
            writeTensor(new_tensor, os.path.join(
                directory, f"test_batchnorm_bn_{name}.bin"))
    net.load_state_dict(state_dict)

    net.eval()
    with torch.no_grad():
        y = net(x_tensor)
    # print(y)
    writeTensor(y.detach(), os.path.join(directory, "test_batchnorm_y.bin"))
