import os

import torch
import torch.nn as nn
from utils import randomInt, randomTensor, writeTensor


def makeLinearTests(directory: str):
    print("Making Linear tests")

    seed = 0
    generator = randomInt(5, 25, seed=seed)

    # fix torch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    batch_size = 31
    in_features = 117
    out_features = 16

    x_tensor = randomTensor((batch_size, in_features), generator)
    writeTensor(x_tensor, os.path.join(directory, "test_linear_x.bin"))

    net = nn.Linear(in_features=in_features, out_features=out_features)
    state_dict = net.state_dict()
    for name, tensor in state_dict.items():
        print("generate {}".format(name))
        new_tensor = randomTensor(tensor.shape, generator)
        state_dict[name] = new_tensor
        writeTensor(new_tensor, os.path.join(
            directory, f"test_linear_fc_{name}.bin"))
    net.load_state_dict(state_dict)

    net.eval()
    with torch.no_grad():
        y = net(x_tensor)
    # print(y)
    writeTensor(y.detach(), os.path.join(directory, "test_linear_y.bin"))
