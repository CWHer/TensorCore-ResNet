import torch
import torch.nn as nn
from utils import randomInt, randomTensor, writeTensor


if __name__ == "__main__":
    seed = 0
    generator = randomInt(5, 25, seed=seed)

    # fix torch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # MaxPool2d
    batch_size = 20
    num_features = 64
    height = 112
    width = 112

    x_tensor = randomTensor(
        (batch_size, num_features, height, width), generator)
    writeTensor(x_tensor, "test_maxpool_x.bin")

    net = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    net.eval()
    with torch.no_grad():
        y = net(x_tensor)
    print(y)
    writeTensor(y.detach(), "test_maxpool_y.bin")

    # AvgPool2d
    batch_size = 20
    num_features = 256
    height = 10
    width = 10

    x_tensor = randomTensor(
        (batch_size, num_features, height, width), generator)
    writeTensor(x_tensor, "test_avgpool_x.bin")

    net = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)

    net.eval()
    with torch.no_grad():
        y = net(x_tensor)
    print(y)
    writeTensor(y.detach(), "test_avgpool_y.bin")
