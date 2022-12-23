from typing import Iterable

import torch
import torch.nn as nn


def randomInt(min: int, max: int, seed: int = 0,
              a=131, c=1031, m=2147483647) -> Iterable[int]:
    seed = (a * seed + c) % m
    while True:
        yield seed % (max - min) + min
        seed = (a * seed + c) % m


def randomTensor(shape: Iterable[int], generator) -> torch.Tensor:
    tensor_size = torch.prod(torch.tensor(shape)).item()
    raw_data = [next(generator) for _ in range(tensor_size)]
    ret_data = list(map(lambda x: x/10, raw_data))
    return torch.FloatTensor(ret_data).view(*shape)


def writeTensor(tensor: torch.Tensor, filename: str):
    # NOTE: content,
    #   1KB header (int_32 shape)
    #   indefinite data (float_32)
    DATA_OFFSET = 1024

    import numpy as np
    shape = np.array(tensor.shape, dtype=np.int32)
    assert shape.size * shape.itemsize < DATA_OFFSET

    tensor = tensor.to(torch.float32).contiguous()
    assert tensor.is_contiguous()

    with open(filename, "wb") as f:
        f.write(shape.tobytes())
        f.seek(DATA_OFFSET)
        f.write(tensor.numpy().tobytes())


if __name__ == "__main__":
    seed = 0
    generator = randomInt(5, 25, seed=seed)

    # fix torch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    batch_size = 2
    num_features = 3
    height = 224
    width = 224

    x_tensor = randomTensor(
        (batch_size, num_features, height, width), generator)
    writeTensor(x_tensor, "test_batchnorm_x.bin")

    net = nn.BatchNorm2d(num_features=num_features)
    state_dict = net.state_dict()
    for name, tensor in state_dict.items():
        if name not in ["num_batches_tracked"]:
            print("generate {}".format(name))
            new_tensor = randomTensor(tensor.shape, generator)
            state_dict[name] = new_tensor
            writeTensor(new_tensor, "test_batchnorm_bn_{}.bin".format(name))
    net.load_state_dict(state_dict)

    net.eval()
    with torch.no_grad():
        y = net(x_tensor)
    print(y)
    writeTensor(y.detach(), "test_batchnorm_y.bin")
