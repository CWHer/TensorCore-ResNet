import os
from typing import Iterable

import numpy as np
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


# Should use < to indicate little endian, otherwise some strange platform problems may occur
def writeTensor(tensor: torch.Tensor, filename: str) -> None:
    # NOTE: content,
    #   1KB header (int_32 aka <i4 shape)
    #   indefinite data (float_32 aka >f4)
    DATA_OFFSET = os.environ.get("DATA_OFFSET", 1024)
    shape = np.array(tensor.shape, dtype=np.dtype("<i4"))
    assert shape.size * shape.itemsize < DATA_OFFSET

    numpy_tensor = tensor.numpy().astype(np.dtype("<f4"))

    with open(filename, "wb") as f:
        f.write(shape.tobytes())
        f.seek(DATA_OFFSET)
        f.write(numpy_tensor.tobytes())


def writeNetwork(network: nn.Module, model_name: str, directory: str) -> None:
    state_dict = network.state_dict()
    for name, tensor in state_dict.items():
        filename = os.path.join(
            directory, model_name + "_" + name.replace('.', '_') + ".bin")
        writeTensor(tensor, filename)
