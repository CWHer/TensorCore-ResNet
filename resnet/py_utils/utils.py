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
