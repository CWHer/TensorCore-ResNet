import os

import argparse
import numpy as np
import torch
import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models, transforms


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist")

        self.path = path
        self.image_names = os.listdir(path)
        self.transform = transform
        print(self.image_names[:10])

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(os.path.join(self.path, image_name)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        # HACK: FIXME: dummy label
        return image, 0

    def __len__(self):
        return len(self.image_names)


def writeTensor(tensor: torch.Tensor, filename: str):
    # NOTE: content,
    #   1KB header (int_32 shape)
    #   indefinite data (float_32)
    DATA_OFFSET = 1024

    shape = np.array(tensor.shape, dtype=np.int32)
    assert shape.size * shape.itemsize < DATA_OFFSET

    tensor = tensor.to(torch.float32).contiguous()
    assert tensor.is_contiguous()

    with open(filename, "wb") as f:
        f.write(shape.tobytes())
        f.seek(DATA_OFFSET)
        f.write(tensor.numpy().tobytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument("--output-dir", type=str, default="dataset_tensor")
    args = parser.parse_args()

    data_mean = [0.485, 0.456, 0.406]
    data_std = [0.229, 0.224, 0.225]
    input_shape = (3, 224, 224)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_mean, std=data_std)
    ])

    path = os.path.join(os.getcwd(), args.dataset_dir)
    dataset = ImageDataset(path, transform)

    data_loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers)

    output_dir = os.path.join(os.getcwd(), args.output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with tqdm.tqdm(total=len(dataset)) as pbar:
        for i, (images, labels) in enumerate(data_loader):
            writeTensor(images, os.path.join(
                output_dir, f"images_{i:04d}.bin"))
            writeTensor(labels,os.path.join(
                output_dir, f"labels_{i:04d}.bin"))
            pbar.update(labels.size(0))
