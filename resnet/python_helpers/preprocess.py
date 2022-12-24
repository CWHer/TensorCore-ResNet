import os
from typing import Iterable

import torch
import tqdm
import numpy as np
import argparse
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models, transforms

# Should use < to indicate little endian, otherwise some strange platform problems may occur

#  1KB header (int_32 aka <i4 shape)
DATA_OFFSET = os.environ.get("DATA_OFFSET", 1024)


def writeTensor(tensor: torch.Tensor, filename: str) -> None:
    # NOTE: content,
    #   indefinite data (float_32 aka >f4)
    shape = np.array(tensor.shape, dtype=np.dtype("<i4"))
    assert shape.size * shape.itemsize < DATA_OFFSET

    numpy_tensor = tensor.numpy().astype(np.dtype("<f4"))

    with open(filename, "wb") as f:
        f.write(shape.tobytes())
        f.seek(DATA_OFFSET)
        f.write(numpy_tensor.tobytes())


def writeNetwork(network: torch.nn.Module, directory: str) -> None:
    state_dict = network.state_dict()
    for name, tensor in state_dict.items():
        filename = os.path.join(directory, "resnet18_" + name.replace('.', '_') + ".bin")
        writeTensor(tensor, filename)


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
        # HACK: the label will be generated by the model
        return image, 0

    def __len__(self):
        return len(self.image_names)


def randomInt(min: int, max: int, seed: int = 0, a=131, c=1031, m=2147483647) -> Iterable[int]:
    seed = (a * seed + c) % m
    while True:
        yield seed % (max - min) + min
        seed = (a * seed + c) % m


def randomTensor(shape: Iterable[int], generator) -> torch.Tensor:
    tensor_size = torch.prod(torch.tensor(shape)).item()
    raw_data = [next(generator) for _ in range(tensor_size)]
    ret_data = list(map(lambda x: x / 10, raw_data))
    return torch.FloatTensor(ret_data).view(*shape)


def write_test_batchnorm_files(file_root):
    seed = 0
    generator = randomInt(5, 25, seed=seed)

    # fix torch random seed
    torch.manual_seed(seed)
    # Hack: not using torch CUDA part

    batch_size = 2
    num_features = 3
    height = 224
    width = 224

    x_tensor = randomTensor((batch_size, num_features, height, width), generator)
    writeTensor(x_tensor, os.path.join(file_root, "test_batchnorm_x.bin"))

    net = torch.nn.BatchNorm2d(num_features=num_features)
    state_dict = net.state_dict()
    for name, tensor in state_dict.items():
        if name not in ["num_batches_tracked"]:
            new_tensor = randomTensor(tensor.shape, generator)
            state_dict[name] = new_tensor
            writeTensor(new_tensor, os.path.join(file_root, "test_batchnorm_bn_{}.bin".format(name)))
    net.load_state_dict(state_dict)

    net.eval()
    with torch.no_grad():
        y = net(x_tensor)
    writeTensor(y, os.path.join(file_root, "test_batchnorm_y.bin"))

def write_test_pooling_files(file_root):
    seed = 0
    generator = randomInt(5, 25, seed=seed)

    # fix torch random seed
    torch.manual_seed(seed)
    # Hack: not using torch CUDA part

    # MaxPool2d
    batch_size = 50
    num_features = 64
    height = 112
    width = 112

    x_tensor = randomTensor(
        (batch_size, num_features, height, width), generator)
    writeTensor(x_tensor,os.path.join(file_root, "test_maxpool_x.bin"))

    net = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    net.eval()
    with torch.no_grad():
        y = net(x_tensor)
    writeTensor(y, os.path.join(file_root, "test_maxpool_y.bin"))

    # AvgPool2d
    batch_size = 2
    num_features = 256
    height = 10
    width = 10

    x_tensor = randomTensor(
        (batch_size, num_features, height, width), generator)
    writeTensor(x_tensor, os.path.join(file_root, "test_avgpool_x.bin"))

    net = torch.nn.AvgPool2d(kernel_size=5, stride=2, padding=2)

    net.eval()
    with torch.no_grad():
        y = net(x_tensor)
    writeTensor(y, os.path.join(file_root, "test_avgpool_y.bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument("--tensor-output-dir", type=str, default="dataset_tensor")
    parser.add_argument("--network-output-dir", type=str, default="resnet18")
    parser.add_argument("--test-data-dir", type=str, default="test_data")

    # Check if pytorch supports cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    data_mean = [0.485, 0.456, 0.406]
    data_std = [0.229, 0.224, 0.225]
    input_shape = (3, 224, 224)

    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    writeNetwork(model, args.network_output_dir)

    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                    transforms.Normalize(mean=data_mean, std=data_std)])

    path = args.dataset_dir
    dataset = ImageDataset(path, transform)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    model = model.to(device)
    model.eval()
    correct, total = 0, 0

    # No need to add cwd of already inside them.
    output_dir = args.tensor_output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    write_test_batchnorm_files(args.test_data_dir)
    write_test_pooling_files(args.test_data_dir)

    with torch.no_grad() and tqdm.tqdm(total=len(dataset)) as pbar:
        for i, (images, labels) in enumerate(data_loader):
            writeTensor(images, os.path.join(output_dir, f"images_{i:04d}.bin"))
            image_device = images.to(device)
            outputs = model(image_device)
            _, predicted = torch.max(outputs.data, 1)
            predicted_label = predicted.cpu()
            writeTensor(predicted_label, os.path.join(output_dir, f"labels_{i:04d}.bin"))
            total += labels.size(0)
            # Don't compute the accuracy here. We only check if the implemented result
            # moves just like original model.
            pbar.update(labels.size(0))

            if i < 3:
                # We need to store the first 3 batches to text to test whether loading is correct.
                val_img_filename = os.path.join(output_dir, f"validation_{i:04d}_image.txt")
                with open(val_img_filename, "w") as f:
                    # Write shape of the tensor.
                    shape = images.shape
                    f.write("{} {} ".format(len(shape), " ".join([str(s) for s in shape])))
                    # Write flattened tensor to text file.
                    flattened_images = images.flatten()
                    size = flattened_images.size(0)
                    for j in range(size):
                        f.write(f"{flattened_images[j]} ")

                val_label_filename = os.path.join(output_dir, f"validation_{i:04d}_label.txt")
                with open(val_label_filename, "w") as f:
                    # Write shape of the tensor.
                    shape = labels.shape
                    f.write("{} {} ".format(len(shape), " ".join([str(s) for s in shape])))
                    # Write flattened tensor to text file.
                    flattened_labels = labels.flatten()
                    size = flattened_labels.size(0)
                    for j in range(size):
                        f.write(f"{flattened_labels[j]} ")
