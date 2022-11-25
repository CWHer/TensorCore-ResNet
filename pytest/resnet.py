import os

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


class DataPrefetcher():
    def __init__(self, data_loader):
        self.loader = iter(data_loader)
        self.stream = torch.cuda.Stream()
        self.preLoad()

    def preLoad(self):
        try:
            self.next_images, self.next_labels = next(self.loader)
        except StopIteration:
            self.next_images = None
            self.next_labels = None
            return
        with torch.cuda.stream(self.stream):
            self.next_images = self.next_images.cuda(non_blocking=True)
            self.next_labels = self.next_labels.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        images = self.next_images
        labels = self.next_labels
        self.preLoad()
        return images, labels


def writeTensor(tensor: torch.Tensor, filename: str):
    # NOTE: content,
    #   1KB header (int_64 shape)
    #   indefinite data (float_32)
    DATA_OFFSET = 1024

    shape = np.array(tensor.shape, dtype=np.int64)
    assert shape.size * shape.itemsize < DATA_OFFSET

    tensor = tensor.to(torch.float32).contiguous()
    assert tensor.is_contiguous()

    with open(filename, "wb") as f:
        f.write(shape.tobytes())
        f.seek(DATA_OFFSET)
        f.write(tensor.numpy().tobytes())


def writeNetwork(network: torch.nn.Module, directory: str):
    state_dict = network.state_dict()
    for name, tensor in state_dict.items():
        filename = os.path.join(directory, name.replace('.', '_') + ".bin")
        writeTensor(tensor, filename)


if __name__ == "__main__":
    batch_size = 4
    num_workers = 1
    data_mean = [0.485, 0.456, 0.406]
    data_std = [0.229, 0.224, 0.225]
    input_shape = (3, 224, 224)

    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    # writeNetwork(model, "resnet18")

    # model summary
    # import torchstat
    # import torchsummary
    # torchstat.stat(model, (3, 224, 224))
    # torchsummary.summary(model, (3, 224, 224), device="cpu")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_mean, std=data_std)
    ])

    path = os.path.join(os.getcwd(), "dataset")
    dataset = ImageDataset(path, transform)

    data_loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True)
    prefetcher = DataPrefetcher(data_loader)

    model.cuda().eval()
    correct, total = 0, 0

    with torch.no_grad() and tqdm.tqdm(total=len(dataset)) as pbar:
        images, labels = prefetcher.next()
        while images is not None:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.update(labels.size(0))
            images, labels = prefetcher.next()

    print(f"Accuracy on the {total} test images: {100 * correct / total:.2f}%")
