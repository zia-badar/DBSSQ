import pickle
from multiprocessing import Pool
from pathlib import Path

import lmdb
import torch
import torchvision
from PIL import Image
from torch.nn.functional import one_hot
from torchvision.datasets import ImageNet
from torchvision.models import ResNet18_Weights
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Resize, Compose, RandomCrop, ConvertImageDtype, Normalize
from torchvision.transforms.v2.functional import pil_to_tensor
from tqdm import tqdm


class image_net:
    def __init__(self, split: str, lmdb_path: Path = Path('./image_net_lmbd/')):
        lmdb_path = Path.joinpath(lmdb_path, split)
        if split == 'train':
            LMDB_MAP_SIZE = 1024 * 1024 * 1024 * 325
            self.transform = Compose([RandomCrop(size=224), ConvertImageDtype(torch.float), Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        else:
            LMDB_MAP_SIZE = 1024 * 1024 * 1024 * 13  # 325, 13 gb
            self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()

        self.lmdb_env = lmdb.open(str(lmdb_path), map_size=LMDB_MAP_SIZE, readonly=True, lock=False)
        self.lmdb_keys = []
        for i in range(len(self)):
            self.lmdb_keys.append(pickle.dumps(i))

        self.no_classes = len(torchvision.models._meta._IMAGENET_CATEGORIES)

    def __getitem__(self, item):
        with self.lmdb_env.begin(buffers=True) as txn:
            data = pickle.loads(txn.get(self.lmdb_keys[item]))

        data = (torch.tensor(data[0]), torch.tensor(data[1]))

        if data[0].shape[0] == 1:
            data = (data[0].repeat((3, 1, 1)), data[1])
        if data[0].shape[0] == 4:
            data = (data[0][:3], data[1])

        return self.transform(data[0]), one_hot(data[1], num_classes=self.no_classes).to(torch.float)

    def __len__(self):
        return self.lmdb_env.stat()['entries']

#=======================================================================================================================
def read_image(query):
    image_path = query[1][0]
    image_tensor = pil_to_tensor(Image.open(image_path))
    image_tensor = (Resize(size=256, interpolation=InterpolationMode.BILINEAR, antialias=True)(image_tensor))
    return pickle.dumps(query[0]), pickle.dumps((image_tensor.cpu().numpy(), query[1][1]))

# create_lmdb_from_image_net('/home/zia/Downloads/imagenet_data_keep_save/', 'train')
# create_lmdb_from_image_net('/home/zia/Downloads/imagenet_data_keep_save/', 'val')
def create_lmdb_from_image_net(dataset_directory: Path, split: str, lmdb_path = Path('./image_net_lmbd/')):
    dataset = ImageNet(root=dataset_directory, split=split)
    lmdb_path = Path.joinpath(lmdb_path, split)
    lmdb_path.mkdir(parents=True, exist_ok=True)
    LMDB_MAP_SIZE = 1024 * 1024 * 1024 * 325  if split == 'train' else 1024 * 1024 * 1024 * 13 # 325, 13 gb
    env = lmdb.open(str(lmdb_path), map_size=LMDB_MAP_SIZE)
    txn = env.begin(write=True)

    no_of_process = 20
    with Pool(processes=no_of_process) as pool:

        queries = []
        for i, sample in tqdm(enumerate(dataset.samples)):
            queries.append((i, sample))

        for key, value in tqdm(pool.imap_unordered(read_image, queries)):
            txn.put(key, value)

    txn.commit()
    env.close()