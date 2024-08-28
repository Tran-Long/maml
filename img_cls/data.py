from pathlib import Path
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import yaml

class CIFAR10Dataset(Dataset):
    def __init__(self, mode="train", config_file=str(Path(__file__).parent/"data_config.yaml")) -> None:
        super().__init__()
        self.config_file = config_file
        self.config = yaml.safe_load(open(self.config_file, "r"))
        self.mode = mode
        if self.mode == "train":
            self.data_dir = Path(self.config["path"])/self.config["train_folder"]
        elif self.mode == "val":
            self.data_dir = Path(self.config["path"])/self.config["val_folder"]
        
        self.image_paths = list(self.data_dir.rglob("*.png"))
        labels = sorted(set([image_path.parent.name for image_path in self.image_paths]))
        self.label2idx = {label: idx for idx, label in enumerate(labels)}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        random.shuffle(self.image_paths)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path), dtype=np.float32) / 255.0
        label = self.label2idx[image_path.parent.name]
        # print(image_path, label)
        return image, label
    
def collate_fn(batch):
    images, labels = zip(*batch)
    images = np.stack(images, axis=0)
    labels = np.array(labels)
    return images, labels