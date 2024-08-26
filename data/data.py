from pathlib import Path

import cv2
import numpy as np
import yaml

rng = np.random.default_rng(37)

class MetaDataset:
    def __init__(self, mode="train", config_file=Path(__file__).parent/"config.yaml") -> None:
        self.config_file = config_file
        self.config = yaml.safe_load(open(self.config_file, "r"))

        self.root_data_dir = Path(self.config["root_path"])
        self.n_ways = self.config["n_ways"]
        self.k_shots = self.config["k_shots"]
        self.meta_test_per_class = self.config["meta_test_per_class"]
        self.mode = mode

        if self.mode == "train":
            self.data_dir = self.root_data_dir / self.config["train_dir_name"]
        elif self.mode == "val":
            self.data_dir = self.root_data_dir / self.config["val_dir_name"]
        else:
            raise ValueError("Invalid mode. Must be 'train' or 'val'.")
        
        all_images_list = list(self.data_dir.rglob("*.png"))
        self.images_dict = {}
        for image_path in all_images_list:
            character_name = f"{image_path.parents[1].name}_{image_path.parent.name}" 
            if character_name not in self.images_dict:
                self.images_dict[character_name] = []
            self.images_dict[character_name].append(image_path)
        self.character_names = list(self.images_dict.keys())

    def __len__(self):
        return len(self.character_names)

    def sample_task(self):
        selected_character_indices = rng.choice(len(self.character_names), self.n_ways, replace=False)
        train_image_paths, val_image_paths = [], []
        train_labels, val_labels = [], []

        for pseudo_cls, char_idx in enumerate(selected_character_indices):
            character_name = self.character_names[char_idx]
            if self.mode == "train":
                meta_test_per_class = self.meta_test_per_class
            else:
                meta_test_per_class = len(self.images_dict[character_name]) - self.k_shots
            selected_image_indices = rng.choice(len(self.images_dict[character_name]), self.k_shots + meta_test_per_class, replace=False)
            train_image_paths.extend([self.images_dict[character_name][idx] for idx in selected_image_indices[:self.k_shots]])
            val_image_paths.extend([self.images_dict[character_name][idx] for idx in selected_image_indices[self.k_shots:]])
            train_labels.extend([pseudo_cls] * self.k_shots)
            val_labels.extend([pseudo_cls] * meta_test_per_class)
        return Dataset(train_image_paths, train_labels, mode="train"), Dataset(val_image_paths, val_labels, mode="test")


class Dataset:
    def __init__(self, images, labels, mode) -> None:
        self.images = images
        self.labels = labels
        self.mode = mode
    
    def transform(self, gray_image: np.ndarray):
        return gray_image
        
    def __load_image(self, image_path):
        print(image_path)
        gray_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        gray_image = cv2.resize(gray_image, (28, 28))
        # gray_image = gray_image.astype(np.float32) / 255.0
        gray_image = np.expand_dims(gray_image, -1)
        return self.transform(gray_image)

    def sample(self):
        images = np.array([self.__load_image(image_path) for image_path in self.images])
        labels = np.array(self.labels)
        return images, labels