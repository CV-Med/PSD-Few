"""Few-shot dataset and episode sampler for PSD-Few.

Images are organized as ``root_dir/<split>/<class>/*.img``. When a split folder
is absent, classes are partitioned into train/val/test at the category level in
a 7:2:1 ratio. The DEIP physics enhancement is applied inside the model during
meta-training rather than in the data loader.
"""

import os
import random
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset

from data.transforms import build_train_transform, build_val_transform


_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')


class FewShotDataset(Dataset):
    """Category-level few-shot dataset."""

    def __init__(self, root_dir, split="train", image_size=84,
                 split_ratio=(0.7, 0.2, 0.1), seed=42):
        """Load the class folders for the split and set up its transforms."""
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.split_ratio = split_ratio
        self.seed = seed

        self.data_by_class = defaultdict(list)
        self._load(split)
        self.classes = sorted(self.data_by_class.keys())
        if len(self.classes) == 0:
            raise ValueError(f"No classes found for split '{split}' under {root_dir}")

        self.transform = (build_train_transform(image_size) if split == "train"
                          else build_val_transform(image_size))

    def _load(self, split):
        """Collect images per class, using the split folder or a 7:2:1 partition."""
        split_dir = os.path.join(self.root_dir, split)
        if os.path.isdir(split_dir):
            self._load_from_folder(split_dir)
            return

        class_folders = sorted(
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d)))
        if not class_folders:
            raise ValueError(f"No class folders found under {self.root_dir}")
        n_train = int(len(class_folders) * self.split_ratio[0])
        n_val = int(len(class_folders) * self.split_ratio[1])
        shuffled = class_folders[:]
        random.Random(self.seed).shuffle(shuffled)
        if split == "train":
            selected = shuffled[:n_train]
        elif split == "val":
            selected = shuffled[n_train:n_train + n_val]
        else:
            selected = shuffled[n_train + n_val:]
        for cls in selected:
            self._load_class(cls, os.path.join(self.root_dir, cls))

    def _load_from_folder(self, split_dir):
        """Register every class sub-folder of a pre-split directory."""
        for class_name in sorted(os.listdir(split_dir)):
            class_path = os.path.join(split_dir, class_name)
            if os.path.isdir(class_path):
                self._load_class(class_name, class_path)

    def _load_class(self, class_name, class_path):
        """Collect the image paths of one class."""
        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(_IMAGE_EXTENSIONS)]
        for img_name in images:
            self.data_by_class[class_name].append(os.path.join(class_path, img_name))

    def __len__(self):
        return sum(len(paths) for paths in self.data_by_class.values())

    def __getitem__(self, idx):
        """Load and transform the image at the given index."""
        all_data = [(p, c) for c, paths in self.data_by_class.items() for p in paths]
        path, class_name = all_data[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), class_name

    def get_class_samples(self, class_name, num_samples):
        """Sample ``num_samples`` transformed images of a class.

        Uses random sampling without replacement, falling back to sampling with
        replacement when the class holds fewer images than requested.
        """
        paths = self.data_by_class[class_name]
        if len(paths) < num_samples:
            selected = random.choices(paths, k=num_samples)
        else:
            selected = random.sample(paths, num_samples)

        images = []
        for path in selected:
            try:
                img = Image.open(path).convert("RGB")
                images.append(self.transform(img))
            except Exception:
                continue

        if len(images) == 0:
            raise ValueError(f"Class '{class_name}' has no valid images")
        return torch.stack(images)


class EpisodeSampler:
    """Samples N-way K-shot episodes with a query set per class."""

    def __init__(self, dataset, n_way, k_shot, n_query, n_episodes):
        """Store the dataset and the episode configuration."""
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        self.classes = dataset.classes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        """Yield one episode at a time.

        Each episode draws ``n_way`` classes and returns the support images and
        labels together with the query images and labels.
        """
        for _ in range(self.n_episodes):
            episode_classes = random.sample(self.classes, self.n_way)

            support_data, support_labels = [], []
            query_data, query_labels = [], []

            for i, cls in enumerate(episode_classes):
                samples = self.dataset.get_class_samples(cls, self.k_shot + self.n_query)

                support_data.append(samples[:self.k_shot])
                support_labels.extend([i] * self.k_shot)

                query_data.append(samples[self.k_shot:])
                query_labels.extend([i] * self.n_query)

            support_images = torch.cat(support_data, dim=0)
            query_images = torch.cat(query_data, dim=0)
            yield (support_images, torch.tensor(support_labels),
                   query_images, torch.tensor(query_labels))
