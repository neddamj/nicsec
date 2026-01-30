import os
import torch
import torchvision.transforms.functional as F

from PIL import Image
from torch.utils import data
from torchvision import transforms
from torch.utils.data import DataLoader

# pip install datasets==2.16.1  # higher version does not work  # huggingface datasets
from datasets import load_dataset


class KodakDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = list(os.listdir(root))
        self.images.sort()

    def __getitem__(self, idx):
        img = Image.open(os.path.join( self.root, self.images[idx] ))
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.zeros(1)

    def __len__(self):
        return len(self.images)

##################################

class WiderFaceDataset:
    def __init__(self, image_size=256, batch_size=1, min_ratio=0.04, max_ratio=0.05, split='validation'):
        """
        Args:
             configuration parameters:
                           - 'min_ratio': float, minimum size of patches (as a ratio of overall image size)
                           - 'max_ratio': float, maximum size of patches (as a ratio of overall image size)
                           - 'image_size': int (for w and h)
                           - 'batch_size': int
                           - 'split': str, 'train' or 'validation', use training or validation image sets
        """
        self.image_size = (image_size, image_size)
        self.batch_size = batch_size
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.split = split

    def _has_small_face(self, example):
        """Filter function to keep images whose face patches have ratios."""
        img = example["image"]
        W, H = img.size
        img_area = W * H
        bboxes = example["faces"]["bbox"]  # [x, y, w, h]

        if len(bboxes) == 0:
            return False

        for x, y, w, h in bboxes:
            face_area = w * h
            ratio = face_area / img_area
            if self.min_ratio <= ratio <= self.max_ratio:
                return True
        return False

    def _resize_and_transform(self, example):
        """Resizes image/bboxes and converts image to Tensor."""
        img = example["image"]
        bboxes = example["faces"]["bbox"]

        orig_w, orig_h = img.size
        new_h, new_w = self.image_size

        # 1. Resize Image
        # F.resize expects (C, H, W) or PIL Image. Returns the same type.
        img_resized = F.resize(img, self.image_size)

        # 2. Resize Bboxes
        sx = new_w / orig_w
        sy = new_h / orig_h

        new_bboxes = []
        for x, y, w, h in bboxes:
            new_bboxes.append([
                x * sx,
                y * sy,
                w * sx,
                h * sy
            ])

        # 3. Update Example
        # Convert image to Tensor (C, H, W) normalized [0, 1]
        example["image"] = transforms.ToTensor()(img_resized)

        # Store bboxes as float tensors initially
        example["faces"]["bbox"] = torch.tensor(new_bboxes, dtype=torch.float32)

        return example

    def _collate_fn(self, batch):
        """Custom collate function for the DataLoader."""
        # Stack images (already tensors from the transform step)
        images = torch.stack([ex["image"] for ex in batch])

        # Collect targets (converting float bboxes to int as per your original code)
        # targets = [torch.tensor(ex["faces"]["bbox"], dtype=torch.int32) for ex in batch]
        # convert bboxes to a list of tensors because different images have different number of boxes
        targets = [ex["faces"]["bbox"].to(torch.int32) for ex in batch]

        return images, targets

    def get_dataloader(self, shuffle=True):
        """Orchestrates the pipeline and returns the DataLoader."""
        # 1. Load Dataset
        ds = load_dataset("wider_face", split=self.split, trust_remote_code=True)

        # 2. Filter
        ds_filtered = ds.filter(self._has_small_face)
        print("Total number of images with desired patches: ", len(ds_filtered))

        # 3. Transform (Resize & Convert)
        # We use .map to apply the transformation to the dataset
        ds_resized = ds_filtered.map(self._resize_and_transform)

        # 4. Create DataLoader
        # Note: We set format to ensure columns are accessible as python objects/tensors
        # often necessary depending on how the map function returns data.
        ds_resized.set_format(type='torch', columns=['image', 'faces'])

        return DataLoader(
            ds_resized,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn
        )

####################################

import os, h5py, torch
from PIL import Image
from torch.utils.data import Dataset

class SVHNFullBBox(Dataset):
    def __init__(self, root, split="train", transform=None, target_transform=None):
        self.root = os.path.join(root, split)  # expects train/test/extra folders from Format 1 tarballs
        self.transform = transform
        self.target_transform = target_transform
        self.records = self._load_digit_struct()

    def _load_digit_struct(self):
        mat = h5py.File(os.path.join(self.root, "digitStruct.mat"), "r")
        names = mat["digitStruct"]["name"]
        bboxes = mat["digitStruct"]["bbox"]
        items = []
        for i in range(len(names)):
            name = "".join(chr(c[0]) for c in mat[names[i][0]][:])
            box_ds = mat[bboxes[i][0]]

            def vals(field):
                v = box_ds[field]
                if len(v) == 1:
                    return [v[0][0]]
                return [box_ds[v[j][0]][0][0] for j in range(len(v))]

            left, top = vals("left"), vals("top")
            width, height = vals("width"), vals("height")
            labels = [int(x if x != 10 else 0) for x in vals("label")]  # 10 == digit 0
            boxes = [[float(l), float(t), float(l + w), float(t + h)] for l, t, w, h in zip(left, top, width, height)]
            items.append({"name": name, "boxes": torch.tensor(boxes), "labels": torch.tensor(labels)})
        mat.close()
        return items

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(os.path.join(self.root, rec["name"])).convert("RGB")
        target = {"boxes": rec["boxes"].float(), "labels": rec["labels"].long(), "image_id": torch.tensor([idx])}
        if self.transform:
            img, target = self.transform(img, target)
        return img, target

def svhn_collate(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)
