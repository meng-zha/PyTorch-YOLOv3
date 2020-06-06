import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import cv2
import xml.etree.ElementTree as ED
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, root_path, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        
        self.root_path=root_path
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def load_images(self,idx):
        image_name = self.img_ids[idx]+'.jpg'
        image_path = os.path.join(self.root_path,image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def load_labels(self,idx):
        annotations = np.zeros((0, 5))

        label_name = self.img_ids[idx]+'.xml'
        label_path = os.path.join(self.root_path,label_name)
        tree = ED.ElementTree(file=label_path)
        objects = tree.findall('object')

        for obj in objects:
            bndbox = obj.find('bndbox')
            annotation = np.zeros((1,5))
            annotation[0,0] = self.class2num(str(obj.find('name').text))
            annotation[0,1] = int(bndbox.find('xmin').text)
            annotation[0,2] = int(bndbox.find('ymin').text)
            annotation[0,3] = int(bndbox.find('xmax').text)
            annotation[0,4] = int(bndbox.find('ymax').text)

            annotations = np.append(annotations,annotation,axis=0)

        return annotations

    def class2num(self,cl):
        c2n = {'face':0,'face_mask':1}
        return c2n[cl]

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(self.load_images(index))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        targets = None
        boxes = torch.from_numpy(self.load_labels(index))
        # Adjust for added padding
        x1,y1,x2,y2 = boxes[:,1],boxes[:,2],boxes[:,3],boxes[:,4]
        x1 += pad[0]
        y1 += pad[2]
        x2 += pad[1]
        y2 += pad[3]
        # Returns (x, y, w, h)
        boxes[:, 1] = ((x1 + x2) / 2) / padded_w
        boxes[:, 2] = ((y1 + y2) / 2) / padded_h
        boxes[:, 3] = (x2-x1)/padded_w
        boxes[:, 4] = (y2-y1)/padded_h

        targets = torch.zeros((len(boxes), 6))
        targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return _, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_ids)
