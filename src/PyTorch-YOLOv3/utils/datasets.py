import glob
import random
import os
import numpy as np
from PIL import Image
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
    def __init__(self, data_path, img_size=416, augment=True, multiscale=True, normalized_labels=False):
        # with open(list_path, "r") as file:
        #     self.img_files = file.readlines()

        # self.label_files = [
        #     path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
        #     for path in self.img_files
        # ]
        
        with open(data_path, 'r') as data_file:
            lines = data_file.readlines()
            self.img_files = [line.split(" ")[0] for line in lines]
            self.label_files = [[int(label) for label in line.split(" ")[1].split(",")] for line in lines]
        
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

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

        img_path = self.img_files[index % len(self.img_files)]
        
        labels = self.label_files[index % len(self.label_files)]
        
        class_label = labels[4]
        x_min = labels[0]
        y_min = labels[1]
        x_max = labels[2]
        y_max = labels[3]
                
        # Adjust for added padding
        # pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
        x_min += pad[0]
        y_min += pad[2]
        x_max += pad[1]
        y_max += pad[3]
        
        x = ((x_min + x_max) / 2) / padded_w
        if x_max >= padded_w:
            return None, None, None
            # x = ((x_min + x_max - 1) / 2) / padded_w
            # print("X: ", x)
            # print("Width of image: ", (x_max - x_min))
        y = ((y_min + y_max) / 2) / padded_h
        if y_max >= padded_h:
            return None, None, None
            # y = ((y_min + y_max - 1) / 2) / padded_h
            # print("Y: ", y)
            # print("Height of image: ", (y_max - y_min))
        w = (x_max - x_min) / padded_w
        h = (y_max - y_min) / padded_h
        
        # Numerical stability
        # if x + w/2 >= 1:
        #     w -= (x + w/2) - 1 + 1e-16
        # if y + h/2 >= 1:
        #     h -= (y + h/2) - 1 + 1e-16
                
        targets = torch.tensor([[0, class_label, x, y, w, h]], dtype=torch.float)
        
        
        # print("X: ", x)
        # print("Y: ", y)
        # print("W: ", w)
        # print("H: ", h)
        
        # print("Far X: ", x + w/2)
        # print("Far y: ", y + h/2)
        
        # assert x + w/2 < 1
        # assert x - w/2 > 0
        # assert y + h/2 < 1
        # assert y - h/2 > 0 
        
        
        # Improting Image class from PIL module 
        
          
        # Cropped image of above dimension 
        # (It will not change orginal image) 
        
        # img = (img.numpy() * 255)[0]
        # print(img)
        # img = Image.fromarray(img)
        # print(img)
        # cropped_image = img.crop((x*padded_w - (w*padded_w) / 2, y*padded_h - (h*padded_h) / 2, x*padded_w + (w*padded_w) / 2, y*padded_h + (h*padded_h) / 2)) 
        # # Shows the image in image viewer 
        # cropped_image.show() 
        
        # assert False


        # targets = None
        # if os.path.exists(label_path):
        #     boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
        #     # Extract coordinates for unpadded + unscaled image
        #     x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
        #     y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
        #     x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
        #     y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
        #     # Adjust for added padding
        #     x1 += pad[0]
        #     y1 += pad[2]
        #     x2 += pad[1]
        #     y2 += pad[3]
        #     # Returns (x, y, w, h)
        #     boxes[:, 1] = ((x1 + x2) / 2) / padded_w
        #     boxes[:, 2] = ((y1 + y2) / 2) / padded_h
        #     boxes[:, 3] *= w_factor / padded_w
        #     boxes[:, 4] *= h_factor / padded_h

        #     targets = torch.zeros((len(boxes), 6))
        #     targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        
        first_batch = list(zip(*batch))
        
        batch = filter(lambda elems: elems[0] is not None, batch)
        
        zipped_batch = list(zip(*batch))
        
        # Make sure that all the elems in batch are not None
        while len(zipped_batch) == 0:
            new_batch = []
            for i in range(len(first_batch)):
                rand_i = random.randrange(self.__len__())
                new_batch.append(self.__getitem__(rand_i))
            zipped_batch = list(zip(*new_batch))
                    
        paths, imgs, targets = zipped_batch
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
        return len(self.img_files)
