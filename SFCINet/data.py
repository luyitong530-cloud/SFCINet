# -*- coding: utf-8 -*-
import os
import random
import numpy as np
from PIL import Image, ImageEnhance
import torch.utils.data as data
import torchvision.transforms as transforms

# -----------------------------------------------------------
# 1. 数据增强函数
# -----------------------------------------------------------
def cv_random_flip(img, label, depth, edge):
    if random.randint(0, 1) == 1:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        edge = edge.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return img, label, depth, edge

def randomCrop(image, label, depth, edge):
    """ 真正的随机裁剪，防止模型产生中心依赖 """
    border = 30
    image_width, image_height = image.size
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    
    x_offset = np.random.randint(0, image_width - crop_win_width + 1)
    y_offset = np.random.randint(0, image_height - crop_win_height + 1)
    
    random_region = (
        x_offset, y_offset,
        x_offset + crop_win_width, y_offset + crop_win_height
    )
    return image.crop(random_region), label.crop(random_region), depth.crop(random_region), edge.crop(random_region)

def randomRotation(image, label, depth, edge):
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, Image.Resampling.BICUBIC)
        depth = depth.rotate(random_angle, Image.Resampling.BICUBIC)
        label = label.rotate(random_angle, Image.Resampling.NEAREST)
        edge = edge.rotate(random_angle, Image.Resampling.NEAREST)
    return image, label, depth, edge

def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

# -----------------------------------------------------------
# 2. 训练数据集类
# -----------------------------------------------------------
class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, depth_root, edge_root, trainsize):
        self.trainsize = trainsize
        self.img_names = [f for f in os.listdir(gt_root) if f.endswith(('.jpg', '.png'))]
        self.image_root = image_root
        self.gt_root = gt_root
        self.depth_root = depth_root
        self.edge_root = edge_root
        self.size = len(self.img_names)
        
        # 统一的图像标准化参数
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), Image.Resampling.LANCZOS),
            transforms.ToTensor(),
            normalize])
            
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), Image.Resampling.NEAREST),
            transforms.ToTensor()])
            
        # 深度图进行与 RGB 完全相同的归一化，以匹配 ImageNet 预训练权重
        self.depths_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), Image.Resampling.LANCZOS), 
            transforms.ToTensor(),
            normalize])

    def __getitem__(self, index):
        name = self.img_names[index]
        
        img_path = os.path.join(self.image_root, name)
        if not os.path.exists(img_path): img_path = img_path.replace('.png', '.jpg')
        
        depth_path = os.path.join(self.depth_root, name)
        if not os.path.exists(depth_path):
            for ext in ['.jpg', '.png', '.bmp']:
                tmp = os.path.join(self.depth_root, name.split('.')[0] + ext)
                if os.path.exists(tmp):
                    depth_path = tmp
                    break

        image = Image.open(img_path).convert('RGB')
        gt = Image.open(os.path.join(self.gt_root, name)).convert('L')
        # 为了匹配 3 通道的归一化，深度图直接读取为 RGB 格式
        depth = Image.open(depth_path).convert('RGB')
        edge = Image.open(os.path.join(self.edge_root, name)).convert('L')

        image, gt, depth, edge = cv_random_flip(image, gt, depth, edge)
        image, gt, depth, edge = randomCrop(image, gt, depth, edge)
        image, gt, depth, edge = randomRotation(image, gt, depth, edge)
        image = colorEnhance(image)

        image = self.img_transform(image)
        depth = self.depths_transform(depth)
        
        # 强行清洗 GT 和 Edge 的脏像素，切断模型输出灰雾的退路
        gt = self.gt_transform(gt)
        gt = (gt > 0.5).float()
        
        edge = self.gt_transform(edge)
        edge = (edge > 0.5).float()
        
        return image, gt, depth, edge

    def __len__(self):
        return self.size

def get_loader(image_root, gt_root, depth_root, edge_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True):
    dataset = SalObjDataset(image_root, gt_root, depth_root, edge_root, trainsize)
    # 【追加 drop_last=True】：防止最后一个残缺 Batch 导致 BatchNorm 报错崩溃
    data_loader = data.DataLoader(dataset=dataset, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    return data_loader

# -----------------------------------------------------------
# 3. 测试数据集类
# -----------------------------------------------------------
class test_dataset:
    def __init__(self, image_root, gt_root, depth_root, testsize):
        self.testsize = testsize
        self.img_list = [f for f in os.listdir(image_root) if f.endswith(('.jpg', '.png'))]
        self.image_root = image_root
        self.gt_root = gt_root
        self.depth_root = depth_root
        
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize), Image.Resampling.LANCZOS),
            transforms.ToTensor(),
            normalize])
            
        self.depths_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize), Image.Resampling.LANCZOS), 
            transforms.ToTensor(),
            normalize])
            
        self.size = len(self.img_list)
        self.index = 0

    def load_data(self):
        name = self.img_list[self.index]
        
        image = Image.open(os.path.join(self.image_root, name)).convert('RGB')
        gt_path = os.path.join(self.gt_root, name.replace('.jpg', '.png'))
        depth_path = os.path.join(self.depth_root, name.replace('.jpg', '.png'))
        
        if not os.path.exists(depth_path):
            depth_path = depth_path.replace('.png', '.jpg')
            if not os.path.exists(depth_path):
                depth_path = depth_path.replace('.jpg', '.bmp')

        gt = Image.open(gt_path).convert('L')
        # 测试集同步转 RGB，保持网络输入通道一致
        depth = Image.open(depth_path).convert('RGB')

        image_tensor = self.transform(image).unsqueeze(0)
        depth_tensor = self.depths_transform(depth).unsqueeze(0)
        
        name_save = name.replace('.jpg', '.png')
        self.index = (self.index + 1) % self.size
        
        return image_tensor, gt, depth_tensor, name_save, np.array(image)