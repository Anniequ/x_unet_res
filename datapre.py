# _*_ coding: utf-8 _*_
# @author: anniequ
# @file: datapre.py
# @time: 2020/11/12 11:07
# @Software: PyCharm

import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
import torchvision.models as models
import albumentations as albu
import cv2 as cv

voc_root = r'D:\project\data_qjp\406\MYVOC224224aug'
np.seterr(divide='ignore', invalid='ignore')


# 读取图片
def read_img(root=voc_root, train=True):
    txt_frame = root + '/ImageSets/Segmentation/' + ('train.txt' if train else 'val.txt')

    with open(txt_frame, 'r') as f:
        images = f.read().split()
    data = [os.path.join(root, 'JPEGImages', i + '.jpg') for i in images]
    label = [os.path.join(root, 'SegmentationClass', i + '.png') for i in images]
    return data, label



# 图片大小不同，同时裁剪data and label
def crop(data, label, height, width):
    'data and label both are Image object'
    box = (0, 0, width, height)
    data = data.crop(box)
    label = label.crop(box)
    return data, label


# VOC数据集中对应的标签
classes = ['background', 'tape', 'scissors', 'nailpolish', 'lighter']

# 各种标签所对应的颜色
colormap = [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128], [128, 0, 0]]
cm2lbl = np.zeros(256 ** 3)

# 枚举的时候i是下标，cm是一个三元组，分别标记了RGB值
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


# 将标签按照RGB值填入对应类别的下标信息
def image2label(im):
    data = np.array(im, dtype="int32")
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype="int64")





def image_transforms(data, label):
    # 将数据转换成tensor，并且做标准化处理
    im_tfs = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data = im_tfs(data)
    label = image2label(label)
    label = torch.from_numpy(label)
    return data, label


class VOCSegDataset(torch.utils.data.Dataset):

    # 构造函数
    def __init__(self, train, height=224, width=224, augmentation=None,transforms=image_transforms):
        self.height = height
        self.width = width
        self.fnum = 0  # 用来记录被过滤的图片数
        self.augmentation = augmentation
        self.transforms = transforms
        data_list, label_list = read_img(train=train)
        self.data_list = self._filter(data_list)
        self.label_list = self._filter(label_list)
        if train == True:
            print("训练集：加载了 " + str(len(self.data_list)) + " 张图片和标签" + ",过滤了" + str(self.fnum) + "张图片")
        else:
            print("测试集：加载了 " + str(len(self.data_list)) + " 张图片和标签" + ",过滤了" + str(self.fnum) + "张图片")

    # 过滤掉长小于height和宽小于width的图片
    def _filter(self, images):
        img = []
        for im in images:
            if (Image.open(im).size[1] >= self.height and
                    Image.open(im).size[0] >= self.width):
                img.append(im)
            else:
                self.fnum = self.fnum + 1
        return img

    # 重载getitem函数，使类可以迭代
    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img = cv.imread(img)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        label = cv.imread(label)
        label = cv.cvtColor(label, cv.COLOR_BGR2RGB)

        if self.augmentation:
            sample = self.augmentation(image=img, mask=label)
            img, label = sample['image'], sample['mask']
        img, label = self.transforms(img, label)
        return img,label

    def __len__(self):
        return len(self.data_list)


def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5), #水平翻转

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0), # 平移缩放旋转

        albu.PadIfNeeded(min_height=224, min_width=224, always_apply=True, border_mode=0),  # 加padding
        albu.RandomCrop(height=224, width=224, always_apply=True), # 随机剪裁

        albu.IAAAdditiveGaussianNoise(p=0.2),  # Add gaussian noise to the input image.
        albu.IAAPerspective(p=0.5),  # Perform a random four point perspective transform of the input

        albu.OneOf(
            [
                albu.CLAHE(p=1), # 对比度受限情况下的自适应直方图均衡化算法
                albu.RandomBrightnessContrast(p=1), # Randomly change brightness and contrast
                albu.RandomGamma(p=1), # Gamma变换
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1), # Sharpen the input image and overlays the result with the original image.
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),  # Randomly change hue, saturation and value of the input image.
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()





if __name__ == '__main__':
    voc_train = VOCSegDataset(True, augmentation=get_training_augmentation())
    voc_test = VOCSegDataset(False)

    for i in range(3):
        image, mask = voc_train[1]
        # print(image.shape)
        visualize(image=image, mask=mask)