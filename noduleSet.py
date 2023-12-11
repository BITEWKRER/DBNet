# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import SimpleITK as sitk
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

from configs import config
from utils.helper import sobel_filter
from utils.logger import logs


class noduleSet(Dataset):
    def __init__(self, lists, mode, transform=None, show=False):
        super(noduleSet, self).__init__()
        self.show = show
        self.transform = transform
        self.lists = lists
        self.mode = mode
        logs(f'{mode[0]},{mode[1]} len: {len(self.lists)}')

    def __len__(self):
        return len(self.lists)

    def __getitem__(self, idx):

        img_name = self.lists[idx].split('.')[-2]

        if config.FileV == 'npy':
            data = np.load(self.lists[idx], allow_pickle=True)
        else:
            data = sitk.ReadImage(self.lists[idx])
            data = sitk.GetArrayFromImage(data)

        img, msk = np.split(data, 2, axis=0)
        ##################################数据增广####################
        if self.transform is not None:  # 图像增强
            img, msk = self.transform(img[:], msk[:])

        #############################################################

        def z_score_norm(img):
            # z-score
            img_mean = np.mean(img)  # torch
            img_std = np.std(img)
            norm_img = (img - img_mean) / img_std
            return norm_img

        def cut_norm(img):
            lungwin = np.array([-1000., 400.])
            newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
            newimg[newimg < 0] = 0
            newimg[newimg > 1] = 1
            return newimg

        def normalize_to_minus_one_one(data):
            # 灰度归一化
            # 归一化至[-1,1]
            max_val = np.max(data)
            min_val = np.min(data)
            return 2 * (data - min_val) / (max_val - min_val) - 1

        def log_transform(volume, c=1.0):
            """
            对数变换方法增强3D体数据对比度
            Args:
                volume (numpy array): 3D输入影像，格式为1x64x64x64
                c (float): 对数变换系数
            Returns:
                numpy array: 增强后的3D影像，格式为1x64x64x64
            """
            volume = volume.astype(np.float32)
            volume = c * np.log(volume + 1.0)
            return volume.astype(np.uint8)

        # 两分支 1
        # 手工特征对比
        img_z = z_score_norm(img)
        img_s = cut_norm(img)
        img_sobel = sobel_filter(img_z[0])

        # 定义Gamma校正值
        gamma_enhanced = 0.6  # 对增强对比度后的图像应用的Gamma校正值
        gamma_edges = 1.5  # 对边缘信息应用的Gamma校正值

        # 对图像应用Gamma校正
        enhanced_gamma_corrected = np.float32(cv2.pow(img_s / 255.0, gamma_enhanced) * 255)
        edges_gamma_corrected = np.float32(cv2.pow(img_sobel / 255.0, gamma_edges) * 255)

        # 融合Gamma校正后的图像
        alpha = 0.5  # 设置融合权重
        beta = 1 - alpha
        img_sobel = cv2.addWeighted(enhanced_gamma_corrected, alpha, edges_gamma_corrected, beta, 0)

        img = np.stack([img_z, img_sobel], axis=1).squeeze(axis=0)
        #	
        # ##################################################
        单分支对比试验
        img = z_score_norm(img)

        if self.show:
            fig, plots = plt.subplots(1, 2)
            if self.mode[1] == '2d':
                plots[0].imshow(img[0], cmap='gray')
                plots[1].imshow(msk[0], cmap='gray')
            else:
                for i in range(img.shape[-1]):
                    plots[0].imshow(img[0, :, :, i], cmap='gray')
                    plots[1].imshow(msk[0, :, :, i], cmap='gray')
                    plt.savefig(config.pic_path + f'nodule_{i}.png', bbox_inches="tight", pad_inches=0)
            plt.show()
        return {
            'name': img_name,
            'img': torch.from_numpy(img).float().contiguous(),
            'msk': torch.from_numpy(msk).float().contiguous()
        }
