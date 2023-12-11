# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import os
from collections import OrderedDict
from functools import partial
from glob import glob
from random import random, seed, getstate, setstate, choice

import cv2
import numpy as np
import pandas
import torch
import torchvision.utils
from matplotlib import pyplot as plt
from matplotlib.image import imsave
from ptflops import get_model_complexity_info
from scipy.ndimage import shift, rotate
from scipy.ndimage import zoom
from skimage.transform import resize
from volumentations import Compose, RandomRotate90, Rotate, Flip

from configs import config

from models.modal.dualCRUNet import dualCRUNetv1
from models.modal.dualCRUNetD2 import dualCRUNetD2
from models.modal.dualCRUNetv2 import dualCRUNet
from models.modal.dualCRUNetv3 import dualCRUNetD4
from models.modal.dualCRUNetv4 import dualCRUNetD3

from models.model_trans.transbts.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from models.model_trans.unetr import UNETR
from models.models_3d.mipt.ASA.ASA import MEDIUMVIT
from models.models_3d.mipt.reconnet import ReconNet
from models.models_3d.mipt.unet_nested.networks.UNet_Nested import UNet_Nested
from models.models_3d.others.resunet.model import UNet3d
from utils.logger import logs


pth_path = config.pth_luna_path
pred_path = config.pred_path


def getAllAttrs(evaluate=False):
    attrs = dict()
    subtlety = ['ExtremelySubtle', 'ModeratelySubtle', 'FairlySubtle', 'ModeratelyObvious', 'Obvious']  #
    internalStructure = ['SoftTissue', 'Fluid', 'Fat', 'Air']
    calcification = ['Popcorn', 'Laminated', 'cSolid', 'Noncentral', 'Central', 'Absent']
    sphericity = ['Linear', 'OvoidLinear', 'Ovoid', 'OvoidRound', 'Round']
    margin = ['PoorlyDefined', 'NearPoorlyDefined', 'MediumMargin', 'NearSharp', 'Sharp']
    lobulation = ['NoLobulation', 'NearlyNoLobulation', 'MediumLobulation', 'NearMarkedLobulation',
                  'MarkedLobulation']

    spiculation = ['NoSpiculation', 'NearlyNoSpiculation', 'MediumSpiculation', 'NearMarkedSpiculation',
                   'MarkedSpiculation']

    texture = ['NonSolidGGO', 'NonSolidMixed', 'PartSolidMixed', 'SolidMixed', 'tSolid']
    maligancy = ['benign', 'uncertain', 'malignant']

    if evaluate:
        attrs.update({'subtlety': subtlety})

        # attrs.update({'internalStructure': internalStructure})  # 不评估该属性  x
        # attrs.update({'calcification': calcification})  # x6

        sphericity = ['Linear', 'OvoidLinear', 'Ovoid', 'OvoidRound', 'Round']  # luna Linear 不存在, 'Linear',
        attrs.update({'sphericity': sphericity})
        attrs.update({'margin': margin})
        attrs.update({'lobulation': lobulation})
        attrs.update({'spiculation': spiculation})
        attrs.update({'texture': texture})
        attrs.update({'maligancy': maligancy})
        size = ['sub36', 'sub6p', 'solid36', 'solid68', 'solid8p']  # 投票时手动指定
        attrs.update({'size': size})
    else:
        attrs = [subtlety, internalStructure, calcification, sphericity, margin, lobulation,
                 spiculation, texture]

    return attrs


def caculateDiameter(img, msk):
    """
    input:1x64x64x64
    description: 通过最小外接球或最小外接矩阵估算肺结节的最大直径
    """
    msk = msk[0]

    msk[msk >= 0.5] = 1
    msk[msk < 0.5] = 0

    # 将掩码转换为 CV_8UC1 格式的图像
    mask_cv8uc1 = (msk * 255).astype(np.uint8)
    max_diameter = 0
    for i in range(mask_cv8uc1.shape[0]):
        try:
            contours, _ = cv2.findContours(mask_cv8uc1[:, :, i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rect = cv2.minAreaRect(contours[0])  # 最小外接矩阵求肺结节直径
            # 提取矩形的尺寸
            width, height = rect[1]

            # 计算直径  
            diameter = max(width, height)

            # (x, y), radius = cv2.minEnclosingCircle(contours[0])  # 最小外接球体求肺结节直径
            # # 计算直径
            # diameter = 2 * radius
            # print('circle:', diameter)
            # imshows(img[0], mask_cv8uc1, k=i)

        except Exception as e:
            continue

        if diameter > max_diameter:
            max_diameter = diameter

    return np.round(max_diameter, 2)


def get_set(k, lesion_list):
    set_len = len(lesion_list)
    copies = int(set_len * config.val_domain)

    val_sidx = (k - 1) * copies
    val_eidx = val_sidx + copies
    if k == 5:
        val_eidx = max(val_eidx, set_len)
    # todo 得到验证集
    val_set = lesion_list[val_sidx:val_eidx]

    train_set = []
    train_set.extend(lesion_list[:val_sidx])
    train_set.extend(lesion_list[val_eidx:])

    return [train_set, val_set]


def set_init(k, seg_path, re, lists, format='*.npy', all=False):
    if re is not None:
        lesion_list = glob(seg_path + re)
    else:
        # lesion_list = glob(seg_path + '*.nii.gz')
        lesion_list = glob(seg_path + format)
    print(len(glob(seg_path + format)))
    lesion_list.sort()
    lesion_list = [item for item in lesion_list if 'sub3c' not in item]
    lesion_list = [item for item in lesion_list if 'solid3c' not in item]
    # 添加验证集
    # train_val = lesion_list[:-len(lesion_list) // 6]
    # test_list = lesion_list[len(train_val):]
    #
    if len(lesion_list) != 0 and not all:
        set_list = get_set(k, lesion_list)
        for i in range(len(set_list)):
            lists[i].extend(set_list[i])
        # lists[-1].extend(test_list)  # 添加测试集
        return lists
    else:
        return lesion_list


def seed_torch(sd=2, original_torch_rng_state=None, original_numpy_rng_state=None, original_gpu_seed=None,
               original_random_state=None):
    #
    if sd:
        # 获取当前状态
        original_torch_rng_state = torch.get_rng_state()
        original_numpy_rng_state = np.random.get_state()
        original_gpu_seed = torch.cuda.initial_seed()
        original_random_state = getstate()
        # fix
        seed(sd)
        os.environ['PYTHONHASHSEED'] = str(sd)
        np.random.seed(sd)
        torch.manual_seed(sd)
        torch.cuda.manual_seed(sd)
        torch.cuda.manual_seed_all(sd)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        return original_torch_rng_state, original_numpy_rng_state, original_gpu_seed, original_random_state
    else:
        setstate(original_random_state)
        os.environ['PYTHONHASHSEED'] = ''
        torch.set_rng_state(original_torch_rng_state)
        np.random.set_state(original_numpy_rng_state)
        torch.cuda.manual_seed(original_gpu_seed)
        torch.cuda.manual_seed_all(original_gpu_seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        return None


def imshows(img, msk=None, k=31, save=False, savename='new_png'):
    """
    input size：64x64x64
    """
    if msk is not None:
        # 创建多个子图，每行显示一个影像
        fig, axs = plt.subplots(1, 2, figsize=(5 * 2, 5))

        axs[0].imshow(img[:, :, k], cmap=plt.cm.gray, alpha=1, interpolation='sinc')
        axs[0].axis('off')

        axs[1].imshow(msk[:, :, k] * img[:, :, k], cmap=plt.cm.gray, alpha=1, interpolation='sinc')
        axs[1].axis('off')  #
        if save:
            print('save png file')
            imsave('img.png', img[:, :, k], cmap=plt.cm.gray)
            imsave('msk.png', msk[:, :, k], cmap=plt.cm.gray)

        plt.tight_layout()
        plt.show()
        plt.close()
    else:
        # 创建多个子图，每行显示一个影像
        fig, axs = plt.subplots(figsize=(5, 5))

        axs.imshow(img[:, :, k], cmap=plt.cm.gray, alpha=1, interpolation='sinc')
        if save:
            print(f'save new png file:{savename}.png')
            imsave(f'{savename}.png', img[:, :, k], cmap=plt.cm.gray)

        axs.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close()




def get_rotate():
    return Compose([
        RandomRotate90((0, 1), p=0.5),
        Rotate((0, 0), (0, 0), (-45, 45), p=0.2),
        Flip(1, p=0.5),
        Flip(0, p=0.5),
    ], p=1.0)


def transforms3d(img, label):
    val = choice([1, 2, 3, 4])
    if val == 1 or val == 2:
        data = {'image': img[0], 'mask': label[0]}
        aug = get_rotate()
        aug_data = aug(**data)
        img, msk = aug_data['image'], aug_data['mask']
        img = img[np.newaxis, ...]
        label = msk[np.newaxis, ...]
    elif val == 3:
        img, label = translate_image(img[0], label[0])  # 平移

    return img, label

def translate_image(image, msk):
    """
    平移图像
    :param image: 输入的图像
    :param shift_amount: 平移的距离，一个包含(x, y, z)的元组
    :return: 平移后的图像
    """
    pixel = 8
    val = choice([1, 2, 3])
    if val == 1:
        random_shift_x = np.random.randint(-pixel, pixel)
        random_shift_y = 0
    elif val == 2:
        random_shift_x = 0
        random_shift_y = np.random.randint(-pixel, pixel)
    else:
        random_shift_x = np.random.randint(-pixel, pixel)
        random_shift_y = np.random.randint(-pixel, pixel)

    # Z轴上的平移设为0（不进行Z轴平移）
    shift_amount = (random_shift_x, random_shift_y, 0)

    image = shift(image, shift_amount, mode='nearest', cval=0)
    msk = shift(msk, shift_amount, mode='nearest', cval=0)

    msk[msk < 0.5] = 0
    msk[msk >= 0.5] = 1

    return image[np.newaxis, ...], msk[np.newaxis, ...]


def avgStd(arr, log=False):
    arr = np.array(arr)
    mean = np.round(arr.mean(), 2)
    if log:
        std = np.round(arr.std(ddof=1), 2)
        return f"{mean:.2f}±{std:.2f}"
    else:
        return mean


def showTime(fold, start, end):
    times = round(end - start, 2)
    hours = round(times / 3600, 2)
    days = round(times / (3600 * 24), 2)
    if fold not in [1, 2, 3, 4, 5]:
        logs(f"Fold {fold}, time: {hours:.2f} hours, {days:.2f} days")
    else:
        logs(f"{fold}, time: {hours:.2f} hours, {days:.2f} days")

    return times

import SimpleITK as sitk

# def sobel_filter(image):
#     image = sitk.GetImageFromArray(image)
#     # image = Gaussian(image)
#     # 创建Sobel算子的滤波器
#     sobel_filter = sitk.SobelEdgeDetectionImageFilter()
#
#     # 应用Sobel算子滤波器
#     sobel_image = sobel_filter.Execute(image)
#
#     arr = sitk.GetArrayFromImage(sobel_image)
#     return arr[np.newaxis, ...]

def sobel_filter(image):
    """
    cv2更加清晰
    """
    # Apply Sobel filter to all 2D slices at once
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    filtered_volume = np.expand_dims(sobel, axis=0)

    return filtered_volume


def canny_filter(image):
    """
    自适应canny算子
    """
    imgs = []

    for i in range(image.shape[2]):
        # 确保像素值的数据类型为CV_8U
        gray_image = cv2.convertScaleAbs(image[:, :, i])
        adaptive_threshold = cv2.adaptiveThreshold(gray_image, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY, 11, 0.05)

        # # 创建CLAHE对象并应用对比度有限自适应直方图均衡化
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # clahe_image = clahe.apply(gray_image)

        imgs.append(adaptive_threshold)

    return np.stack(imgs, axis=2)[np.newaxis, ...]


def get_counter(msk):
    msk = sitk.GetImageFromArray(msk)
    msk = sitk.Cast(msk, sitk.sitkUInt8)
    label_contour = sitk.LabelContourImageFilter()
    label_contour.SetBackgroundValue(0)
    contour_img = label_contour.Execute(msk)
    contour_arr = sitk.GetArrayFromImage(contour_img)
    return contour_arr[np.newaxis, ...]


def load_model_k_checkpoint(pthPath, mode, model_name, optimizer, loss_name, model, k, verbose=True):
    if verbose:
        logs(f'============load {model_name} == Fold {k} check point============')
    file = os.path.join(pthPath, f'{mode}_{model_name}_{str(k)}_{optimizer}_{loss_name}_checkpoint.pth')
    print(file)
    if not os.path.exists(file):
        logs('pth not exist')
        exit(0)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        check_point = torch.load(file, map_location=device)
        # print(device)
        # 检查当前模型和state_dict的键
        print('load keys')  # 当前模型的键
        # print(check_point.keys())  # 加载的state_dict的键

        try:
            model.load_state_dict(check_point)
        except Exception as e:
            # print(e.args)
            model.load_state_dict(check_point, strict=True)
            print('pth与model不一致！')
            raise ValueError


def get_parm(model='2d', model_name='None', verbose=False):
    from torch.autograd import Variable
    model = model.lower()
    print(model_name)
    if model == '2d':
        SIZE = config.input_size_2d
        x = Variable(torch.rand(8, 1, SIZE, SIZE)).cuda()
        model = get_model2d(model_name, config.device)
        macs, params = get_model_complexity_info(model, (1, SIZE, SIZE), as_strings=True,
                                                 print_per_layer_stat=verbose, verbose=verbose)
    else:
        SIZE = config.input_size_3d
        x = Variable(torch.rand(8, 1, SIZE, SIZE, SIZE)).cuda()
        model = get_model3d(model_name, config.device)
        macs, params = get_model_complexity_info(model, (1, SIZE, SIZE, SIZE), as_strings=True,
                                                 print_per_layer_stat=verbose, verbose=verbose)

    if verbose:
        y = model(x)
        print('Output shape:', y.shape)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    return macs, params


def checkThickSlice(seriesuid):
    """
    判断是否是厚层CT
    """
    df = pandas.read_csv('/zsm/jwj/baseExpV7/LIDCXML/annos/all_device_thick_info.csv')
    row_data = df[df['seriesuid'] == seriesuid].iloc[0]

    if row_data['slice_thickness'] <= 2.5:
        return True
    else:
        return False


import torch.nn as nn


def weights_init(m, xohnorm='he'):
    classname = m.__class__.__name__
    try:
        if classname.find('Conv') != -1:
            if xohnorm == 'x':
                nn.init.xavier_uniform_(m.weight.data)
            else:
                nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('LayerNorm') != -1:
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    except Exception as e:
        print(e.args)


def get_model3d(model_name, device, verbose=False):
    model_name = model_name.lower()
    filtersless = [16, 32, 64, 128, 256]
    filtersmid = [32, 64, 128, 184, 256]
    filtersbig = [32, 64, 128, 256, 320]
    filterhuge = [64, 128, 256, 512, 512]

    side = False
    upsample = True
    # unet++3d
    if model_name == 'unet':
        model = UNet3d(1, 1, False)
    elif model_name == 'unetpp':
        model = UNet_Nested()
    elif model_name == 'reconnet':
        model = ReconNet(32, 1)
    elif model_name == 'unetr':
        model = UNETR(in_channels=1, out_channels=1, img_size=(64, 64, 64), pos_embed='conv', norm_name='instance')
    elif model_name == 'transbts':
        _, model = TransBTS(img_dim=64, num_classes=1) 
    elif model_name == 'asa':
        model = MEDIUMVIT(in_channels=1, out_channels=1, img_size=(64, 64, 64), norm_name='instance')
    elif model_name == 'dualbsp':
        model = dualCRUNetv1('conv')
    elif model_name == 'dualbspa':
        model = dualCRUNetv1('conv')
    elif model_name == 'dualbca':
        model = dualCRUNetv1('conv')
    elif model_name == 'dualbcaa':
        model = dualCRUNetv1('conv')
    elif model_name == 'dualbasa':
        model = dualCRUNetv1('conv')
    elif model_name == 'dualbd5df':  # 不同归一化方法
        model = dualCRUNet('conv')
    elif model_name == 'dualbd5zsc':  # zscore+sobel + cut norm
        model = dualCRUNet('conv')
    elif model_name == 'dualbd5cs':  # cut+sobel
        model = dualCRUNet('conv')
    elif model_name == 'dualbd5zs':  # zscore+sobel
        model = dualCRUNet('conv')
    elif model_name == 'dualbd2':  # 同归一化方法
        model = dualCRUNetD2('conv')
    elif model_name == 'dualbd3':  # 同归一化方法
        model = dualCRUNetD3('conv')
    elif model_name == 'dualbd4':  # 同归一化方法
        model = dualCRUNetD4('conv')
    elif model_name == 'dualbd5':  # 同归一化方法
        model = dualCRUNet('conv')
    else:
        raise Exception(f"no model name as {model_name}")

    if config.device != 'cpu' and torch.cuda.is_available():
        if verbose:
            logs(f'Use {device}')
        model.to(device)
    else:
        if verbose:
            logs('Use CPU')
        model.to('cpu')

    if config.train and not config.loadModel:
        if model_name.find('dualb') != -1:
            model.apply(partial(weights_init, xohnorm='x'))
        else:
            model.apply(weights_init)

    return model


if __name__ == '__main__':
    # TODO
    model3d = ['unet','unetpp', 'reconnet', 'transbts', 'unetr',  ]

    get_parm('3d', 'asa', True)
