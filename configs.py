# -*-coding:utf-8 -*-
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""

# todo 数据集路径
from torch import nn


class GC(nn.Module):
    version = None

    basePath = None
    basePathV2 = None
    pic_path = None

    # todo 数据集路径,npy 格式
    seg_path_luna_3d = None
    seg_path_luna_2d = None
    seg_path_lidc_3d = None
    seg_path_lidc_2d = None

    """模型参数路径"""
    pth_luna_path = None
    pth_lidc_path = None
    pth_lci_path = None
    """日志路径"""
    xml_path = None
    csv_path = None

    # todo 训练、评估配置
    dataset = 'luna'
    csv_name = 'debug'  # 日志、csv 保存的文件名,
    mode = '3d'  # 训练类型
    train = False  # 控制日志文件名称
    sup = False  # 深监督学习
    pretrained = False
    """超参数"""
    device = None
    pin_memory = False
    num_worker = 12
    epochs = 500
    show = False
    val_domain = 0.2
    train_domain = 0.8
    lr = 3e-4
    earlyEP = 50
    k_fold = 5
    train_batch_size = 12
    val_and_test_batch_size = 12
    optimizer = 'adam'
    model_name = ''
    loss_name = ''
    lost_loss = False
    # todo focal loss
    alpha = 0.25
    gamma = 2
    # todo input size
    input_size_2d = 64
    input_size_3d = 64

    def __init__(self, train=False, dataset='luna', log_name='debug', mode='3d', device='cuda:0', pathV=2, LossV=1,
                 FileV='npy', MetricsV=1, sup=False, server='zsm', pretrained=7, loadModel=False):
        super(GC, self).__init__()
        self.sup = sup
        self.device = device
        self.LossV = LossV
        self.FileV = FileV
        self.MetricsV = MetricsV
        self.version = pathV
        self.train = train
        self.dataset = dataset
        self.log_name = log_name
        self.mode = mode
        self.server = server
        self.pretrained = pretrained
        self.loadModel = loadModel

        self.SetDatasetPath()
        self.SetPthPath()

        self.updateVersion(server, pathV)
        self.xml_path = f'{self.basePath}/LIDCXML/lidcxml/'
        self.csv_path = f'{self.basePath}/LIDCXML/annos/'
        self.pred_path = f'{self.basePath}/predict'
        self.pic_path = f'{self.basePath}/pic/'

    def updateVersion(self, server='zsm', version=2):
        if pathV == 1:
            self.basePath = f'/{server}/jwj/baseExp/'
        else:
            self.basePath = f'/{server}/jwj/baseExpV{pathV}/'

    def SetDatasetPath(self):
        self.updateVersion(server=self.server, version=2)
        # self.basePath = f'/{server}/jwj/baseExpV5/'
        self.seg_path_luna_3d = f'{self.basePath}/$segmentation/seg_luna_3d/'
        self.seg_path_luna_2d = f'{self.basePath}/$segmentation/seg_luna_2d/'
        self.seg_path_lidc_3d = f'{self.basePath}/$segmentation/seg_lidc_3d/'
        self.seg_path_lidc_2d = f'{self.basePath}/$segmentation/seg_lidc_2d/'

    def SetPthPath(self, ):
        self.updateVersion(server=self.server, version=2)
        self.pth_luna_path = f'{self.basePath}/pth_luna/'
        self.pth_lci_path = f'{self.basePath}/pth_lci/'
        self.pth_lidc_path = f'{self.basePath}/pth_lidc/'


"""
添加属性后需要在trainBase进行添加
"""
train = False
dataset = 'luna'
log_name = 'eva'
mode = '3d'  # 2d,3d
pathV = 7  # 项目、数据集路径
LossV = 1  # 损失函数版本
FileV = 'npy'  # 文件类型 npy，nii.gz
MetricsV = 1  # 评估指标版本
device = 'cuda:0'
sup = False
server = ''
pretrained = False
saveshow = False    # 评估时进行保存
loadModel = False

config = GC(train=train, dataset=dataset, log_name=log_name, mode=mode, device=device, pathV=pathV, LossV=LossV,
            FileV=FileV, MetricsV=MetricsV, sup=sup, server=server, pretrained=pretrained, loadModel=loadModel)
