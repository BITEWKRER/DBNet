# -*-coding:utf-8 -*-
"""
# Time       ：2023/3/28 15:23
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import torch
from ptflops import get_model_complexity_info
from torch import nn
from torch.nn.functional import interpolate

from models.modal.AttentionRebuild import CASP
from utils.norm import Swish, unetconv3d, conv3d


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class Upsample(nn.Module):
    def __init__(self, channel, mode='ins'):
        super().__init__()
        self.mode = mode
        if mode != 'ins':
            self.up = nn.ConvTranspose3d(channel, channel, 2, 2, padding=0, bias=False)
            self.norm = nn.BatchNorm3d(channel)
            self.swish = Swish()

    def _upsample(self, src, tar):
        return interpolate(src, size=tar.shape[2:], mode='trilinear', align_corners=True)

    def forward(self, src, target=None):
        if self.mode == 'ins':
            return self._upsample(src, target)
        else:
            return self.swish(self.norm(self.up(src)))


class Downsample(nn.Module):

    def __init__(self, stride, channel=None, mode='pool'):  # pool 设置在enconder
        super(Downsample, self).__init__()
        assert stride is not None
        self.mode = mode
        if mode == 'pool':
            self.maxpool = nn.MaxPool3d(stride, stride, ceil_mode=True)  # ceil_mode 向上取整
        else:
	    pass

    def forward(self, x):
        if self.mode == 'pool':
            return self.maxpool(x)
        else:
            return None


class Enconder(nn.Module):
    def __init__(self, in_c=1, filters=[16, 32, 64, 128, 256], down_mode='pool'):
        super(Enconder, self).__init__()
        self.in_c = in_c
        self.conv1 = unetconv3d(1, filters[0], )
        self.maxpool1 = Downsample(stride=2, channel=filters[0], mode=down_mode)

        self.conv2 = unetconv3d(filters[0], filters[1], )
        self.maxpool2 = Downsample(stride=2, channel=filters[1], mode=down_mode)

        self.conv3 = unetconv3d(filters[1], filters[2], )
        self.maxpool3 = Downsample(stride=2, channel=filters[2], mode=down_mode)

        self.conv4 = unetconv3d(filters[2], filters[3], )
        self.maxpool4 = Downsample(stride=2, channel=filters[3], mode=down_mode)

        self.conv5 = unetconv3d(filters[3], filters[4], )

    def forward(self, input, shuffle=False):
        h1 = self.conv1(input)
        h2 = self.maxpool1(h1)

        h2 = self.conv2(h2)
        h3 = self.maxpool2(h2)

        h3 = self.conv3(h3)
        h4 = self.maxpool3(h3)

        h4 = self.conv4(h4)
        h5 = self.maxpool4(h4)

        # v13
        h5 = self.conv5(h5)

        return h1, h2, h3, h4, h5


class DBNet(nn.Module):

    def __init__(self, depth=1):
        super(DBNet, self).__init__()
        upmodel = 'ins'
        filters = [32, 64, 128, 256, 512]  
        self.up = Enconder(filters=filters, down_mode='pool') 
        self.down = Enconder(filters=filters, down_mode='pool')

        self.fusion4_up = unetconv3d(filters[4] + filters[3], filters[3])
        self.fusion3_up = unetconv3d(filters[3] + filters[2], filters[2])
        self.fusion2_up = unetconv3d(filters[2] + filters[1], filters[1])
        self.fusion1_up = unetconv3d(filters[1] + filters[0], filters[0])

        self.fusion4_down = unetconv3d(filters[4] + filters[3], filters[3])
        self.fusion3_down = unetconv3d(filters[3] + filters[2], filters[2])
        self.fusion2_down = unetconv3d(filters[2] + filters[1], filters[1])
        self.fusion1_down = unetconv3d(filters[1] + filters[0], filters[0])
        # todo 下采样
        self.up4_up = Upsample(filters[4], mode=upmodel)
        self.up3_up = Upsample(filters[3], mode=upmodel)
        self.up2_up = Upsample(filters[2], mode=upmodel)
        self.up1_up = Upsample(filters[1], mode=upmodel)

        self.up4_down = Upsample(filters[4], mode=upmodel)
        self.up3_down = Upsample(filters[3], mode=upmodel)
        self.up2_down = Upsample(filters[2], mode=upmodel)
        self.up1_down = Upsample(filters[1], mode=upmodel)

        # todo 残差连接
        self.res4 = conv3d(filters[4] * 2, filters[3], n=1, ks=1, padding=0)
        self.res3 = conv3d(filters[3] * 2, filters[2], n=1, ks=1, padding=0)
        self.res2 = conv3d(filters[2] * 2, filters[1], n=1, ks=1, padding=0)
        self.res1 = conv3d(filters[1] * 2, filters[0], n=1, ks=1, padding=0)

        self.attn5 = CASP(1024, 64, hwd=4)
        self.attn4 = CASP(512, 32, hwd=8)
        self.attn3 = CASP(256, 16, hwd=16)
        self.attn2 = CASP(128, 8, hwd=32)

        self.out = nn.Conv3d(64, 1, kernel_size=3, padding=1)

    def fusion(self, fx1, fx2, target_1, targer_2, block_1, block_2, res_block, _upsample_1=_upsample,
               _upsample_2=_upsample):

        hdup = _upsample_1(fx1, target_1)
        fusion_up = block_1(torch.cat((hdup, target_1), dim=1))

        hddown = _upsample_2(fx2, targer_2)
        fusion_down = block_2(torch.cat((hddown, targer_2), dim=1))

        res = res_block(torch.cat((hdup, hddown), dim=1))

        return fusion_up + res, fusion_down + res

    def forward(self, x):
        img = x[:, 0:1, :, :, :]
        border = x[:, 1:, :, :, :]

        hu1, hu2, hu3, hu4, hu5 = self.up(img)
        hx1, hx2, hx3, hx4, hx5 = self.down(border, True)

        attnu5, attnx5 = self.attn5(hu5, hx5)
        fu4, fx4 = self.fusion(attnu5, attnx5, hu4, hx4, self.fusion4_up, self.fusion4_down, self.res4, self.up4_up,
                               self.up4_down)

        attnu4, attnx4 = self.attn4(fu4, fx4)
        fu3, fx3 = self.fusion(attnu4, attnx4, hu3, hx3, self.fusion3_up, self.fusion3_down, self.res3, self.up3_up,
                               self.up3_down)

        attnu3, attnx3 = self.attn3(fu3, fx3)
        fu2, fx2 = self.fusion(attnu3, attnx3, hu2, hx2, self.fusion2_up, self.fusion2_down, self.res2, self.up2_up,
                               self.up2_down)

        attnu2, attnx2 = self.attn2(fu2, fx2)
        fu1, fx1 = self.fusion(attnu2, attnx2, hu1, hx1, self.fusion1_up, self.fusion1_down, self.res1, self.up1_up,
                               self.up1_down)

        return self.out(torch.cat((fu1, fx1), dim=1))


if __name__ == '__main__':
    from torch.autograd import Variable

    var = torch.rand(1, 2, 64, 64, 64)
    x = Variable(var).cuda()
    model = dualCRUNetv1(deconder='conv').cuda()


    macs, params = get_model_complexity_info(model, (2, 64, 64, 64), as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    y = model(x)
    print('Output shape:', y.shape)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


