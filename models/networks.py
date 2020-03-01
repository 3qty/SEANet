import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init
import functools
import torch.nn.functional as F
from torch.optim import lr_scheduler
from functools import reduce
from torchvision import models
from models.cbam import CBAM
from models.bam import BAM
###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.5)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        # print(gpu_ids)
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, init_type='normal', init_gain=0.02,gpu_ids=[]):
    net = SEAUNet(input_nc,output_nc)

    return init_net(net, init_type, init_gain, gpu_ids)



##############################################################################
# Network Design
##############################################################################

class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction_ratio=16):
        super(ChannelAttention, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(num_features, num_features // reduction_ratio),
            nn.ReLU(),
            nn.Linear(num_features // reduction_ratio, num_features)
        )

    def forward(self, x):
        flat_features = x.reshape(x.size(0),x.size(1),-1)
        avg_features = flat_features.mean(2)
        avg_attention = self.fc(avg_features)

        max_features = flat_features.max(2)[0]
        max_attention = self.fc(max_features)

        att_sum = avg_attention+max_attention

        channel_attention = att_sum.unsqueeze(2).unsqueeze(3).expand_as(x)
        return channel_attention

class SpatialAttention(nn.Module):
    def __init__(self,num_features,reduction_ratio=16):
        super(SpatialAttention,self).__init__()
        self.compress_conv = nn.Conv2d(num_features,num_features//reduction_ratio-2,kernel_size=1)

        #Hybrid Dilated Convolution
        self.convs = nn.Sequential(
            nn.Conv2d(num_features//reduction_ratio,num_features//reduction_ratio,kernel_size=3,padding=1,dilation=1),
            nn.ReLU(),
            nn.Conv2d(num_features//reduction_ratio,num_features//reduction_ratio,kernel_size=3,padding=2,dilation=2),
            nn.ReLU(),
            nn.Conv2d(num_features//reduction_ratio,num_features//reduction_ratio,kernel_size=3,padding=5,dilation=5),
            nn.ReLU(),
            nn.Conv2d(num_features//reduction_ratio,1,kernel_size=1)
        )
    def forward(self,x):
        avg_feature = torch.mean(x,1).unsqueeze(1)
        max_feature = torch.max(x,1)[0].unsqueeze(1)
        compress_features = self.compress_conv(x)
        spatial_attention = self.convs(torch.cat([avg_feature,max_feature,compress_features],1))
        return spatial_attention.expand_as(x)


class SEA(nn.Module):
    def __init__(self,num_features):
        super(SEA,self).__init__()
        self.channel_att = ChannelAttention(num_features)
        self.spatial_att = SpatialAttention(num_features)

    def forward(self, in_tensor):
        att = 1 + F.sigmoid(self.channel_att(in_tensor) * self.spatial_att(in_tensor))
        return att * in_tensor

class DenseConv(nn.Module):
    def __init__(self, num_input_features, growth_rate):
        super(DenseConv, self).__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(num_input_features, growth_rate, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.single_conv(x)
        return torch.cat([x, out], 1)

#Separable Element-wise Attention Block
class SEAB(nn.Module):
    def __init__(self, num_layers,in_features, out_features,growth_rate,use_sea=True):
        super(SEAB, self).__init__()
        # flag of whether using separable element-wise attention
        self.use_sea=use_sea

        if (self.use_sea):
            self.sea = SEA(out_features)
        conv_ls = []
        for i in range(num_layers):
            conv_ls.append(DenseConv(in_features + i * growth_rate, growth_rate))
        self.convs = nn.Sequential(*conv_ls)
        # transition convolution
        self.transition = nn.Conv2d(in_features + num_layers * growth_rate, out_features, kernel_size=1, stride=1)

        #convolution for input features when channel number is inconsistent
        self.conv_align=None
        if(in_features != out_features):
            self.conv_align=nn.Conv2d(in_features,out_features,kernel_size=1,stride=1)

    def forward(self, x):
        out = self.convs(x)
        out = self.transition(out)

        if(self.use_sea):
            out=self.sea(out)

        if(self.conv_align):
            out = out + self.conv_align(x)
        else:
            out = out + x
        return out

class SEAUNet(nn.Module):
    def __init__(self, input_nc,output_nc):
        super(SEAUNet, self).__init__()

        self.down1 = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(True),
            SEAB(4,64, 64, 32)
        )
        self.down2 = nn.Sequential(
            nn.AvgPool2d(3,2,1),
            SEAB(4,64, 64, 32),
        )
        self.down3 = nn.Sequential(
            nn.AvgPool2d(3, 2,1),
            SEAB(4,64, 128, 32),
        )
        self.down4 = nn.Sequential(
            nn.AvgPool2d(3, 2,1),
            SEAB(4,128, 128, 32,)
        )
        self.down5 = nn.Sequential(
            nn.AvgPool2d(3, 2,1),
            SEAB(8,128, 256, 32),
        )
        self.down6 = nn.Sequential(
            nn.AvgPool2d(3,2, 1),
            SEAB(8,256, 256, 32),
        )
        self.down7 = nn.Sequential(
            nn.AvgPool2d(3, 2,1),
            SEAB(8,256, 512, 32),
        )
        self.down8 = nn.Sequential(
            nn.AvgPool2d(3,2, 1),
            SEAB(8,512, 512, 32),
        )
        self.up8 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            SEAB(8,512, 512, 32),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.up7 = nn.Sequential(
            SEAB(8,1024, 512, 32),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.up6 = nn.Sequential(
            SEAB(8,1024, 256, 32),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.up5 = nn.Sequential(
            SEAB(8,512, 256, 32),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.up4 = nn.Sequential(
            SEAB(4,512, 128, 32),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.up3 = nn.Sequential(
            SEAB(4,256, 128, 32),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.up2 = nn.Sequential(
            SEAB(4,256, 64, 32),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.up1 = nn.Sequential(
            SEAB(4,128, 64, 32),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.up0 = nn.Sequential(
            SEAB(4,128, 64, 32),
            nn.ConvTranspose2d(64,64,4,2,1)
        )

        self.to_rgb_Bsub = nn.Sequential(
            nn.Conv2d(32, output_nc, kernel_size=1, bias=False),
            nn.Tanh()
        )
        self.to_rgb_Bout = nn.Sequential(
            nn.Conv2d(32, output_nc, kernel_size=1, bias=False),
            nn.Tanh()
        )

    def _clip_rgb(self, rgb):
        return  F.hardtanh(rgb,inplace=True)

    def forward(self, x):
        d_ft256 = self.down1(x)
        d_ft128 = self.down2(d_ft256)
        d_ft64 = self.down3(d_ft128)
        d_ft32 = self.down4(d_ft64)
        d_ft16 = self.down5(d_ft32)
        d_ft8 = self.down6(d_ft16)
        d_ft4 = self.down7(d_ft8)
        d_ft2 = self.down8(d_ft4)
        u_ft2 = self.up8(d_ft2)
        u_ft4 = self.up7(torch.cat([u_ft2 , d_ft2],1))
        u_ft8 = self.up6(torch.cat([u_ft4 , d_ft4],1))
        u_ft16 = self.up5(torch.cat([u_ft8 , d_ft8],1))
        u_ft32 = self.up4(torch.cat([u_ft16 , d_ft16],1))
        u_ft64 = self.up3(torch.cat([u_ft32 , d_ft32],1))
        u_ft128 = self.up2(torch.cat([u_ft64 , d_ft64],1))
        u_ft256 = self.up1(torch.cat([u_ft128 , d_ft128],1))
        u_ft512 = self.up0(torch.cat([u_ft256 , d_ft256],1))

        Residual_Out = self.to_rgb_Bsub(u_ft512[:,:32])
        Bsub = self._clip_rgb(x[:,:3] + Residual_Out)
        Bout = self.to_rgb_Bout(u_ft512[:,32:])

        output_list = [Bsub,Bout]
        return output_list

