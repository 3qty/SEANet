import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
import numpy as np
from torchvision import models

class SEAUnetModel(BaseModel):
    def name(self):
        return 'SEAUnetModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='aligned')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = [ 'pixel','perc','style','edge']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'B_final','B_sub','B_out']

        # which output need to be evaluated during 'val' phase
        self.val_names = ['B_sub', 'B_out', 'B_final']

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            #define assisted module
            self.sobel_edge = ImageGradient(self.gpu_ids)
            vgg16_extractor = VGG16FeatureExtractor()
            self.per_cri = InpaintingLoss(vgg16_extractor,self.gpu_ids)

            #define loss
            self.criterionL1 = torch.nn.L1Loss()

            #define optimizer
            self.optimizers = []
            self.optimizer_G = torch.optim.SGD(self.netG.parameters(),lr=opt.lr,momentum=0.9)
            self.optimizers.append(self.optimizer_G)


    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        if(self.isTrain):
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        output_list = self.netG(self.real_A)
        self.B_sub = output_list[0]
        self.B_out = output_list[1]
        self.B_final = (self.B_sub+self.B_out)/2.0


    def backward_G(self):

        # Pixel Loss
        self.loss_pixel=500*(self.criterionL1(self.B_sub,self.real_B)+\
                        self.criterionL1(self.B_out,self.real_B))

        # Perceptual Loss and Style Loss
        loss_dict_Bsub=self.per_cri(self.B_sub,self.real_B)
        loss_dict_Bout=self.per_cri(self.B_out,self.real_B)

        self.loss_perc = 1.5*(loss_dict_Bsub['prc']+loss_dict_Bout['prc'])
        self.loss_style = 250*(loss_dict_Bsub['style']+loss_dict_Bout['style'])

        # Edge Loss
        gt_edge = self.sobel_edge(self.real_B)
        Bsub_edge = self.sobel_edge(self.B_sub)
        Bout_edge = self.sobel_edge(self.B_out)

        self.loss_edge = (self.criterionL1(Bout_edge,gt_edge)+
                           self.criterionL1(Bsub_edge,gt_edge)
                           )

        self.loss_G = self.loss_pixel + self.loss_perc + self.loss_style + self.loss_edge
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


# Sobel Operator
class ImageGradient(nn.Module):
    def __init__(self, gpu_ids=[]):
        super(ImageGradient, self).__init__()
        self.gpu_ids = gpu_ids

        # different weight of sobel operator

        # a = np.array([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        #                [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        #                [[1, 0, -1], [2, 0, -2], [1, 0, -1]]],
        #               [[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        #                [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        #                [[1, 0, -1], [2, 0, -2], [1, 0, -1]]],
        #               [[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        #                [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        #                [[1, 0, -1], [2, 0, -2], [1, 0, -1]]]])

        a = np.array([[[[3, 0, -3], [10, 0, -10], [3, 0, -3]],
                       [[3, 0, -3], [10, 0, -10], [3, 0, -3]],
                       [[3, 0, -3], [10, 0, -10], [3, 0, -3]]],
                      [[[3, 0, -3], [10, 0, -10], [3, 0, -3]],
                       [[3, 0, -3], [10, 0, -10], [3, 0, -3]],
                       [[3, 0, -3], [10, 0, -10], [3, 0, -3]]],
                      [[[3, 0, -3], [10, 0, -10], [3, 0, -3]],
                       [[3, 0, -3], [10, 0, -10], [3, 0, -3]],
                       [[3, 0, -3], [10, 0, -10], [3, 0, -3]]]])
        conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        if len(self.gpu_ids) > 0:
            conv1.weight = nn.Parameter(torch.from_numpy(a).cuda(self.gpu_ids[0]).float(), requires_grad=False)
        else:
            conv1.weight = nn.Parameter(torch.from_numpy(a).float(), requires_grad=False)
        # different weight of sobel operator
        # b = np.array([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        #                [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        #                [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]],
        #               [[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        #                [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        #                [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]],
        #               [[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        #                [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        #                [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]])

        b = np.array([[[[3, 10, 3], [0, 0, 0], [-3, -10, -3]],
                       [[3, 10, 3], [0, 0, 0], [-3, -10, -3]],
                       [[3, 10, 3], [0, 0, 0], [-3, -10, -3]]],
                      [[[3, 10, 3], [0, 0, 0], [-3, -10, -3]],
                       [[3, 10, 3], [0, 0, 0], [-3, -10, -3]],
                       [[3, 10, 3], [0, 0, 0], [-3, -10, -3]]],
                      [[[3, 10, 3], [0, 0, 0], [-3, -10, -3]],
                       [[3, 10, 3], [0, 0, 0], [-3, -10, -3]],
                       [[3, 10, 3], [0, 0, 0], [-3, -10, -3]]]])
        conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        if len(self.gpu_ids) > 0:
            conv2.weight = nn.Parameter(torch.from_numpy(b).cuda(self.gpu_ids[0]).float(), requires_grad=False)
        else:
            conv2.weight = nn.Parameter(torch.from_numpy(b).float(), requires_grad=False)

        self.conv1 = conv1
        self.conv2 = conv2

    def forward(self, input):
        if len(self.gpu_ids) > 0 and isinstance(input.data, torch.cuda.FloatTensor):

            G_x = nn.parallel.data_parallel(self.conv1, input, self.gpu_ids)
            G_y = nn.parallel.data_parallel(self.conv2, input, self.gpu_ids)
        else:
            G_x = self.conv1(input)
            G_y = self.conv2(input)

        return torch.abs(G_x) + torch.abs(G_y)


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
           torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


class InpaintingLoss(nn.Module):
    def __init__(self, extractor, gpu_ids):
        super().__init__()
        self.l1 = nn.L1Loss()
        extractor = extractor

        if len(gpu_ids) > 0:
            # print(gpu_ids)
            assert (torch.cuda.is_available())
            extractor.to(gpu_ids[0])
            extractor = torch.nn.DataParallel(extractor, gpu_ids)
        self.extractor = extractor

    def forward(self, output, gt):
        loss_dict = {}
        output_comp = output

        if output.shape[1] == 3:
            feat_output_comp = self.extractor(output_comp)
            feat_output = self.extractor(output)
            feat_gt = self.extractor(gt)
        elif output.shape[1] == 1:
            feat_output_comp = self.extractor(torch.cat([output_comp] * 3, 1))
            feat_output = self.extractor(torch.cat([output] * 3, 1))
            feat_gt = self.extractor(torch.cat([gt] * 3, 1))
        else:
            raise ValueError('only gray an')

        loss_dict['prc'] = 0.0
        for i in range(3):
            loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i].detach())
            loss_dict['prc'] += self.l1(feat_output_comp[i], feat_gt[i].detach())

        loss_dict['style'] = 0.0
        for i in range(3):
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]).detach())
            loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[i]),
                                          gram_matrix(feat_gt[i]).detach())

        #loss_dict['tv'] = total_variation_loss(output_comp)

        return loss_dict
