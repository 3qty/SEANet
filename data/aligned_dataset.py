import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np

class AlignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.dir_AB = os.path.join(opt.dataroot, opt.setname,opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        # print(AB_path)
        AB = Image.open(AB_path).convert('RGB')

        w, h = AB.size

        # Return data in 'train' or 'val' phase
        if self.opt.phase=='train' or self.opt.phase=='val':
            w2 = int(w / 2)
            ori_A = AB.crop((0, 0, w2, h))
            ori_B = AB.crop((w2, 0, w, h))

            A=ori_A.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
            B=ori_B.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
            if self.opt.phase=='val' and self.opt.saveimg:
                ori_A = np.array(ori_A)
            ori_B=np.array(ori_B)

            A = transforms.ToTensor()(A)
            B = transforms.ToTensor()(B)

            w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
            h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))

            A = A[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
            B = B[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]


            A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)

            B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

            if self.opt.direction == 'BtoA':
                input_nc = self.opt.output_nc
                output_nc = self.opt.input_nc
            else:
                input_nc = self.opt.input_nc
                output_nc = self.opt.output_nc

            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A = A.index_select(2, idx)
                B = B.index_select(2, idx)

            if input_nc == 1:  # RGB to gray
                tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
                A = tmp.unsqueeze(0)

            if output_nc == 1:  # RGB to gray
                tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
                B = tmp.unsqueeze(0)

            if self.opt.phase=='train':
                return {'A': A, 'B': B,
                        'A_paths': AB_path, 'B_paths': AB_path}

            elif self.opt.phase=='val':
                if self.opt.saveimg:
                    return {'A': A, 'B': B,"INPUT":ori_A,'GT':ori_B,
                        'A_paths': AB_path, 'B_paths': AB_path}
                else:
                    return {'A': A, 'B': B,'GT':ori_B,
                        'A_paths': AB_path, 'B_paths': AB_path}

        # Return data in 'test' phase
        else:
            A = AB.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
            A = transforms.ToTensor()(A)
            A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
            return {'A': A,"INPUT":np.array(AB),
                    'A_paths': AB_path}
    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
