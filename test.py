import os
from options.test_options import TestOptions,ValOptions
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from skimage.measure import compare_psnr,compare_ssim
import ntpath


def get_val_opt():
    opt = ValOptions().parse()

    opt.num_threads = 1  # test code only supports num_threads = 1
    opt.batch_size = 4  # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    result_root = os.path.join(opt.results_dir, opt.name)
    if not os.path.exists(result_root):
        os.makedirs(result_root)
    return opt

def initial_dataset(opt,set_name):
    gt_prefix, gt_form = get_gt_info(set_name)
    opt.dataroot = "./dataroot/rain" + set_name + "/test/"
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    print("Current test set is "+set_name+" size:"+str(len(dataset)))
    record_fn = os.path.join(opt.results_dir, opt.name, "evaluation_" + set_name + ".txt")
    return gt_prefix,gt_form,dataset,record_fn

def save_test_image(dataset,set_name,opt,model,epoch):
    opt.epoch = epoch
    model.load_networks(opt.epoch)

    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s_rain%s_1itr' % (opt.phase, opt.epoch,set_name))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    for i, data in enumerate(dataset):

        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)

    webpage.save()

def model_test_val(opt,dataset,model,epoch):
    def process_img(img, h, w):
        # print(img.min(),img.max())

        img = (img + 1) / 2.0
        img = F.interpolate(img, [h, w], mode='bilinear')

        img = img.permute(0,2, 3, 1).cpu().numpy()
        return img

    saveimg=opt.saveimg or opt.phase=='test'

    if saveimg:
        img_dir = os.path.join(opt.results_dir,opt.name,opt.setname+'_'+str(epoch))
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

    is_val = (opt.phase == 'val')
    if is_val:
        record_fn = os.path.join(opt.results_dir, opt.name, "evaluation_" + opt.setname + ".txt")
        with open(record_fn, 'a') as f:
            f.write("epoch(" + str(epoch)+")\n")

        val_dic = {}
        for val_name in model.val_names:
            val_dic[val_name] = [0.0, 0.0, 0.0, 0.0]


        for i, data in enumerate(dataset):
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()

            gt = data['GT']
            gt = gt.numpy()

            batch_size=gt.shape[0]

            if saveimg:
                ori = data['INPUT']
                for i in range(batch_size):
                    short_path = ntpath.basename(data["A_paths"][i])
                    prefix = short_path.split('.')[0]

                    cur_ori = ori.numpy()[i]
                    plt.imsave(os.path.join(img_dir,prefix+"_ori.png"),cur_ori)
                    plt.imsave(os.path.join(img_dir,prefix+"_gt.png"),gt[i])

            ori_h, ori_w = gt.shape[1], gt.shape[2]

            if (gt.dtype == np.uint8):
                gt = gt / 255.0
                gt = gt.astype(np.float32)

            for val_name in model.val_names:
                val_out=visuals[val_name]
                val_out=process_img(val_out,ori_h,ori_w)
                for i in range(batch_size):
                    cur_psnr=compare_psnr(gt[i],val_out[i])
                    cur_ssim=compare_ssim(gt[i],val_out[i],multichannel=True)
                    val_dic[val_name][0]+=cur_psnr
                    val_dic[val_name][1]+=cur_ssim
                    if saveimg:
                        plt.imsave(os.path.join(img_dir, prefix + '_'+val_name+".png"), val_out[i])


        for val_name in val_dic:
            val_dic[val_name][2] = val_dic[val_name][0] / len(dataset)
            val_dic[val_name][3] = val_dic[val_name][1] / len(dataset)

        with open(record_fn,'a')as f:
             for val_name in val_dic:
                 f.write('   psnr_'+val_name+": "+str(val_dic[val_name][2])+
                         '   ssim_'+val_name+": "+str(val_dic[val_name][3])+'\n'
                         )
        return val_dic[val_name][2],val_dic[val_name][3]

    else:
        for i, data in enumerate(dataset):
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()

            ori = data['INPUT']
            batch_size = ori.shape[0]
            for i in range(batch_size):
                short_path = ntpath.basename(data["A_paths"][i])
                prefix = short_path.split('.')[0]

                cur_ori = ori.numpy()[i]
                plt.imsave(os.path.join(img_dir, prefix + "_ori.png"), cur_ori)

            ori_h, ori_w = ori.shape[1], ori.shape[2]

            for val_name in model.val_names:
                val_out = visuals[val_name]
                val_out = process_img(val_out, ori_h, ori_w)
                for i in range(batch_size):
                    plt.imsave(os.path.join(img_dir, prefix + '_' + val_name + ".png"), val_out[i])
        return 0.0,0.0
def eval_in_training(model,epoch,isTest=False):
    opt = get_val_opt()
    #save image result of validation set in last iteration, otherwise compute ssim and psnr only
    if epoch == opt.saveimg_epoch:
        opt.saveimg=True

    set_name=opt.setname
    opt.batch_size=1
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    print("Current test set is "+set_name+" size:"+str(len(dataset)))
    psnr,ssim=model_test_val(opt,dataset,model,epoch)

    return psnr,ssim

if __name__ == '__main__':
    opt = get_val_opt()

    opt.epoch = 'latest'
    opt.saveimg = True
    opt.phase = 'test'
    opt.isTrain=False
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    ssim, psnr = eval_in_training(model, opt.epoch)
