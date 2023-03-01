from models.Discriminator import Discriminator
from models.VGG19 import Vgg19
from models.DINet import DINet
from utils.training_utils import get_scheduler, update_learning_rate,GANLoss
from torch.utils.data import DataLoader
from dataset.dataset_DINet_frame import DINetDataset
from sync_batchnorm import convert_model
from config.config import DINetTrainingOptions

import random
import numpy as np
import os
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    '''
        frame training code of DINet
        we use coarse-to-fine training strategy
        so you can use this code to train the model in arbitrary resolution
    '''
    # load config
    opt = DINetTrainingOptions().parse_args()
    # set seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    # load training data in memory
    train_data = DINetDataset(opt.train_data,opt.augment_num,opt.mouth_region_size)
    training_data_loader = DataLoader(dataset=train_data,  batch_size=opt.batch_size, shuffle=True,drop_last=True)
    train_data_length = len(training_data_loader)
    # init network
    net_g = DINet(opt.source_channel,opt.ref_channel,opt.audio_channel).cuda()
    net_dI = Discriminator(opt.source_channel ,opt.D_block_expansion, opt.D_num_blocks, opt.D_max_features).cuda()
    net_vgg = Vgg19().cuda()
    # parallel
    net_g = nn.DataParallel(net_g)
    net_g = convert_model(net_g)
    net_dI = nn.DataParallel(net_dI)
    net_vgg = nn.DataParallel(net_vgg)
    # setup optimizer
    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr_g)
    optimizer_dI = optim.Adam(net_dI.parameters(), lr=opt.lr_dI)
    # coarse2fine
    if opt.coarse2fine:
        print('loading checkpoint for coarse2fine training: {}'.format(opt.coarse_model_path))
        checkpoint = torch.load(opt.coarse_model_path)
        net_g.load_state_dict(checkpoint['state_dict']['net_g'])
    # set criterion
    criterionGAN = GANLoss().cuda()
    criterionL1 = nn.L1Loss().cuda()
    # set scheduler
    net_g_scheduler = get_scheduler(optimizer_g, opt.non_decay, opt.decay)
    net_dI_scheduler = get_scheduler(optimizer_dI, opt.non_decay, opt.decay)
    # start train
    for epoch in range(opt.start_epoch, opt.non_decay+opt.decay+1):
        net_g.train()
        for iteration, data in enumerate(training_data_loader):
            # read data
            source_image_data,source_image_mask, reference_clip_data,deepspeech_feature = data
            source_image_data = source_image_data.float().cuda()
            source_image_mask = source_image_mask.float().cuda()
            reference_clip_data = reference_clip_data.float().cuda()
            deepspeech_feature = deepspeech_feature.float().cuda()
            # network forward
            fake_out = net_g(source_image_mask,reference_clip_data,deepspeech_feature)
            # down sample output image and real image
            fake_out_half = F.avg_pool2d(fake_out, 3, 2, 1, count_include_pad=False)
            target_tensor_half = F.interpolate(source_image_data, scale_factor=0.5, mode='bilinear')
            # (1) Update D network
            optimizer_dI.zero_grad()
            # compute fake loss
            _,pred_fake_dI = net_dI(fake_out)
            loss_dI_fake = criterionGAN(pred_fake_dI, False)
            # compute real loss
            _,pred_real_dI = net_dI(source_image_data)
            loss_dI_real = criterionGAN(pred_real_dI, True)
            # Combined DI loss
            loss_dI = (loss_dI_fake + loss_dI_real) * 0.5
            loss_dI.backward(retain_graph=True)
            optimizer_dI.step()
            # (2) Update G network
            _, pred_fake_dI = net_dI(fake_out)
            optimizer_g.zero_grad()
            # compute perception loss
            perception_real = net_vgg(source_image_data)
            perception_fake = net_vgg(fake_out)
            perception_real_half = net_vgg(target_tensor_half)
            perception_fake_half = net_vgg(fake_out_half)
            loss_g_perception = 0
            for i in range(len(perception_real)):
                loss_g_perception += criterionL1(perception_fake[i], perception_real[i])
                loss_g_perception += criterionL1(perception_fake_half[i], perception_real_half[i])
            loss_g_perception = (loss_g_perception / (len(perception_real) * 2)) * opt.lamb_perception
            # # gan dI loss
            loss_g_dI = criterionGAN(pred_fake_dI, True)
            # combine perception loss and gan loss
            loss_g =  loss_g_perception + loss_g_dI
            loss_g.backward()
            optimizer_g.step()

            print(
                "===> Epoch[{}]({}/{}):  Loss_DI: {:.4f} Loss_GI: {:.4f} Loss_perception: {:.4f} lr_g = {:.7f} ".format(
                    epoch, iteration, len(training_data_loader), float(loss_dI), float(loss_g_dI),  float(loss_g_perception),optimizer_g.param_groups[0]['lr']))

        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_dI_scheduler, optimizer_dI)
        #checkpoint
        if epoch %  opt.checkpoint == 0:
            if not os.path.exists(opt.result_path):
                os.mkdir(opt.result_path)
            model_out_path = os.path.join(opt.result_path, 'netG_model_epoch_{}.pth'.format(epoch))
            states = {
                'epoch': epoch + 1,
                'state_dict': {'net_g': net_g.state_dict(), 'net_dI': net_dI.state_dict()},#
                'optimizer': {'net_g': optimizer_g.state_dict(), 'net_dI': optimizer_dI.state_dict()}#
            }
            torch.save(states, model_out_path)
            print("Checkpoint saved to {}".format(epoch))
