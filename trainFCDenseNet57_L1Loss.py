import os
import sys
import pickle
import pandas as pd
import random
import itertools
import logging
import numpy as np
import nibabel as nib

from statistics import median

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
import torch.autograd.profiler as profiler

from utils.utilities import DatasetUNetMax # DatasetB0toTensor

#  from SR_DenseNet_2018 import Net
from models.FC_DenseNet import FCDenseNet57

from utils.pLoss.perceptual_loss import PerceptualLoss
from pytorch_msssim import SSIM
from utils.utilities import tensorboard_images

# ----------------------------------------------------------------------
#Initial params
gpuID="0"
seed = 1701
num_workers= 16
batch_size = 128


log_path = r'/home/sciarra/From-B0-to-Tensor/Results/TBLogs/B0toTensor'
save_path = r'/home/sciarra/From-B0-to-Tensor/Results/Output/B0toTensor'

checkpoint2load = None
useCuda=True
do_val=True
pre_create_val=True
do_profile=False

#Training params
trainID="FCDenseNet_B0toTensor_L1Loss"
num_epochs = 1000
lr = 1e-4
patch_size=None#(32,32,32) #Set it to None if not desired
patchQ_len = 50
patches_per_volume = 25
log_freq = 10 
save_freq = 50 
preload_h5 = True

#Network Params
depth = 5
wf = 6 #number of filters in the first layer is 2**wf
padding = True
batch_norm = False
up_mode='upsample'
is3D=False
aux_handle_mode='ignore'
returnLatent=False
dropout_prob=0.0
save_size_interp=False
n_channels=7 #1
out_channels = 7

os.environ["CUDA_VISIBLE_DEVICES"] = gpuID
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__" :
    device = torch.device("cuda:0" if torch.cuda.is_available() and useCuda else "cpu")
    tb_writer = SummaryWriter(log_dir = os.path.join(log_path,trainID))
    os.makedirs(save_path, exist_ok=True)
    logname = os.path.join(save_path, 'log_'+trainID+'.txt')

    logging.basicConfig(filename=logname,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

    # ----------------------------------
    print('\n ... Loading data ... \n')
    train_in = nib.load('./data/train-4D-input.nii.gz').get_fdata()
    train_o_dwi = nib.load('./data/train-4D-dwi.nii.gz').get_fdata()
    train_o_dti = nib.load('./data/train-5D-tensor.nii.gz').get_fdata()

    test_in = nib.load('./data/test-4D-input.nii.gz').get_fdata()
    test_o_dwi = nib.load('./data/test-4D-dwi.nii.gz').get_fdata()
    test_o_dti = nib.load('./data/test-5D-tensor.nii.gz').get_fdata()
    print('\n ... data loaded ... \n')
    # -------------------------------------
    print('\n ... concatenating data for target ... \n')
    ## 
    ttrain_dwi = np.expand_dims(train_o_dwi, axis=3)
    train_out = np.concatenate((ttrain_dwi, train_o_dti), axis=3)

    ttest_dwi = np.expand_dims(test_o_dwi, axis=3)
    test_out = np.concatenate((ttest_dwi, test_o_dti), axis=3)

    print('\n ... concatenation done ... \n')

    print('\n', train_in.shape, train_out.shape)
    print('\n', test_in.shape, test_out.shape)
    # -------------------------------------
    trainset = DatasetUNetMax(train_in, train_out,transform=None)
    train_loader = DataLoader(dataset=trainset,batch_size=batch_size,shuffle=True, num_workers=num_workers)

    valset = DatasetUNetMax(test_in, test_out, transform=None)
    val_loader = DataLoader(dataset=valset,batch_size=batch_size,shuffle=True, num_workers=num_workers)
    
    
    # model = SRDenseNet(num_channels=n_channels, growth_rate=16, num_blocks=8, num_layers=8)
    model = FCDenseNet57(n_classes=7)  # Net()
    model.to(device)
    
    optimizer = Adam(params=model.parameters(), lr=lr)

    # loss_func = PerceptualLoss(device=device, loss_model="resnext1012D", n_level=1, resize=None, loss_type="L1")
    # loss_func = SSIM(data_range=1, channel=out_channels)
    loss_func = torch.nn.L1Loss()
    ssim = SSIM(data_range=1, channel=out_channels)

    scaler = GradScaler()

    if checkpoint2load:
        chk = torch.load(checkpoint2load, map_location=device)
        model.load_state_dict(chk['state_dict'])
        optimizer.load_state_dict(chk['optimizer'])
        scaler.load_state_dict(chk['AMPScaler'])   
        best_loss = chk['best_loss']    
        start_epoch = chk['epoch'] + 1
    else:
        start_epoch = 0
        best_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        #Train
        model.train()
        runningLoss = []
        train_loss = []
        print('Epoch '+ str(epoch)+ ': Train')
        with tqdm(total=len(train_loader)) as pbar:
            for i, (imgs_train, target_train) in enumerate(train_loader):
                try:
                    images = Variable(imgs_train).float().to(device)
                    gt = Variable(target_train).float().to(device)

                    optimizer.zero_grad()

                    with autocast():
                        out = model(images)
                        # print(out.shape)
                        #  print(images.shape, gt.shape)
                        loss = loss_func(out, gt)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    loss = round(loss.data.item(),4)
                    train_loss.append(loss)
                    runningLoss.append(loss)
                    logging.info('[%d/%d][%d/%d] Train Loss: %.4f' % ((epoch+1), num_epochs, i, len(train_loader), loss))
                    
                    #For tensorboard
                    if i % log_freq == 0:
                        niter = epoch*len(train_loader)+i
                        tb_writer.add_scalar('Train/Loss', median(runningLoss), niter)
                        tensorboard_images(tb_writer, images.detach().float().cpu(), out.detach().float().cpu(), gt, epoch, 'train')
                        runningLoss = []
                except EOFError as e: 
                    logging.error(str(e))
                pbar.update(1)
        if epoch % save_freq == 0:            
            checkpoint = {
                'epoch': epoch,
                'best_loss': best_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'AMPScaler': scaler.state_dict()         
            }
            torch.save(checkpoint, os.path.join(save_path, trainID+".pth.tar"))
        tb_writer.add_scalar('Train/EpochLoss', median(train_loss), epoch)

        torch.cuda.empty_cache()

        #Validate
        if val_loader:
            model.eval()
            with torch.no_grad():  
                runningLoss = []
                val_loss = []
                val_ssim = []
                print('Epoch '+ str(epoch)+ ': Val')
                with tqdm(total=len(val_loader)) as pbar:              
                    for i, (imgs_val, target_val) in enumerate(val_loader):
                        try:
                            images = Variable(imgs_val).float().to(device)
                            gt = Variable(target_val).float().to(device)

                            with autocast():
                                out = model(images)
                                loss = loss_func(out, gt)
                                loss_ssim = ssim(out, gt.type(out.dtype))

                            loss = round(loss.data.item(),4)
                            loss_ssim = round(loss_ssim.data.item(),4)
                            val_loss.append(loss)
                            val_ssim.append(loss_ssim)
                            runningLoss.append(loss)
                            logging.info('[%d/%d][%d/%d] Val Loss: %.4f' % ((epoch+1), num_epochs, i, len(val_loader), loss))
                            #For tensorboard
                            if i % log_freq == 0:
                                niter = epoch*len(val_loader)+i
                                tb_writer.add_scalar('Val/Loss', median(runningLoss), niter)
                                tensorboard_images(tb_writer, images.detach().float().cpu(), out.detach().float().cpu(), gt, epoch, 'val')
                                runningLoss = []
                        except EOFError as ex:
                            logging.error(ex)
                        pbar.update(1)
            if median(val_loss) < best_loss:
                best_loss = median(val_loss)
                checkpoint = {
                    'epoch': epoch,
                    'best_loss': best_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'AMPScaler': scaler.state_dict()         
                }
                torch.save(checkpoint, os.path.join(save_path, trainID+"_best.pth.tar"))
            tb_writer.add_scalar('Val/EpochLoss', median(val_loss), epoch)
            tb_writer.add_scalar('Val/EpochSSIM', median(val_ssim), epoch)
