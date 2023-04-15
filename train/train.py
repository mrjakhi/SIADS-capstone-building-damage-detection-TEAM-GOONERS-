import os, sys, math, torch, shutil, logging, torchvision
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from torchvision import transforms
from time import localtime, strftime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from utils.raster_label_visualizer import RasterLabelVisualizer
from models.siam_unet import SiamUnet
from utils.utils import AverageMeter, load_json_files
from utils.train_evalmetrics import compute_eval_metrics, compute_confusion_mtrx
from utils.datasets import DisasterDataset
import hydra
from omegaconf import DictConfig

@hydra.main(config_path='../configs',
			config_name='config.yaml',
            version_base=None)
def main(cfg: DictConfig):
    device = torch.device(cfg.params.device if torch.cuda.is_available() else "cpu")
    
    global viz, labels_set_dmg, labels_set_bld
    global xBD_train, xBD_val
    global train_loader, val_loader, test_loader
    global weights_loss, mode
    
    xBD_train, xBD_val = load_dataset()

    train_loader = DataLoader(xBD_train, batch_size=cfg.params.batch_size, shuffle=True, num_workers=8, pin_memory=False)
    val_loader = DataLoader(xBD_val, batch_size=cfg.params.batch_size, shuffle=False, num_workers=8, pin_memory=False)

    labels_set_dmg = cfg.params.labels_dmg
    labels_set_bld = cfg.params.labels_bld
    mode = cfg.params.mode

    eval_results_tr_dmg = pd.DataFrame(columns=['epoch', 'class', 'precision', 'recall', 'f1', 'accuracy'])
    eval_results_tr_bld = pd.DataFrame(columns=['epoch', 'class', 'precision', 'recall', 'f1', 'accuracy'])
    eval_results_val_dmg = pd.DataFrame(columns=['epoch', 'class', 'precision', 'recall', 'f1', 'accuracy'])
    eval_results_val_bld = pd.DataFrame(columns=['epoch', 'class', 'precision', 'recall', 'f1', 'accuracy'])

    # set up logger directory    
    logger_dir = os.path.join(cfg.experiment.out_dir, cfg.experiment.experiment_name, 'logs')
    os.makedirs(logger_dir, exist_ok=True)
    
    # define model
    model = SiamUnet().to(device=device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.params.lr)
    
    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2000, verbose=True)
    
    # Initialize
    starting_epoch = 1
    best_acc = 0.0
    weights_loss = cfg.params.weights_loss
    weights_seg_tf = torch.FloatTensor(cfg.params.weights_seg)
    weights_damage_tf = torch.FloatTensor(cfg.params.weights_damage)
    
    # Loss function
    criterion_seg_1 = nn.CrossEntropyLoss(weight=weights_seg_tf).to(device=device)
    criterion_seg_2 = nn.CrossEntropyLoss(weight=weights_seg_tf).to(device=device)
    criterion_damage = nn.CrossEntropyLoss(weight=weights_damage_tf).to(device=device)
    
    # Initialize logger instances
    logger_train = SummaryWriter(log_dir=logger_dir)
    logger_val = SummaryWriter(log_dir=logger_dir)
    
    # Training epochs
    epoch = starting_epoch
    step_tr = 1
    epochs = cfg.epochs
    
    while (epoch <= epochs):
        # Train
        logger_train.add_scalar( 'learning_rate', optimizer.param_groups[0]["lr"], epoch)
        train_start_time = datetime.now()
        model, optimizer, step_tr, confusion_mtrx_df_tr_dmg, confusion_mtrx_df_tr_bld = train(train_loader, model, \
            criterion_seg_1, criterion_seg_2, criterion_damage, optimizer, epochs, epoch, step_tr, \
            logger_train, device)
        train_duration = datetime.now() - train_start_time
        logger_train.add_scalar('time_training', train_duration.total_seconds(), epoch)
        
        # damage level eval train
        eval_results_tr_dmg = compute_eval_metrics(epoch, labels_set_dmg, confusion_mtrx_df_tr_dmg, eval_results_tr_dmg)
        eval_results_tr_dmg_epoch = eval_results_tr_dmg.loc[eval_results_tr_dmg['epoch'] == epoch,:]
        f1_harmonic_mean = 0
        for metrics in ['f1']:
            for index, row in eval_results_tr_dmg_epoch.iterrows():
                if int(row['class']) in labels_set_dmg[1:]:
                    if metrics == 'f1':
                        f1_harmonic_mean += 1.0/(row[metrics]+1e-10)
        f1_harmonic_mean = 4.0/f1_harmonic_mean
        logger_train.add_scalar( 'tr_dmg_harmonic_mean_f1', f1_harmonic_mean, epoch)
        # bld level eval train
        eval_results_tr_bld = compute_eval_metrics(epoch, labels_set_bld, confusion_mtrx_df_tr_bld, eval_results_tr_bld)
        eval_results_tr_bld_epoch = eval_results_tr_bld.loc[eval_results_tr_bld['epoch'] == epoch,:]
        
        # Validation
        eval_start_time = datetime.now()
        confusion_mtrx_df_val_dmg, confusion_mtrx_df_val_bld, losses_val = validation(val_loader, model, \
            criterion_seg_1, criterion_seg_2, criterion_damage, epoch, logger_val)
        eval_duration = datetime.now() - eval_start_time
        logger_val.add_scalar('time_validation', eval_duration.total_seconds(), epoch)
        # decay Learning Rate
        scheduler.step(losses_val)
        
        # damage level eval validation
        eval_results_val_dmg = compute_eval_metrics(epoch, labels_set_dmg, confusion_mtrx_df_val_dmg, eval_results_val_dmg)
        eval_results_val_dmg_epoch = eval_results_val_dmg.loc[eval_results_val_dmg['epoch'] == epoch,:]
        for metrics in ['f1']:
            for index, row in eval_results_val_dmg_epoch.iterrows():
                if int(row['class']) in labels_set_dmg[1:]:
                    if metrics == 'f1':
                        f1_harmonic_mean += 1.0/(row[metrics]+1e-10)
        f1_harmonic_mean = 4.0/f1_harmonic_mean
        logger_val.add_scalar( 'val_dmg_harmonic_mean_f1', f1_harmonic_mean, epoch)
        
        # bld level eval validation
        eval_results_val_bld = compute_eval_metrics(epoch, labels_set_bld, confusion_mtrx_df_val_bld, eval_results_val_bld)
        eval_results_val_bld_epoch = eval_results_val_bld.loc[eval_results_val_bld['epoch'] == epoch,:]
        
        # Select best model based on average accuracy across all classes 
        val_acc_avg = f1_harmonic_mean
        if val_acc_avg > best_acc:
            best_acc = max(val_acc_avg, best_acc)
            best_model = model
        epoch += 1
    # Close Tensorboard Summarywriters    
    logger_train.flush()
    logger_train.close()
    logger_val.flush()
    logger_val.close()    
    # Save best model:
    # add code here
    return best_model

# Function to train model in each epoch
def train(loader, model, criterion_seg_1, criterion_seg_2, criterion_damage, optimizer, epoch, step_tr, logger_train, device):
    confusion_mtrx_df_tr_dmg = pd.DataFrame(columns=['epoch', 'batch_idx', 'class', 'true_pos', 'true_neg', 'false_pos', 'false_neg', 'total_pixels'])
    confusion_mtrx_df_tr_bld = pd.DataFrame(columns=['epoch', 'batch_idx', 'class', 'true_pos', 'true_neg', 'false_pos', 'false_neg', 'total_pixels'])

    losses_tr = AverageMeter()
    loss_seg_pre = AverageMeter()
    loss_seg_post = AverageMeter()
    loss_dmg = AverageMeter()
    
    for batch_idx, data in enumerate(tqdm(loader)): 
                         
        x_pre = data['pre_image'].to(device=device)  # move to device, e.g. GPU
        x_post = data['post_image'].to(device=device)  
        y_seg = data['building_mask'].to(device=device)  
        y_cls = data['damage_mask'].to(device=device)  
        
        model.train()
        optimizer.zero_grad()
        scores = model(x_pre, x_post)
        
        # modify damage prediction based on UNet arm
        softmax = torch.nn.Softmax(dim=1)
        preds_seg_pre = torch.argmax(softmax(scores[0]), dim=1)
        for c in range(0,scores[2].shape[1]):
            scores[2][:,c,:,:] = torch.mul(scores[2][:,c,:,:], preds_seg_pre)
            
        loss = weights_loss[0]*criterion_seg_1(scores[0], y_seg) + weights_loss[1]*criterion_seg_2(scores[1], y_seg) + weights_loss[2]*criterion_damage(scores[2], y_cls)
        loss_seg_pre_tr = criterion_seg_1(scores[0], y_seg)
        loss_seg_post_tr = criterion_seg_2(scores[1], y_seg)
        loss_dmg_tr = criterion_damage(scores[2], y_cls)
        
        losses_tr.update(loss.item(), x_pre.size(0))
        loss_seg_pre.update(loss_seg_pre_tr.item(), x_pre.size(0))
        loss_seg_post.update(loss_seg_post_tr.item(), x_pre.size(0))
        loss_dmg.update(loss_dmg_tr.item(), x_pre.size(0))

        loss.backward()  # compute gradients
        optimizer.step()
        
        # compute predictions & confusion metrics
        softmax = torch.nn.Softmax(dim=1)
        preds_seg_pre = torch.argmax(softmax(scores[0]), dim=1)
        preds_seg_post = torch.argmax(softmax(scores[1]), dim=1)
        preds_cls = torch.argmax(softmax(scores[2]), dim=1)

        confusion_mtrx_df_tr_dmg = compute_confusion_mtrx(confusion_mtrx_df_tr_dmg, epoch, batch_idx, labels_set_dmg, preds_cls, y_cls, y_seg)
        confusion_mtrx_df_tr_bld = compute_confusion_mtrx(confusion_mtrx_df_tr_bld, epoch, batch_idx, labels_set_bld, preds_seg_pre, y_seg, [])
        
    logger_train.add_scalars('loss_tr', {'_total':losses_tr.avg, '_seg_pre': loss_seg_pre.avg, '_seg_post': loss_seg_post.avg, '_dmg': loss_dmg.avg}, epoch)    
    step_tr += 1
    return model, optimizer, step_tr, confusion_mtrx_df_tr_dmg, confusion_mtrx_df_tr_bld

# Function to validate model in each epoch
def validation(loader, model, criterion_seg_1, criterion_seg_2, criterion_damage, epoch, logger_val):
    confusion_mtrx_df_val_dmg = pd.DataFrame(columns=['epoch', 'batch_idx', 'class', 'true_pos', 'true_neg', 'false_pos', 'false_neg', 'total_pixels'])
    confusion_mtrx_df_val_bld = pd.DataFrame(columns=['epoch', 'batch_idx', 'class', 'true_pos', 'true_neg', 'false_pos', 'false_neg', 'total_pixels'])
    losses_val = AverageMeter()
    loss_seg_pre = AverageMeter()
    loss_seg_post = AverageMeter()
    loss_dmg = AverageMeter()
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader)):
            x_pre = data['pre_image'].to(device=device)  # move to device, e.g. GPU
            x_post = data['post_image'].to(device=device)  
            y_seg = data['building_mask'].to(device=device)  
            y_cls = data['damage_mask'].to(device=device)  

            model.eval()  # put model to evaluation mode
            scores = model(x_pre, x_post)

            # modify damage prediction based on UNet arm
            softmax = torch.nn.Softmax(dim=1)
            preds_seg_pre = torch.argmax(softmax(scores[0]), dim=1)
            for c in range(0,scores[2].shape[1]):
                scores[2][:,c,:,:] = torch.mul(scores[2][:,c,:,:], preds_seg_pre)
            
            loss = weights_loss[0]*criterion_seg_1(scores[0], y_seg) + weights_loss[1]*criterion_seg_2(scores[1], y_seg) + weights_loss[2]*criterion_damage(scores[2], y_cls)
            loss_seg_pre_val = criterion_seg_1(scores[0], y_seg)
            loss_seg_post_val = criterion_seg_2(scores[1], y_seg)
            loss_dmg_val = criterion_damage(scores[2], y_cls)
        
            losses_val.update(loss.item(), x_pre.size(0))
            loss_seg_pre.update(loss_seg_pre_val.item(), x_pre.size(0))
            loss_seg_post.update(loss_seg_post_val.item(), x_pre.size(0))
            loss_dmg.update(loss_dmg_val.item(), x_pre.size(0))

            # compute predictions & confusion metrics
            softmax = torch.nn.Softmax(dim=1)
            preds_seg_pre = torch.argmax(softmax(scores[0]), dim=1)
            preds_seg_post = torch.argmax(softmax(scores[1]), dim=1)
            preds_cls = torch.argmax(softmax(scores[2]), dim=1)
            
            confusion_mtrx_df_val_dmg = compute_confusion_mtrx(confusion_mtrx_df_val_dmg, epoch, batch_idx, labels_set_dmg, preds_cls, y_cls, y_seg)
            confusion_mtrx_df_val_bld = compute_confusion_mtrx(confusion_mtrx_df_val_bld, epoch, batch_idx, labels_set_bld, preds_seg_pre, y_seg, [])
    
    logger_val.add_scalars('loss_val', {'_total': losses_val.avg, '_seg_pre': loss_seg_pre.avg, '_seg_post': loss_seg_post.avg, '_dmg': loss_dmg.avg}, epoch)
    return confusion_mtrx_df_val_dmg, confusion_mtrx_df_val_bld, losses_val.avg

def load_dataset():
    splits = load_json_files(config['disaster_splits_json'])
    data_mean_stddev = load_json_files(config['disaster_mean_stddev'])

    train_ls = [] 
    val_ls = []
    for _, val in splits.items():
        train_ls += val['train'] 
        val_ls += val['val']
    xBD_train = DisasterDataset(config['data_dir_shards'], config['shard_no'], 'train', data_mean_stddev, transform=True, normalize=True)
    xBD_val = DisasterDataset(config['data_dir_shards'], config['shard_no'], 'val', data_mean_stddev, transform=False, normalize=True)

    print('xBD_disaster_dataset train length: {}'.format(len(xBD_train)))
    print('xBD_disaster_dataset val length: {}'.format(len(xBD_val)))

    return xBD_train, xBD_val

if __name__ == "__main__":
    main()