import os, sys
# PACKAGE_PARENT = '..'
# SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
# sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import json
import torch
import argparse
import logging
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from datetime import datetime
from torchvision import transforms
# from data.raster_label_visualizer import RasterLabelVisualizer
from models.siam_unet import SiamUnet
from utils.datasets import DisasterDataset
from utils.inference_evalmetrics import compute_eval_metrics, compute_confusion_mtrx
from utils.utils import AverageMeter, load_json_files
from utils.eval_building_level import _evaluate_tile, get_label_and_pred_polygons_for_tile_mask_input, allowed_classes
from torch.utils.tensorboard import SummaryWriter


def main():

    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    
    eval_results_val_dmg = pd.DataFrame(columns=['class', 'precision', 'recall', 'f1', 'accuracy'])
    eval_results_val_dmg_building_level = pd.DataFrame(columns=['class', 'precision', 'recall', 'f1', 'accuracy'])
    eval_results_val_bld = pd.DataFrame(columns=['class', 'precision', 'recall', 'f1', 'accuracy'])

    # Set up logger directory    
    logger_dir = os.path.join(config['out_dir'], config['test_name'], 'logs')
    os.makedirs(logger_dir, exist_ok=True)
    # Initialize logger instances
    logger_test= SummaryWriter(log_dir=logger_dir)
    
    # Load test data
    global test_dataset, test_loader, labels_set_dmg, labels_set_bld, viz
    label_map = load_json_files(args.label_map_json)
    test_dataset = load_dataset()
    
    labels_set_dmg = config['labels_dmg']
    labels_set_bld = config['labels_bld']
    
    #load model and its state from the given checkpoint
    model = SiamUnet()
    checkpoint_path = config['best_model']
    print('Loading checkpoint from {}'.format(checkpoint_path))
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device=device)
        logging.info(f"Using checkpoint at epoch {checkpoint['epoch']}, val f1 is {checkpoint.get('val_f1_avg', 'Not Available')}")
    except:
        print('No valid checkpoint is provided.')
        return
    
    # Running inference
    confusion_mtrx_df_val_dmg, confusion_mtrx_df_val_bld, confusion_mtrx_df_val_dmg_building_level = validate(test_dataset, \
        model, logger_test, evals_dir)
    
    # damage level eval validation (pixelwise)
    eval_results_val_dmg = compute_eval_metrics(labels_set_dmg, confusion_mtrx_df_val_dmg, eval_results_val_dmg)
    f1_harmonic_mean = 0
    metrics = 'f1'
    for index, row in eval_results_val_dmg.iterrows():
        if (int(row['class']) in labels_set_dmg[1:]) & (metrics == 'f1'):
            f1_harmonic_mean += 1.0/(row[metrics]+1e-10)
    f1_harmonic_mean = 4.0/f1_harmonic_mean
    eval_results_val_dmg = eval_results_val_dmg.append({'class':'harmonic-mean-of-all', \
        'precision':'-', 'recall':'-', 'f1':f1_harmonic_mean, 'accuracy':'-'}, ignore_index=True)
    
    # damage level eval validation (building-level)
    eval_results_val_dmg_building_level = compute_eval_metrics(labels_set_dmg, confusion_mtrx_df_val_dmg_building_level, eval_results_val_dmg_building_level)
    f1_harmonic_mean = 0
    metrics = 'f1'
    for index, row in eval_results_val_dmg_building_level.iterrows():
        if (int(row['class']) in labels_set_dmg[1:]) & (metrics == 'f1'):
            f1_harmonic_mean += 1.0/(row[metrics]+1e-10)
    f1_harmonic_mean = 4.0/f1_harmonic_mean
    eval_results_val_dmg_building_level = eval_results_val_dmg_building_level.append({'class':'harmonic-mean-of-all', \
        'precision':'-', 'recall':'-', 'f1':f1_harmonic_mean, 'accuracy':'-'}, ignore_index=True)                    

    # bld detection eval validation (pixelwise)
    eval_results_val_bld = compute_eval_metrics(labels_set_bld, confusion_mtrx_df_val_bld, eval_results_val_bld)

    # save confusion metrices
    confusion_mtrx_df_val_bld.to_csv(os.path.join(evals_dir, 'confusion_mtrx_bld.csv'), index=False)
    confusion_mtrx_df_val_dmg.to_csv(os.path.join(evals_dir, 'confusion_mtrx_dmg.csv'), index=False)
    confusion_mtrx_df_val_dmg_building_level.to_csv(os.path.join(evals_dir, 'confusion_mtrx_dmg_building_level.csv'), index=False)
    
    # save evalution metrics
    eval_results_val_bld.to_csv(os.path.join(evals_dir, 'eval_results_bld.csv'), index=False)
    eval_results_val_dmg.to_csv(os.path.join(evals_dir, 'eval_results_dmg.csv'), index=False)
    eval_results_val_dmg_building_level.to_csv(os.path.join(evals_dir, 'eval_results_dmg_building_level.csv'), index=False)
    
    return

# Validation 
def validate(loader, model):
    softmax = torch.nn.Softmax(dim=1)
    model.eval()  # put model to evaluation mode
    confusion_mtrx_df_val_dmg = pd.DataFrame(columns=['img_idx', 'class', 'true_pos', 'true_neg', 'false_pos', 'false_neg', 'total'])
    confusion_mtrx_df_val_bld = pd.DataFrame(columns=['img_idx', 'class', 'true_pos', 'true_neg', 'false_pos', 'false_neg', 'total'])
    confusion_mtrx_df_val_dmg_building_level = pd.DataFrame(columns=['img_idx', 'class', 'true_pos', 'true_neg', 'false_pos', 'false_neg', 'total'])
    
    with torch.no_grad():
        for img_idx, data in enumerate(tqdm(loader)): # assume batch size is 1
            c = data['pre_image'].size()[0]
            h = data['pre_image'].size()[1]
            w = data['pre_image'].size()[2]

            x_pre = data['pre_image'].reshape(1, c, h, w).to(device=device)
            x_post = data['post_image'].reshape(1, c, h, w).to(device=device)
            y_seg = data['building_mask'].to(device=device)  
            y_cls = data['damage_mask'].to(device=device)

            scores = model(x_pre, x_post)
                    
            # compute accuracy for segmenation model on pre_ images
            preds_seg_pre = torch.argmax(softmax(scores[0]), dim=1)

            # modify damage prediction based on UNet arm predictions       
            for c in range(0,scores[2].shape[1]):
                scores[2][:,c,:,:] = torch.mul(scores[2][:,c,:,:], preds_seg_pre)
            preds_cls = torch.argmax(softmax(scores[2]), dim=1)
            
            path_pred_mask = data['preds_img_dir'] +'.png'
            im = Image.fromarray(preds_cls.cpu().numpy()[0,:,:].astype(np.uint8))
            if not os.path.exists(os.path.split(data['preds_img_dir'])[0]):
                os.makedirs(os.path.split(data['preds_img_dir'])[0])
            im.save(path_pred_mask)

            # compute building-level confusion metrics
            pred_polygons_and_class, label_polygons_and_class = get_label_and_pred_polygons_for_tile_mask_input(y_cls.cpu().numpy().astype(np.uint8), path_pred_mask)
            results, list_preds, list_labels = _evaluate_tile(pred_polygons_and_class, label_polygons_and_class, allowed_classes, 0.1)
            total_objects = results[-1]
            for label_class in results:
                if label_class != -1:
                    true_pos_cls = results[label_class]['tp'] if 'tp' in results[label_class].keys() else 0
                    true_neg_cls = results[label_class]['tn'] if 'tn' in results[label_class].keys() else 0
                    false_pos_cls = results[label_class]['fp'] if 'fp' in results[label_class].keys() else 0
                    false_neg_cls = results[label_class]['fn'] if 'fn' in results[label_class].keys() else 0
                    confusion_mtrx_df_val_dmg_building_level = confusion_mtrx_df_val_dmg_building_level.append({'img_idx':img_idx, 'class':label_class, 'true_pos':true_pos_cls, 'true_neg':true_neg_cls, 'false_pos':false_pos_cls, 'false_neg':false_neg_cls, 'total':total_objects}, ignore_index=True)

            # compute comprehensive pixel-level comfusion metrics
            confusion_mtrx_df_val_dmg = compute_confusion_mtrx(confusion_mtrx_df_val_dmg, img_idx, labels_set_dmg, preds_cls, y_cls, y_seg)
            confusion_mtrx_df_val_bld = compute_confusion_mtrx(confusion_mtrx_df_val_bld, img_idx, labels_set_bld, preds_seg_pre, y_seg, [])

    return confusion_mtrx_df_val_dmg, confusion_mtrx_df_val_bld, confusion_mtrx_df_val_dmg_building_level

def load_dataset():
    splits = load_json_files(args.data_inference_dict)
    data_mean_stddev = load_json_files(args.data_mean_stddev)    
    test_ls = []
    for item, val in splits.items():
        test_ls += val['test']
    test_dataset = DisasterDataset(args.data_img_dir, test_ls, data_mean_stddev, transform=False, normalize=True)
    assert len(test_dataset) > 0
    return test_dataset

if __name__ == "__main__":
    main()