import os, sys
import json
import torch
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
from utils.inferencedatasets import DisasterDataset
from utils.inference_evalmetrics import compute_eval_metrics, compute_confusion_mtrx
from utils.utils import AverageMeter, load_json_files
from utils.eval_building_level import _evaluate_tile, get_label_and_pred_polygons_for_tile_mask_input, allowed_classes
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

@hydra.main(config_path='./configs',
			config_name='config.yaml',
            version_base=None)
def main(cfg: DictConfig):

    device = torch.device(cfg.params.device if torch.cuda.is_available() else "cpu")
    
    eval_results_val_dmg = pd.DataFrame(columns=['class', 'precision', 'recall', 'f1', 'accuracy'])
    eval_results_val_dmg_building_level = pd.DataFrame(columns=['class', 'precision', 'recall', 'f1', 'accuracy'])
    eval_results_val_bld = pd.DataFrame(columns=['class', 'precision', 'recall', 'f1', 'accuracy'])

    # Set up evals directory
    evals_dir = os.path.join(cfg.experiment.out_dir, cfg.experiment.test_name, 'evals')
    os.makedirs(evals_dir, exist_ok=True)
    # Set up logger directory    
    logger_dir = os.path.join(cfg.experiment.out_dir, cfg.experiment.test_name, 'logs')
    os.makedirs(logger_dir, exist_ok=True)
    # Initialize logger instances
    logger_test= SummaryWriter(log_dir=logger_dir)
    
    # Load test data
    global test_dataset, test_loader, labels_set_dmg, labels_set_bld, viz
    label_map = load_json_files(cfg.params.label_map)
    test_dataset = load_dataset()
    
    labels_set_dmg = cfg.params.labels_dmg
    labels_set_bld = cfg.params.labels_bld
    
    #load model and its state from the given checkpoint
    model = SiamUnet()
    best_mdl_path = cfg.model.best_mdl
    print(f'Loading checkpoint from {best_mdl_path}')

    checkpoint = torch.load(best_mdl_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device=device)

    # Running inference
    confusion_mtrx_df_val_dmg, confusion_mtrx_df_val_bld, confusion_mtrx_df_val_dmg_building_level = validate(device, test_dataset, \
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

    # building detection eval validation (pixelwise)
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
def validate(device, loader, model, logger_test, evals_dir):
    
    softmax = torch.nn.Softmax(dim=1)
    model.eval()  # put model to evaluation mode
    confusion_mtrx_df_val_dmg = pd.DataFrame(columns=['img_idx', 'class', 'true_pos', 'true_neg', 'false_pos', 'false_neg', 'total'])
    confusion_mtrx_df_val_bld = pd.DataFrame(columns=['img_idx', 'class', 'true_pos', 'true_neg', 'false_pos', 'false_neg', 'total'])
    confusion_mtrx_df_val_dmg_building_level = pd.DataFrame(columns=['img_idx', 'class', 'true_pos', 'true_neg', 'false_pos', 'false_neg', 'total'])
    
    with torch.no_grad():
        for img_idx, data in enumerate(tqdm(loader)): # assume batch size is 1
            
            #Initiate empty tensor of size 1024,1024 to later stich each tile together
            fullimg_tensor = torch.empty((1, 1024, 1024), dtype=torch.float64)
            fullimg_seg_pre_tensor = torch.empty((1, 1024, 1024), dtype=torch.float64)

            m, n = 0, 0
            ind = [0, 256, 512, 768, 1024]
            
            # Running loop for each tile in set of 16 tiles
            for k, tiledict in data.items():
                c = tiledict['pre_image'].size()[0]
                h = tiledict['pre_image'].size()[1]
                w = tiledict['pre_image'].size()[2]

                x_pre = tiledict['pre_image'].reshape(1, c, h, w).to(device=device)
                x_post = tiledict['post_image'].reshape(1, c, h, w).to(device=device)


                scores = model(x_pre, x_post)
                    
                # compute accuracy for segmenation model on pre_ images
                preds_seg_pre = torch.argmax(softmax(scores[0]), dim=1)
                preds_seg_post = torch.argmax(softmax(scores[1]), dim=1)
                
                # modify damage prediction based on UNet arm predictions       
                for c in range(0,scores[2].shape[1]):
                    scores[2][:,c,:,:] = torch.mul(scores[2][:,c,:,:], preds_seg_pre)
                preds_cls = torch.argmax(softmax(scores[2]), dim=1)
                
                save_tiles = False # change it to save tile output
                if save_tiles == True:
                    path_pred_mask = data['preds_img_dir'] +'.png'
                    im = Image.fromarray(preds_cls.cpu().numpy()[0,:,:].astype(np.uint8))
                    
                #Stitch preds_cls to 1024x1024 image
                m = int(k)//4
                fullimg_tensor[0,ind[m]:ind[m+1],ind[n]:ind[n+1]] = preds_cls.cpu()
                fullimg_seg_pre_tensor[0,ind[m]:ind[m+1],ind[n]:ind[n+1]] = preds_seg_pre.cpu()
                n += 1
                if n == 4: 
                    n = 0
                    
            preds_seg_pre = fullimg_seg_pre_tensor.to(device=device)
            preds_cls = fullimg_tensor.to(device=device)
            
            #print(pred_cls.shape, preds_seg_pre.shape)
            
            pred_mask = Image.fromarray(fullimg_tensor.cpu().numpy()[0,:,:].astype(np.uint8))
            path_pred_mask_full = tiledict['full_image_dir'] +'.png'
            pred_mask.save(path_pred_mask_full)
            
            y_seg = tiledict['building_mask'].to(device=device)  
            y_cls = tiledict['damage_mask'].to(device=device)

            # compute building-level confusion metrics
            pred_polygons_and_class, label_polygons_and_class = get_label_and_pred_polygons_for_tile_mask_input(y_cls.cpu().numpy().astype(np.uint8), path_pred_mask_full)
            results, list_preds, list_labels = _evaluate_tile(pred_polygons_and_class, label_polygons_and_class, allowed_classes, 0.1)
            total_objects = results[-1]
            #print(total_objects)
            for label_class in results:
                if label_class != -1:
                    true_pos_cls = results[label_class]['tp'] if 'tp' in results[label_class].keys() else 0
                    true_neg_cls = results[label_class]['tn'] if 'tn' in results[label_class].keys() else 0
                    false_pos_cls = results[label_class]['fp'] if 'fp' in results[label_class].keys() else 0
                    false_neg_cls = results[label_class]['fn'] if 'fn' in results[label_class].keys() else 0
                    confusion_mtrx_df_val_dmg_building_level = confusion_mtrx_df_val_dmg_building_level.append({'img_idx':img_idx, 'class':label_class, 'true_pos':true_pos_cls, 'true_neg':true_neg_cls, 'false_pos':false_pos_cls, 'false_neg':false_neg_cls, 'total':total_objects}, ignore_index=True)
            
            print()
            # compute comprehensive pixel-level comfusion metrics
            confusion_mtrx_df_val_dmg = compute_confusion_mtrx(confusion_mtrx_df_val_dmg, img_idx, labels_set_dmg, preds_cls, y_cls, y_seg)
            confusion_mtrx_df_val_bld = compute_confusion_mtrx(confusion_mtrx_df_val_bld, img_idx, labels_set_bld, preds_seg_pre, y_seg, [])

    return confusion_mtrx_df_val_dmg, confusion_mtrx_df_val_bld, confusion_mtrx_df_val_dmg_building_level

def load_dataset():
    # Setup to be able to use config.yaml
    cwd = hydra.utils.get_original_cwd() 
    cfg_path = os.path.join(cwd, "configs", "config.yaml")
    cfg = OmegaConf.load(cfg_path)
    
    # Core function:
    # Setting up to run inference on Sample data folder in the repo; update path in config to change the data dir
    images_dir = cfg.paths.sampledata + '/images'
    incidents = list(set(['_'.join(i.split('_')[:2]) for i in os.listdir(images_dir)])) # Unique incidents 
    test_dataset = DisasterDataset(images_dir, incidents, transform=False, normalize=False)
    assert len(test_dataset) > 0
    return test_dataset

if __name__ == "__main__":
    main()