import streamlit as st
import os, sys, cv2
import pandas as pd
import numpy as np
from io import BytesIO, StringIO
from  PIL import Image
from glob import glob
import hydra, os, sys, json
import torch
import torch.nn as nn
from omegaconf import DictConfig
from omegaconf import OmegaConf
from tqdm import tqdm
from datetime import datetime
from torchvision import transforms
from models.siam_unet import SiamUnet
from utils.appdatasets import DisasterDataset
from utils.inference_evalmetrics import compute_eval_metrics, compute_confusion_mtrx
from utils.utils import AverageMeter, load_json_files
from utils.raster_label_visualizer import RasterLabelVisualizer
#Execute in terminal:
#python -m streamlit run SIADS-capstone-building-damage-detection-TEAM-GOONERS-/local/streamlit_app.py

APP_TITLE = 'Building Damage Detection (Beta)'
APP_SUB_TITLE = 'Compare Pre And Post Satellite Images From Disaster Hit Areas'

path = './Sample data/app_data/'
imgpath = path+'images/'

def main():
    st.set_page_config(APP_TITLE)
    st.title(APP_TITLE)
    st.caption(APP_SUB_TITLE)
    
    # Load Data
    st.sidebar.markdown('<p class="font">Upload Images</p>', unsafe_allow_html=True)
    with st.sidebar.expander("About the App"):
        st.write("""
            Upload Pre and Post Disaster Satellite Images of the area and hit 'Assess Building Damage'
        """)

        st.write("""
            Upload Satellite Images :
        """)
        uploadedfile_pre = st.file_uploader("Pre Disaster", type=['jpg','png','jpeg'], key=1)
        if uploadedfile_pre is not None:
            save_uploaded_file(uploadedfile_pre)

        uploadedfile_post = st.file_uploader("Post Disaster", type=['jpg','png','jpeg'], key=2)
        if uploadedfile_post is not None:
            save_uploaded_file(uploadedfile_post)
            
        assess = st.button("Assess Building Damage")
        # st.write(assess)
        # st.write(os.getcwd())
    
    if uploadedfile_pre is not None and uploadedfile_post is not None:
        files = os.listdir(imgpath)#[f for f in os.listdir(imgpath) if os.path.isfile(f)]
        pre_image = glob([imgpath+'/'+f for f in files if '_pre_' in f][0])
        pre_image = Image.open(pre_image[0])
        post_image = glob([imgpath+'/'+f for f in files if '_post_' in f][0])
        post_image = Image.open(post_image[0])
    # Run Inference Model
    
    # Display Images   
    # if uploaded_file is not None:
    # image = Image.open(uploaded_file)
    
    col1, col2 = st.columns( [0.5, 0.5])
    
    with col1:
        st.markdown('<p style="text-align: center;">Pre Disaster</p>',unsafe_allow_html=True)
        if assess: st.image(pre_image,width=300)  

    with col2:
        st.markdown('<p style="text-align: center;">Post Disaster</p>',unsafe_allow_html=True)
        if assess: st.image(post_image,width=300) 
    # Display Metrics
    if assess:
        st.write("Generating damage assessment mask with post disaster image")
        inference_model()
    
def save_uploaded_file(uploadedfile):
  with open(os.path.join(imgpath[:-1],uploadedfile.name),"wb") as f:
     f.write(uploadedfile.getbuffer())
  return st.success("Saved file :{} in the Images directory".format(uploadedfile.name))

@hydra.main(config_path='./configs',
			config_name='config.yaml',
            version_base=None)
def inference_model(cfg: DictConfig):
    device = torch.device(cfg.params.device if torch.cuda.is_available() else "cpu")
    
    # Load two images here in the desired format, pre and post
    test_dataset = load_data()
    
    # Labels
    labels_set_dmg = cfg.params.labels_dmg
    labels_set_bld = cfg.params.labels_bld
    
    # Load model and its state from the saved best model
    model = SiamUnet()
    best_mdl_path = cfg.model.best_mdl
    print(f'Loading checkpoint from {best_mdl_path}')

    checkpoint = torch.load(best_mdl_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device=device)
    # Predict
    softmax = torch.nn.Softmax(dim=1)
    model.eval()  # put model to evaluation mode
    with torch.no_grad():
        for img_idx, data in enumerate(tqdm(test_dataset)): # assume batch size is 1
            # fullimg_tensor = torch.empty((0, 256, 256), dtype=torch.float64)
            fullimg_tensor = torch.empty((1, 1024, 1024), dtype=torch.float64)
            m, n = 0, 0
            ind = [0, 256, 512, 768, 1024]
            for k, tiledict in data.items():
                c = tiledict['pre_image'].size()[0]
                h = tiledict['pre_image'].size()[1]
                w = tiledict['pre_image'].size()[2]

                x_pre = tiledict['pre_image'].reshape(1, c, h, w).to(device=device)
                x_post = tiledict['post_image'].reshape(1, c, h, w).to(device=device)

                scores = model(x_pre, x_post)
                # print(x_pre.shape, scores[0].shape, scores[2].shape)
                # compute accuracy for segmenation model on pre_ images
                preds_seg_pre = torch.argmax(softmax(scores[0]), dim=1)
                preds_seg_post = torch.argmax(softmax(scores[1]), dim=1)
                
                # modify damage prediction based on UNet arm predictions       
                for c in range(0,scores[2].shape[1]):
                    scores[2][:,c,:,:] = torch.mul(scores[2][:,c,:,:], preds_seg_pre)
                preds_cls = torch.argmax(softmax(scores[2]), dim=1)
                
                path_pred_mask = tiledict['preds_img_dir'] +'.png'
                im = Image.fromarray(preds_cls.cpu().numpy()[0,:,:].astype(np.uint8))
                im.save(path_pred_mask)
                
                #Stitch preds_cls to 1024x1024 image
                m = int(k)//4
                fullimg_tensor[0,ind[m]:ind[m+1],ind[n]:ind[n+1]] = preds_cls.cpu()
                n += 1
                if n == 4: 
                    n = 0
    #print(fullimg_tensor.shape)
    full_im = Image.fromarray(fullimg_tensor.cpu().numpy()[0,:,:].astype(np.uint8))
    path_pred_mask_full = tiledict['full_image_dir'] +'.png'
    full_im.save(path_pred_mask_full)
    viz = RasterLabelVisualizer(label_map=cfg.params.label_map)#"./utils/xBD_label_map.json")
    im, buf = viz.show_label_raster(full_im, size=(15, 15))
    
    return st.pyplot(viz.plot_color_legend()), st.image(im,width=700)

def load_data():
    # Setup to be able to use config.yaml
    cwd = os.getcwd()#hydra.utils.get_original_cwd() 
    cfg_path = os.path.join(cwd, "configs", "config.yaml")
    cfg = OmegaConf.load(cfg_path)
    
    images_dir = cfg.paths.testdata + '/images'
    incidents = list(set(['_'.join(i.split('_')[:2]) for i in os.listdir(images_dir)])) # Unique incidents
    print(incidents)
    test_dataset = DisasterDataset(images_dir, incidents, transform=False, normalize=False)
    assert len(test_dataset) > 0
    return test_dataset

if __name__ == "__main__":
    files = glob(imgpath+'*')
    for f in files:
        os.remove(f)
    main()