from PIL import Image
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
import random
import os
class DisasterDataset(Dataset):
    def __init__(self, data_dir, incidents, transform:bool, normalize:bool):
        
        self.data_dir = data_dir
        self.images = os.listdir(self.data_dir)
        self.masks = os.listdir(self.data_dir.replace('images','masks'))
        self.transform = transform
        self.normalize = normalize
        self.incidents = incidents

    def __len__(self):
        return len(self.incidents)
    
    @classmethod
    def apply_transform(self, pre_img, post_img, mask, damage_class):
        '''
        apply tranformation functions on PIL images 
        '''
        if random.random() > 0.5:
            # Resize
            img_h = pre_img.size[0]
            img_w = pre_img.size[1]
            
            resize = transforms.Resize(size=(int(round(1.016*img_h)), int(round(1.016*img_w))))
            pre_img = resize(pre_img)
            post_img = resize(post_img)

            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(pre_img, output_size=(img_h, img_w))
            pre_img = TF.crop(pre_img, i, j, h, w)
            post_img = TF.crop(post_img, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            pre_img = TF.hflip(pre_img)
            post_img = TF.hflip(post_img)

        # Random vertical flipping
        if random.random() > 0.5:
            pre_img = TF.vflip(pre_img)
            post_img = TF.vflip(post_img)
        else:
            return pre_img, post_img
        
    @classmethod
    def apply_normalize(self, pre_img, post_img):
        '''
        apply normalize function on PIL images 
        '''
        # Pre disaster image PIL 
        #pre_img = np.array(pre_img)
        img = np.array(pre_img)
        mean_pre = (img[:,:,0].mean(), img[:,:,1].mean(), img[:,:,2].mean()) 
        stddev_pre = (img[:,:,0].std(), img[:,:,1].std(), img[:,:,2].std())
        
        norm_pre = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_pre, std=stddev_pre),])
                #transforms.ToPILImage()])
        
        pre_img = norm_pre(pre_img)#np.array(pre_img).astype(dtype='float64')/255.0)
        pre_img = np.array(pre_img)
        #pre_img = Image.fromarray(pre_img)
        
        # Post disaster image PIL 
        #post_img = np.array(post_img)
        img = np.array(post_img)
        mean_post = (img[:,:,0].mean(), img[:,:,1].mean(), img[:,:,2].mean()) 
        stddev_post = (img[:,:,0].std(), img[:,:,1].std(), img[:,:,2].std())
        
        norm_post = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_post, std=stddev_post),])
                #transforms.ToPILImage()])
        
        post_img = norm_post(np.array(post_img).astype(dtype='float64')/255.0)
        post_img = np.array(post_img)
        #post_img = Image.fromarray(post_img)

        return pre_img, post_img
        
    @classmethod
    def slice_tile(self, pre_img, post_img, mask, damage_class):
        
        img_h = pre_img.size[0]
        img_w = pre_img.size[1]
        
        h_idx = [0, 256, 512, 768]
        w_idx = [0, 256, 512, 768]

        img_h = 256
        img_w = 256
        
        sliced_dict = {}
        counter = 0
        for i in h_idx: 
            for j in w_idx:
                pre_img_sub = TF.crop(pre_img, i, j, img_h, img_w)
                post_img_sub = TF.crop(post_img, i, j, img_h, img_w)
                mask_sub = TF.crop(mask, i, j, img_h, img_w)
                damage_class_sub = TF.crop(damage_class, i, j, img_h, img_w)
                sliced_dict[str(counter)] = {'pre_image': TF.to_tensor(pre_img_sub), 
                                            'post_image': TF.to_tensor(post_img_sub), 
                                            'mask': TF.to_tensor(mask_sub), 
                                            'damage_class': TF.to_tensor(damage_class_sub)}
                counter += 1 # counts up to 15 i.e. 16 tiles of 256x256

        return sliced_dict

    def __getitem__(self, idx):
        path = self.data_dir # Path of image folder
        event = self.incidents[idx]
        
        image_list = [path+'/'+i for i in self.images]   # Path of each images
        # get files using glob
        pre_img_file = glob([i for i in image_list if '_pre_' in i][idx])
        pre_img = Image.open(pre_img_file[0])
        post_img_file = glob([i for i in image_list if '_post_' in i][idx])
        post_img = Image.open(post_img_file[0])

        path = self.data_dir.replace('images','masks') # Path of mask folder
        mask_list = [path+'/'+i for i in self.masks]   # Path of each mask
        # get files using glob
        mask_file = glob([i for i in mask_list if '_pre_' in i][idx])
        mask = Image.open(mask_file[0])
        damage_class_file = glob([i for i in mask_list if '_post_' in i][idx])
        damage_class = Image.open(damage_class_file[0])

        if self.transform is True:
            pre_img, post_img = self.apply_transform(pre_img, post_img)
        
        if self.normalize is True:
            pre_img, post_img = self.apply_normalize(pre_img, post_img)
        
        sliced_dict = self.slice_tile(pre_img, post_img, mask, damage_class)
        
        # Add prediction path to each sub dict in sliced_dict
        for i, data in sliced_dict.items():
            data['preds_img_dir'] = self.data_dir.replace('images','tiles')+'/'+self.incidents[idx]+f'_tile_{i}'
            data['full_image_dir'] = self.data_dir.replace('images','tiles')+'/'+self.incidents[idx]
            data['building_mask'] = torch.from_numpy(np.array(mask)).type(torch.LongTensor)
            data['damage_mask'] = torch.from_numpy(np.array(damage_class)).type(torch.LongTensor)
        
        return sliced_dict
