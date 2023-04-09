import json
import os
import random
import shutil
import time

from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)

from PIL import Image

import shapely.wkt

from tqdm import tqdm

# ------------------------------------------------------------------------------------------------------------- #
# Variable intializations

color_dict = {'none': 'c', 'no-damage': 'w', 'minor-damage': 'darkseagreen',
              'major-damage': 'orange', 'destroyed': 'red', 'un-classified': 'b'}


# Define the directory to search

# directory = os.getcwd()
# print(directory)
# files = os.listdir(directory)
# print(files)


# directory_label = r"C:\Users\mjakhi\PycharmProjects\SIADS-capstone-building-damage-detection-TEAM-GOONERS-\labels"
# directory_img = r"C:\Users\mjakhi\PycharmProjects\SIADS-capstone-building-damage-detection-TEAM-GOONERS-\images"

# # Define the file extensions to search for
# image_extensions = [".png"]
# json_extensions = [".json"]

# ------------------------------------------------------------------------------------------------------------- #

# Function definitions

def generate_dataframe(ann_dict):
    """
    Generate main annotation dataframe.
    :return: ann_df (pandas DataFrame)
    """
    # Remove text file
    skipfile = "./skipped.txt"
    if os.path.exists(skipfile):
        os.remove(skipfile)

    ann_list = []
    for k, ann in ann_dict.items():
        if ann['features']['xy']:
            # Get features
            feature_type = []
            uids = []
            pixwkts = []
            dmg_cats = []
            imids = []
            types = []

            for i in ann['features']['xy']:
                feature_type.append(i['properties']['feature_type'])
                uids.append(i['properties']['uid'])
                pixwkts.append(i['wkt'])
                if 'subtype' in list(i['properties'].keys()):
                    dmg_cats.append(i['properties']['subtype'])
                else:
                    dmg_cats.append("none")
                imids.append(ann['metadata']['img_name'].split('_')[1])
                types.append(ann['metadata']['img_name'].split('_')[2])

            geowkts = [i['wkt'] for i in ann['features']['lng_lat']]
            # Get Metadata
            cols = list(ann['metadata'].keys())
            vals = list(ann['metadata'].values())

            newcols = ['obj_type', 'img_id', 'type', 'pixwkt', 'geowkt', 'dmg_cat', 'uid'] + cols
            newvals = [[f, _id, t, pw, gw, dmg, u] + vals for f, _id, t, pw, gw, dmg, u in
                       zip(feature_type, imids, types, pixwkts, geowkts, dmg_cats, uids)]
            df = pd.DataFrame(newvals, columns=newcols)
            ann_list.append(df)
        else:
            # Skip images with no annotations
            append_write = 'a' if os.path.exists(skipfile) else 'w'
            with open(skipfile, append_write) as skipped:
                skipped.write(os.path.basename(k) + '\n')

    df_new = pd.concat(ann_list, ignore_index=True)

    return df_new


def pre_post_split(ann_df):
    """
    Generate pre-disaster and post-disaster dataframes from main dataframe.
    :return: pre disaster and post disaster dataframes
    """
    pre_df = ann_df.loc[ann_df["type"] == 'pre']
    post_df = ann_df.loc[ann_df["type"] == 'post']
    return pre_df, post_df


def view_pre_post(ann_df, disaster="guatemala-volcano", imid="00000000"):
    """
    Visualise the effect of a disaster via pre and post disaster images.
    :param disaster: Disaster name from the following list:
    ['guatemala-volcano', 'hurricane-florence', 'hurricane-harvey',
    'hurricane-matthew', 'hurricane-michael', 'mexico-earthquake',
    'midwest-flooding', 'palu-tsunami', 'santa-rosa-wildfire','socal-fire']
    :param imid: img id
    :return: None
    """
    assert disaster in ann_df.disaster.unique()

    split_df = pre_post_split(ann_df)
    prdf = split_df[0]
    # print(prdf)

    podf = split_df[1]
    pre_df = prdf[(prdf['img_id'] == imid) & (prdf['disaster'] == disaster)]
    post_df = podf[(podf['img_id'] == imid) & (podf['disaster'] == disaster)]

    assert len(pre_df) + len(post_df) == len(ann_df[(ann_df['disaster'] == disaster) & (ann_df['img_id'] == imid)])

    fig, axes = plt.subplots(1, 2, figsize=(15, 15), sharey=True)

    # Get pre and post disaster images
    print(pre_df.img_name.unique()[0])
    pre_im = plt.imread(os.path.join(directory_img, pre_df.img_name.unique()[0]))
    post_im = plt.imread(os.path.join(directory_img, post_df.img_name.unique()[0]))
    axes[0].imshow(pre_im)
    axes[1].imshow(post_im)

    # Get pre-disaster building polygons
    for _, row in pre_df.iterrows():
        poly = shapely.wkt.loads(row['pixwkt'])
        dmg_stat = row['dmg_cat']
        axes[0].plot(*poly.exterior.xy, color='c')

    # Get post-disaster building polygons
    for _, row in post_df.iterrows():
        poly = shapely.wkt.loads(row['pixwkt'])
        dmg_stat = row['dmg_cat']
        axes[1].plot(*poly.exterior.xy, color=color_dict[dmg_stat])

    axes[0].title.set_text('Pre Disaster')
    axes[0].axis('off')
    axes[1].title.set_text('Post Disaster')
    axes[1].axis('off')

    plt.suptitle(disaster + "_" + imid, fontsize=14, fontweight='bold')

    plt.show()


# ------------------------------------------------------------------------------------------------------------- #
# directory_label = r"C:\Users\mjakhi\PycharmProjects\SIADS-capstone-building-damage-detection-TEAM-GOONERS-\labels"
# directory_img = r"C:\Users\mjakhi\PycharmProjects\SIADS-capstone-building-damage-detection-TEAM-GOONERS-\images"

# imgs = sorted([os.path.join(directory_img, im) for im in os.listdir(directory_img)])
# jsons = sorted([os.path.join(directory_label, im) for im in os.listdir(directory_label)])

# ------------------------------------------------------------------------------------------------------------- #

# Create annotation dictionary and annotation dataframe

def ann_dictionary_ann_df(directory_label_path):
    print('Loading annotations into memory...')
    jsons = sorted([os.path.join(directory_label_path, im) for im in os.listdir(directory_label_path)])

    tic = time.time()
    anns = [json.load(open(ann, 'r')) for ann in jsons]
    # print(anns)
    print('Done (t={:0.2f}s)'.format(time.time() - tic))

    ann_dict = dict(zip(jsons, anns))
    # print(ann_dict)

    # Create annotation dataframe
    print('Creating annotation dataframe...')
    tic = time.time()
    ann_df = generate_dataframe(ann_dict)
    # print("ann_df columns: ", ann_df.columns)
    # print("ann_df : ", ann_df)

    print('Done (t={:0.2f}s)'.format(time.time() - tic))

    return ann_df

# ------------------------------------------------------------------------------------------------------------- #

# Pre-post disaster visualization

# view_pre_post(ann_df, 'guatemala-volcano', '00000015')

# ------------------------------------------------------------------------------------------------------------- #
