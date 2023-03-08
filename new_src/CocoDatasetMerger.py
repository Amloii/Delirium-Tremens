import json
import os
import pandas as pd
import numpy as np
import shutil
import uuid
import ast
from PIL import Image
from datetime import datetime

class CocoDatasetMerger():
    
    def __init__(self, config):
        self.dataset_input_folder = config['dataset_input_folder']
        self.annotation_path = config['dataset_input_folder'] + 'annotations.json'
        if not os.path.exists(self.annotation_path):
            self.dataset = {"info": {},
                        "licenses": [],
                        "images": [],
                        "annotations": [],
                        "categories": []
                        }
            if not os.path.exists(config['dataset_input_folder'] + '/images/'):
                os.mkdir(config['dataset_input_folder'] + '/images/')
        else:
            f = open(self.annotation_path)
            self.dataset = json.load(f)
            
        self.cat2id_map = {c['name']: c['id'] for c in self.dataset['categories']}
        self.dataset_image_folder = config['dataset_input_folder']
        self.new_category = config['new_category']
        if config['new_category'] not in self.cat2id_map.keys():
            if len(self.dataset['categories']) > 0:
                new_index = np.max([c['id'] for c in self.dataset['categories']]) + 1
            else:
                new_index = 0
            self.cat2id_map[config['new_category']] = new_index
            self.dataset['categories'].append({'id': int(new_index), 'name': config['new_category']})

    def add_new_image(self, image_path):
        if len(self.dataset['images']) > 0:
            new_image_index = np.max([c['id'] for c in self.dataset['images']]) + 1
        else:
            new_image_index = 0
        new_image_name = 'images\\' + str(uuid.uuid4()) + '.jpg'

        shutil.copy(image_path, os.path.join(self.dataset_image_folder, new_image_name))
        
        im = Image.open(image_path)
        w, h = im.size
        
        row_image = {'width': int(w),
                    'height': int(h),
                    'id': int(new_image_index),
                    'file_name': new_image_name}
        self.dataset['images'].append(row_image)
        
        return new_image_index
        
        
    def rescue_image_index(self, image_path):
        image_index = [x['id'] for x in self.dataset['images'] if x['file_name'] == 'images\\' + image_path]
        if len(image_index) > 0:
            return image_index[0]
        else:
            raise ValueError('Could not found the image')
        
        
    def add_new_annotation(self, image_index, prediction):
        if len(self.dataset['annotations']) > 0:
            new_annotation_index = np.max([c['id'] for c in self.dataset['annotations']]) + 1
        else:
            new_annotation_index = 0
        bbox = ast.literal_eval(prediction['bbox']) 
        w = int(bbox[2] - bbox[0])
        h = int(bbox[3] - bbox[1])
        x_1 = int(bbox[0])
        y_1 = int(bbox[1])
        row_annotation = {'id': int(new_annotation_index),
                            'image_id': int(image_index),
                            'category_id': int(self.cat2id_map[self.new_category]),
                            'segmentation': [],
                            'bbox': [x_1, y_1, w, h],
                            'ignore': 0,
                            'iscrowd': 0,
                            'area': int(bbox[2] * bbox[3])}
        self.dataset['annotations'].append(row_annotation)
        
        
    def save_dataset(self, overwrite=False):
        
        if overwrite:
            with open(self.annotation_path, 'w') as f:
                json.dump(self.dataset, f)
        else:
            date_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            with open(self.dataset_input_folder + date_time + '_annotations.json', 'w') as f:
                json.dump(self.dataset, f, indent=4)
                
                
    def resume_classes(self):
        
        annon_df = pd.DataFrame(self.dataset['annotations'])
        cats_df = pd.DataFrame(self.dataset['categories'])
        id2cat_map = pd.Series(cats_df.name.values,index=cats_df.id).to_dict()
        annon_df['category_name'] = annon_df['category_id'].map(id2cat_map)
        
        print('-'*20)
        print('DATASET:')
        print(self.dataset_input_folder )
        print('CATEGORIES:')
        print(annon_df.category_name.value_counts())
        print('-'*20)
