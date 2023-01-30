import json
import pandas as pd
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import shutil
import re
import ast
from sklearn.model_selection import train_test_split

class CocoTransformer():
    
    def __init__(self, config):
        
        self.test_split = config['test_split']
        self.raw_image_folder = config['raw_image_folder']

        self.image_folder_train = config['image_folder'] + 'train/'
        if not os.path.exists(self.image_folder_train):
            os.makedirs(self.image_folder_train)
        
        self.image_folder_test = config['image_folder'] + 'test/'
        if not os.path.exists(self.image_folder_test):
            os.makedirs(self.image_folder_test)
            
    def create_dataset(self, image_list, image_folder):
        
            dataset_dict = {
                            "info": {},
                            "licenses": [],
                            "images": [],
                            "annotations": [],
                            "categories": []
                        }
            
            for index_cat, cat in enumerate(list(set(self.annon_df.labels))):
                
                dataset_dict['categories'].append({
                    'id': index_cat,
                    'name': cat.replace(' ','_')
                })
                
            sub_index = 0 
            for num_image, image_path in tqdm(enumerate(image_list), total=len(image_list)):

                extension = re.findall('\.(.*)$', image_path)[0]
                image_name = str(num_image).zfill(7) + '.' + extension
                shutil.copy(self.raw_image_folder + image_path, image_folder + image_name)

                with Image.open(self.raw_image_folder + image_path) as imag:
                    width, height = imag.size

                dataset_dict['images'].append({
                    'coco_url': '',
                    'date_captured': '',
                    'file_name': image_name,
                    'flickr_url': '',
                    'id': num_image,
                    'license': 0,
                    'width': width,
                    'height': height
                })

                temp_df = self.annon_df[self.annon_df.im_path == image_path]
                for _, row_annon in temp_df.iterrows():
                    sub_index = sub_index + 1
                    box = ast.literal_eval(row_annon.bbox)
                    w = box[2] - box[0]
                    h = box[3] - box[1]
                    x_1 = box[0]
                    y_1 = box[1]
                    bbox = [x_1, y_1, w, h]
                    cat = self.cat2id_map[row_annon.labels]
                    
                    dataset_dict['annotations'].append({
                        'area': w*h,
                        'bbox': bbox,
                        'category_id': cat,
                        'id': sub_index,
                        'image_id': num_image,
                    })
            
            with open(image_folder + 'annon_coco.json', 'w') as f:
                json.dump(dataset_dict, f)

            print(F'Dataset created in {image_folder}annon_coco.json')
        
    def transform(self, list_dict_annon):
        
        self.annon_df = pd.DataFrame(list_dict_annon)
        self.cat2id_map = {cat:index_cat for index_cat, cat in enumerate(list(set(self.annon_df.labels)))}

        image_list = list(set(self.annon_df.im_path))
        image_list_train, image_list_test = train_test_split(image_list, test_size=self.test_split, random_state=42)  

        CocoTransformer.create_dataset(self, image_list_train, self.image_folder_train)  
        CocoTransformer.create_dataset(self, image_list_test, self.image_folder_test)  