
#Import libraries
import os
from ZeroShotObjectDetection import ZeroShotObjectDetection
from CocoTransformer import CocoTransformer

config = {}
config['raw_image_folder'] = 'E:/Datasets/Furniture_v2/image'
config['model_zeroshotobjectdetection'] = "google/owlvit-base-patch32"
config['categories_list'] = ['couch', 'kitchen island', 'armoire', 'chair', 'lamp', 'vanity', 'mirror', 'footstool', 'shelve', 'curtain', 'art frame', 'flower pot', 'cushion', 'carpet', 'bed', 'cabinet', 'table']
config['conf_threshold'] = 0.2
config['iou_threshold'] = 0.2
config['test_split'] = 0.2
config['image_folder'] = 'E:/Datasets/Furniture_v2/'
config['output_folder'] = 'C:/Users/danie/Documents/GitHub/Delirium-Tremens/data/'

if __name__ == "__main__":
    
    object_detector = ZeroShotObjectDetection(config)
    coco_transformer = CocoTransformer(config)
    
    list_dict_detections = ZeroShotObjectDetection.predict(config['raw_image_folder'])
    CocoTransformer.transform(list_dict_detections)