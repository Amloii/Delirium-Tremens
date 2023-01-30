
#Import libraries
import os
from ZeroShotObjectDetection import ZeroShotObjectDetection
from CocoTransformer import CocoTransformer

config = {}
config['raw_image_folder'] = 'C:/Users/danie/Pictures/homedeco/'
config['model_zeroshotobjectdetection'] = "google/owlvit-base-patch32"
config['categories_list'] = ['bottle']
config['conf_threshold'] = 0.2
config['iou_threshold'] = 0.2
config['test_split'] = 0.2
config['image_folder'] = 'C:/Users/danie/Documents/Python Scripts/AdsViu/Datasets/Wine/'
config['output_folder'] = 'C:/Users/danie/Documents/Python Scripts/AdsViu/Datasets/Wine/'

if __name__ == "__main__":
    
    object_detector = ZeroShotObjectDetection(config)
    coco_transformer = CocoTransformer(config)
    
    list_dict_detections = object_detector.predict(images_dir=config['raw_image_folder'])
    coco_transformer.transform(list_dict_detections)