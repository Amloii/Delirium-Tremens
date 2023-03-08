
#Import libraries
import os
from tqdm import tqdm
from ZeroShotObjectDetection import ZeroShotObjectDetection
from Preprocess import Preprocess
from CocoDatasetMerger import CocoDatasetMerger
import os
from tqdm import tqdm

config = {}
config['new_images_folder'] = 'C:/Users/danie/Pictures/homedeco/wine/'
config['dataset_input_folder'] = 'C:/Users/danie/Documents/Python Scripts/AdsViu/Datasets/Wine/'
config['new_category'] = 'Wine'
config['include_previous_images'] = False

config['model_zeroshotobjectdetection'] = "google/owlvit-base-patch32"
config['categories_list'] = ['bottle']
config['conf_threshold'] = 0.5
config['iou_threshold'] = 0.2


if __name__ == "__main__":
    object_detector = ZeroShotObjectDetection(config)
    coco_dataset_merger = CocoDatasetMerger(config)

    if config['include_previous_images']:
        
        for root, _, files in os.walk(config['dataset_input_folder'] + 'images\\', topdown=False):
            for im_path in tqdm(files[:10]):
                image_index = coco_dataset_merger.rescue_image_index(im_path)
                list_prediction = object_detector.predict(image_path=os.path.join(root, im_path))
                for prediction in list_prediction:
                    coco_dataset_merger.add_new_annotation(image_index, prediction)

    Preprocess.remove_duplicated_images(config['new_images_folder'])
    
    for root, _, files in os.walk(config['new_images_folder'], topdown=False):
        for im_path in tqdm(files[:10]):
            if im_path[-3:] in ['jpg', 'peg', 'png', 'fif']:
                image_index = coco_dataset_merger.add_new_image(image_path=os.path.join(root, im_path))
                list_prediction = object_detector.predict(image_path=os.path.join(root, im_path))
                for prediction in list_prediction:
                    coco_dataset_merger.add_new_annotation(image_index, prediction)

    coco_dataset_merger.save_dataset(overwrite=True)
    coco_dataset_merger.resume_classes()

