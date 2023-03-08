from transformers import pipeline
import numpy as np
import re


class ZeroShotObjectDetection():
    
    def __init__(self, config):
        self.object_detector = pipeline(
        "zero-shot-object-detection", model=config['model_zeroshotobjectdetection'])
        self.categories_list = config['categories_list'] 
        self.categories_textseed = [["a image of a " + cat for cat in self.categories_list]]
        self.conf_threshold = config['conf_threshold'] 
        self.iou_threshold = config['iou_threshold'] 
        
    # Functions
    def __bb_intersection_over_union(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou
        
    def filter_nms_detections(results, iou_threshold=0.5):       
        bbox_list = []
        score_list = []
        label_list = []
        
        for result_dict in results[0]:
            bbox = [result_dict['box']['xmin'], 
                    result_dict['box']['ymin'],
                    result_dict['box']['xmax'],
                    result_dict['box']['ymax']]
            cat = re.findall('a image of a (.*)', result_dict['label'])[0]
            sco = round(result_dict['score'], 3)
            
            bbox_list.append(bbox)
            score_list.append(sco)
            label_list.append(cat)
        
        bbox_list_2 = []
        score_list_2 = []
        label_list_2 = []
        ignore_index = []
        for i in range(len(bbox_list)):
            if i not in ignore_index:
                group_list = [i]
                for j in range(i+1,len(bbox_list)):
                    iou = ZeroShotObjectDetection.__bb_intersection_over_union(bbox_list[i], bbox_list[j])
                    if iou > iou_threshold:
                        group_list.append(j)
                ignore_index += group_list
                keeped_case = group_list[np.argmax([score_list[x] for x in group_list])]
                bbox_mean = [int(x) for x in np.mean([bbox_list[x] for x in group_list], axis=0)]
                bbox_list_2.append(bbox_mean)
                score_list_2.append(score_list[keeped_case])
                label_list_2.append(label_list[keeped_case])

        return bbox_list_2, score_list_2, label_list_2

    def predict(self, image_path):

        try:
            results = self.object_detector(
                        image_path,
                        text_queries=self.categories_textseed,
                        threshold=self.conf_threshold,
                    )
                            

            boxes, scores, labels = ZeroShotObjectDetection.filter_nms_detections(results, iou_threshold=self.iou_threshold) 
            
            list_prediction = []
            for bbox, score, label in zip(boxes, scores, labels):
                            
                dict_row = {'im_path': image_path,
                            'bbox': str(bbox),
                            'score': str(score),
                            'labels': label}
                list_prediction.append(dict_row)
                
            return list_prediction
        
        except Exception as e:
            print(e)
            return {}
        
