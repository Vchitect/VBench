import os
import sys

from .grit_src.image_dense_captions import image_caption_api, init_demo, dense_pred_to_caption, dense_pred_to_caption_only_name,dense_pred_to_caption_tuple
from detectron2.data.detection_utils import read_image

class DenseCaptioning():
    def __init__(self, device):
        self.device = device
        self.demo =  None


    def initialize_model(self, model_weight):
        self.demo = init_demo(self.device, model_weight=model_weight)
        
    def initialize_model_det(self, model_weight):
        self.demo = init_demo(self.device, model_weight = model_weight, task="ObjectDet")
    
    def image_dense_caption(self, image_src):
        dense_caption = image_caption_api(image_src, self.device)
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        print("Step2, Dense Caption:\n")
        print(dense_caption)
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        return dense_caption
    
    def run_caption_api(self,image_src):
        img = read_image(image_src, format="BGR")
        print(img.shape)
        predictions, visualized_output = self.demo.run_on_image(img)
        new_caption = dense_pred_to_caption_only_name(predictions)
        return new_caption

    def run_caption_tensor(self,img):
        predictions, visualized_output = self.demo.run_on_image(img)
        new_caption = dense_pred_to_caption_tuple(predictions)
        return new_caption, visualized_output

    def run_det_tensor(self,img):
        predictions, visualized_output = self.demo.run_on_image(img)
        return predictions, visualized_output

