import os
import sys

from .grit_src.image_dense_captions import init_demo, dense_pred_to_caption, dense_pred_to_caption_only_name,dense_pred_to_caption_tuple
from detectron2.data.detection_utils import read_image

class DenseCaptioning():
    def __init__(self, device):
        self.device = device
        self.demo =  None


    def initialize_model(self, model_weight):
        self.demo = init_demo(self.device, model_weight=model_weight)
        
    def initialize_model_det(self, model_weight):
        self.demo = init_demo(self.device, model_weight = model_weight, task="ObjectDet")

    def run_caption_tensor(self,img):
        predictions = self.demo.run_on_image(img)
        new_caption = dense_pred_to_caption_tuple(predictions)
        return new_caption

    def run_caption_tensor_batch(self, images)
        predictions = self.demo.run_on_batch(images)
        new_caption_batch = [dense_pred_to_caption_tuple(prediction) for prediction in predictions]
        return new_caption_batch

