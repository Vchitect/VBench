import os
import torch
from itertools import compress
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image

# constants
WINDOW_NAME = "GRiT"
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
from vbench.utils import CACHE_DIR

# sys.path.insert(0, f"{CUR_DIR}/../")
# print(CUR_DIR)
import sys
sys.path.append(os.path.join(CUR_DIR, './centernet2/'))
from centernet.config import add_centernet_config

from .grit.config import add_grit_config
from .grit.predictor import VisualizationDemo

class ObjDescription:
    def __init__(self, object_descriptions):
        self.data = object_descriptions

    def __getitem__(self, item):
        assert type(item) == torch.Tensor
        assert item.dim() == 1
        if len(item) > 0:
            assert item.dtype == torch.int64 or item.dtype == torch.bool
            if item.dtype == torch.int64:
                return ObjDescription([self.data[x.item()] for x in item])
            elif item.dtype == torch.bool:
                return ObjDescription(list(compress(self.data, item)))

        return ObjDescription(list(compress(self.data, item)))

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "ObjDescription({})".format(self.data)

def dense_pred_to_caption(predictions):
    boxes = predictions["instances"].pred_boxes if predictions["instances"].has("pred_boxes") else None
    object_description = predictions["instances"].pred_object_descriptions.data
    new_caption = ""
    for i in range(len(object_description)):
        new_caption += (object_description[i] + ": " + str([int(a) for a in boxes[i].tensor.cpu().detach().numpy()[0]])) + "; "
    return new_caption

def dense_pred_to_caption_only_name(predictions):
    object_description = predictions["instances"].pred_object_descriptions.data
    new_caption = ",".join(object_description)
    del predictions
    return new_caption

def dense_pred_to_caption_tuple(predictions):
    boxes = predictions["instances"].pred_boxes if predictions["instances"].has("pred_boxes") else None
    object_description = predictions["instances"].pred_object_descriptions.data
    object_type = predictions["instances"].det_obj.data
    new_caption = []
    for i in range(len(object_description)):
        # new_caption += (object_description[i] + ": " + str([int(a) for a in boxes[i].tensor.cpu().detach().numpy()[0]])) + "; "
        new_caption.append((object_description[i], [int(a) for a in boxes[i].tensor.cpu().detach().numpy()[0]], object_type))
    return new_caption

def setup_cfg(args):
    cfg = get_cfg()
    if args["cpu"]:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_grit_config(cfg)
    cfg.merge_from_file(args["config_file"])
    cfg.merge_from_list(args["opts"])
    # Set score_threshold for builtin models
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args["confidence_threshold"]
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args["confidence_threshold"]
    if args["test_task"]:
        cfg.MODEL.TEST_TASK = args["test_task"]
    cfg.MODEL.BEAM_SIZE = 1
    cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
    cfg.USE_ACT_CHECKPOINT = False
    cfg.freeze()
    return cfg


def get_parser(device, model_weight=f"{CACHE_DIR}/grit_model/grit_b_densecap_objectdet.pth"):
    arg_dict = {'config_file': f"{CUR_DIR}/configs/GRiT_B_DenseCap_ObjectDet.yaml", 'cpu': False, 'confidence_threshold': 0.5, 'test_task': 'DenseCap', 'opts': ["MODEL.WEIGHTS", model_weight]}
    if device.type == "cpu":
        arg_dict["cpu"] = True
    return arg_dict

def image_caption_api(image_src, device, model_weight):
    args2 = get_parser(device, model_weight)
    cfg = setup_cfg(args2)
    demo = VisualizationDemo(cfg)
    if image_src:
        img = read_image(image_src, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)
        new_caption = dense_pred_to_caption(predictions)
    return new_caption

def init_demo(device, model_weight, task="DenseCap"):
    args2 = get_parser(device, model_weight)
    if task!="DenseCap":
        args2["test_task"]=task
    cfg = setup_cfg(args2)
    
    demo = VisualizationDemo(cfg)
    return demo
