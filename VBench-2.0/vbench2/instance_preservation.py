# Copyright (c) Tencent Inc. All rights reserved.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import cv2
import argparse
import os.path as osp
import torch
import decord
decord.bridge.set_bridge('torch')
import vbench2.hack_registry
from mmengine.config import Config, DictAction
from mmengine.runner.amp import autocast
from mmengine.dataset import Compose
from mmengine.utils import ProgressBar
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg
import supervision as sv
import warnings
from vbench2.utils import load_dimension_info
from tqdm import tqdm
warnings.filterwarnings("ignore")

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
MASK_ANNOTATOR = sv.MaskAnnotator()


class LabelAnnotator(sv.LabelAnnotator):

    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


LABEL_ANNOTATOR = LabelAnnotator(text_padding=4,
                                 text_scale=0.5,
                                 text_thickness=1)


def inference_detector(model,
                       image_path,
                       texts,
                       test_pipeline,
                       max_dets=100,
                       score_thr=0.3,
                       use_amp=False,
                       show=False,
                       annotation=False):
    data_info = dict(img_id=0, img_path=image_path, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])

    with autocast(enabled=use_amp), torch.no_grad():
        output = model.test_step(data_batch)[0]
        pred_instances = output.pred_instances
        pred_instances = pred_instances[pred_instances.scores.float() >
                                        score_thr]

    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]

    pred_instances = pred_instances.cpu().numpy()
    boxes=pred_instances['bboxes']
 
    return len(boxes)

def camera_motion(prompt_dict_ls, camera):
    sim = []
    video_results = []

    for prompt_dict in tqdm(prompt_dict_ls):
        label = prompt_dict['auxiliary_info']
        video_path = prompt_dict['video_list']
    
        end_frame=-1
        scene_list = split_video_into_scenes(video_path, 5.0)
        if len(scene_list)!=0:
            end_frame = int(scene_list[0][1].get_frames())
        video_reader = decord.VideoReader(video_path)
        video = video_reader.get_batch(range(len(video_reader))) 
        frame_count, height, width = video.shape[0], video.shape[1], video.shape[2]
        video = video.permute(0, 3, 1, 2)[None].float().cuda() # B T C H W
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        predict_results = camera.predict(video, fps, end_frame)
        video_score = 1.0 if label in predict_results else 0.0
        video_results.append({'caption': video_path, 'final_score': video_score})
        sim.append(video_score)
    
    avg_score = np.mean(sim)
    return avg_score, video_results

def Instance_Preservation(prompt_dict_ls, model, test_pipeline):
    final_score=0
    processed_json=[]
    for prompt_dict in tqdm(prompt_dict_ls):
        label = prompt_dict['auxiliary_info']
        if ' and ' in label:
            label = [i for i in label.split('and')]
        else:
            label=[label]
        video_paths = prompt_dict['video_list']
        for video_path in video_paths:
        
            f_score=0
            video_reader = decord.VideoReader(video_path)
            for frame in video_reader:
                fram = frame.detach().cpu().numpy()
                first_frame_bgr = fram[:, :, [2, 1, 0]].astype('uint8')
                image_path = os.path.join(os.path.dirname(video_path), 'division_used.png')
                cv2.imwrite(image_path, first_frame_bgr)
                score=0
                for idx, item in enumerate(label):
                    num = int(item.split('.')[0].strip())
                    texts = [[item.split('.')[1].strip()]] + [[' ']]
                    model.reparameterize(texts)
                    pred_num = inference_detector(model,
                                                image_path,
                                                texts,
                                                test_pipeline,
                                                100,
                                                0.28,
                                                use_amp=False,
                                                show=False,
                                                annotation=False)
                    if pred_num==num:
                        score+=1
                f_score+=score/len(label)
            if os.path.exists(image_path):
                os.remove(image_path)
            new_item={
                'video_path':video_path,
                'video_results':f_score/len(video_reader)
            }
            processed_json.append(new_item)
            final_score += f_score/len(video_reader)
            
    return final_score/len(prompt_dict_ls), processed_json
    
def compute_instance_preservation(json_dir, device, submodules_dict, **kwargs):
    cfg = Config.fromfile("vbench2/third_party/YOLO-World/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py")
    model = init_detector(cfg, checkpoint=submodules_dict['model'], device=device)
    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    test_pipeline = Compose(test_pipeline_cfg)
    
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='instance_preservation', lang='en')
    all_results, video_results = Instance_Preservation(prompt_dict_ls, model, test_pipeline)
    all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results