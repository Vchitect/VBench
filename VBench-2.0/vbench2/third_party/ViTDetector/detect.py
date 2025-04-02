import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import cv2
import mmcv
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmyolo.registry import VISUALIZERS
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from vbench2.third_party.ViTDetector.logger import create_logger
from vbench2.third_party.ViTDetector.config import get_config
from vbench2.third_party.ViTDetector.models import build_model
from torch.nn.parallel import DataParallel
import math
from torchvision import datasets, transforms
from timm.data.transforms import _pil_interp
from collections import defaultdict
from typing import List, Dict
 

logger = create_logger(output_dir='./', dist_rank=0, name="abnormality_detection")

class Detector:
    def __init__(self, config_file, weight_file, device='cuda'):
        self.model_human = init_detector(config_file, weight_file, device='cuda')
        self.model_face_hand = init_detector(config_file, weight_file, device='cuda')

        # change data loader
        self.model_human.cfg.test_dataloader.dataset.pipeline[0].type = 'mmdet.LoadImageFromNDArray'
        self.test_pipeline = Compose(self.model_human.cfg.test_dataloader.dataset.pipeline)


    def inference_detector(self, model, image, texts, test_pipeline, score_thr=0.3):
        data_info = dict(img_id=0, img=image, texts=texts)
        data_info = test_pipeline(data_info)
        data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                        data_samples=[data_info['data_samples']])

        with torch.no_grad():
            output = model.test_step(data_batch)[0]
            pred_instances = output.pred_instances
            pred_instances = pred_instances[pred_instances.scores.float() >
                                            score_thr]
        output.pred_instances = pred_instances
        return output
    
    def detect_video(self, video_path):
        human_text = "human"
        human_texts = [[t.strip()] for t in human_text.split(',')] + [[' ']]
        # face,hand detection
        face_hand_text = "face,hand"
        face_hand_texts = [[t.strip()] for t in face_hand_text.split(',')] + [[' ']]

        # text parameter modification
        self.model_human.reparameterize(human_texts)
        self.model_face_hand.reparameterize(face_hand_texts)

        video_reader = mmcv.VideoReader(video_path)
        total_frames = len(video_reader)

        results = []  # save the final results


        for frame_idx, frame in tqdm(enumerate(video_reader), total=total_frames, desc="processing video frame", disable=True):
            annotated_frame = frame.copy()

            # 1. human detection
            result_human = self.inference_detector(self.model_human, frame, human_texts, self.test_pipeline, score_thr=0.1)
            pred_instances = result_human.pred_instances
            human_bboxes = pred_instances.bboxes.cpu().numpy()

            for person_idx, bbox in enumerate(human_bboxes):
                # human box results
                result_item = {
                    "frame_index": frame_idx,
                    "person_index": person_idx,
                    "bbox": bbox.tolist(),
                    "label": "human"
                }
                results.append(result_item)

                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, "human", (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)

                # crop human
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # 2. human face, hand detection
                result_face_hand = self.inference_detector(self.model_face_hand, crop, face_hand_texts, self.test_pipeline, score_thr=0.1)
                pred_face_hand = result_face_hand.pred_instances
                fh_bboxes = pred_face_hand.bboxes.cpu().numpy()
                fh_labels = pred_face_hand.labels.cpu().numpy()

                for fh_bbox, label_idx in zip(fh_bboxes, fh_labels):
                    adj_bbox = [
                        float(fh_bbox[0] + x1),
                        float(fh_bbox[1] + y1),
                        float(fh_bbox[2] + x1),
                        float(fh_bbox[3] + y1)
                    ]
                    # 0:face, 1:hand
                    if label_idx == 0:
                        label = "face"
                        color = (255, 0, 0)  
                    elif label_idx == 1:
                        label = "hand"
                        color = (0, 0, 255)  
                    else:
                        label = "unknown"
                        color = (0, 255, 255)  

                    result_item = {
                        "frame_index": frame_idx,
                        "person_index": person_idx,
                        "bbox": adj_bbox,
                        "label": label
                    }
                    results.append(result_item)

        return results


class Analyzer:
    def __init__(self, model_configs, device='cuda', batch_size=128, class_thresholds=None):
        self.device = device
        self.models = {}
        self.transforms = {}
        self._initialize_models(model_configs)
        self.batch_size = batch_size
        self.class_threshold = class_thresholds

    def _initialize_models(self, model_configs):
        for category, config in model_configs.items():
            model, model_config = self._build_model(config["cfg_path"], config["weight_path"])
            self.models[category] = DataParallel(model).to(self.device).eval()
            self.transforms[category] = self._build_transform(model_config)

    def _build_model(self, cfg_path, weight_path):
        args = type('Args', (), {
            'cfg': cfg_path, 
            'opts': None,
            'local_rank': 0,
            })()
        config = get_config(args)
        model = build_model(config, is_pretrain=False)
        checkpoint = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        return model, config

    def _build_transform(self, config):
        t = []

        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                    interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        return transforms.Compose(t)
    
    def preprocess(self, image: np.ndarray, category: str) -> torch.Tensor:
        img = Image.fromarray(image)
        img = self.transforms[category](img)
        return img.unsqueeze(0)
        
        
    def analyze(self, video_path: str, detection_results: List[dict]) -> Dict:
        self.frame_cache = {}
        cap = cv2.VideoCapture(video_path)
        frame_results = []
        total_abnormal = 0
        total_people = 0
        
        # frame level results
        frame_detections = defaultdict(list)
        for d in detection_results:
            frame_detections[d['frame_index']].append(d)
        
        for frame_idx in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc="Processing frames", disable=True):
            ret, frame = cap.read()
            if not ret:
                break
            
            # for cropping
            self.frame_cache[frame_idx] = frame
            
            frame_result = self.process_frame(frame_idx, frame_detections.get(frame_idx, []))
            frame_results.append(frame_result)
            
            if frame_result['person_count'] > 0:
                total_abnormal += frame_result['abnormal_count']
                total_people += frame_result['person_count']
        
        cap.release()
        del self.frame_cache
        
        final_score = total_abnormal / total_people if total_people > 0 else 0.0
        return {
            'video_results': 1 - final_score,
            'frame_results': frame_results
        }

    def process_frame(self, frame_idx: int, detections: List[dict]) -> Dict:
        person_data = defaultdict(dict)
        for d in detections:
            person_id = d['person_index']
            category = d['label']
            # person_data[person_id][category] = d['bbox']
            if person_id not in person_data:
                person_data[person_id] = {}
            if category not in person_data[person_id]:
                person_data[person_id][category] = []
            person_data[person_id][category].append(d['bbox'])
        
        batches = defaultdict(lambda: {'images': [], 'person_ids': [], 'bbox': []})
        
        for person_id, categories in person_data.items():
            for category in ['human', 'face', 'hand']:
                if category in categories:
                    bboxes = categories[category]
                    for bbox in bboxes:
                        image = self.smart_cut(self.frame_cache[frame_idx], bbox)
                        if image is not None:
                            batches[category]['images'].append(image)
                            batches[category]['person_ids'].append(person_id)
                            batches[category]['bbox'].append(bbox)
        
        predictions = defaultdict(dict)
        for category in batches:

            results = []
            # infer per batchsize
            for i in range(0, len(batches[category]['images']), self.batch_size):
                results.extend(self.predict_batch(category, batches[category]['images'][i:i+self.batch_size]))
            
            for pid, pred, bbox in zip(batches[category]['person_ids'], results, batches[category]['bbox']):
                # predictions[pid][category] = pred
                # maybe more than one prediction for each person and category
                if pid not in predictions:
                    predictions[pid] = {}
                if category not in predictions[pid]:
                    predictions[pid][category] = []
                predictions[pid][category].append((pred, bbox))            
        
        # abnormal count
        abnormal_count = 0
        person_results = []
        for person_id in person_data:
            scores = predictions.get(person_id, {})
            # is_abnormal = any(np.argmax(scores.get(cat, [0.5, 0.5])) == 0 for cat in ['human', 'face', 'hand'])
            
            is_abnormal = False
            for cat, cat_scores in scores.items():
                for score, _ in cat_scores:
                    if cat in self.class_threshold:
                        if score[0] > self.class_threshold[cat]:
                            is_abnormal = True
                            break
                if is_abnormal:
                    break
            
            person_results.append({
                'person_id': person_id,
                'abnormal': is_abnormal,
                'scores': scores
            })
            abnormal_count += int(is_abnormal)
        
        return {
            'frame': frame_idx,
            'person_count': len(person_data),
            'abnormal_count': abnormal_count,
            'persons': person_results
        }

    def predict_batch(self, category: str, batch: List[np.ndarray]) -> List[List[float]]:
        preprocessed = [self.preprocess(img, category) for img in batch]
        with torch.no_grad():
            inputs = torch.cat(preprocessed, dim=0).to(self.device)
            outputs = self.models[category](inputs).cpu()
        return F.softmax(outputs, dim=1).numpy().tolist()


    
    def smart_cut(self, frame, bbox, resize=None):
        x1, y1, x2, y2 = map(float, bbox)
        H, W = frame.shape[:2]
        
        # bbox width, height and center calculation
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            raise ValueError("Invalid bbox dimensions")
        mid_x, mid_y = (x1 + x2)/2, (y1 + y2)/2

        # crop outer square
        max_len = max(w, h)
        # range of center movement
        x_min = max(x2 - max_len/2, max_len/2)
        x_max = min(x1 + max_len/2, W - max_len/2)
        y_min = max(y2 - max_len/2, max_len/2)
        y_max = min(y1 + max_len/2, H - max_len/2)
        
        if x_min <= x_max and y_min <= y_max:
            # valid point 
            adj_x = min(max(mid_x, x_min), x_max)
            adj_y = min(max(mid_y, y_min), y_max)
            # square points calculation
            x1_sq = adj_x - max_len/2
            y1_sq = adj_y - max_len/2
            x2_sq, y2_sq = x1_sq + max_len, y1_sq + max_len
            
            x1_int = math.floor(x1_sq)
            y1_int = math.floor(y1_sq)
            x2_int = math.ceil(x2_sq)
            y2_int = math.ceil(y2_sq)
            
            # range validation
            if x1_int >= 0 and y1_int >= 0 and x2_int <= W and y2_int <= H:
                cropped = frame[y1_int:y2_int, x1_int:x2_int]
                if cropped.size > 0:
                    return cv2.resize(cropped, (resize, resize)) if resize else cropped

        # crop inner inner
        min_len = min(w, h)
        # range of center movement
        x_min = max(min_len/2, 0.0)
        x_max = W - min_len/2
        y_min = max(min_len/2, 0.0)
        y_max = H - min_len/2
        
        adj_x = min(max(mid_x, x_min), x_max)
        adj_y = min(max(mid_y, y_min), y_max)
        
        # square points
        x1_sq = adj_x - min_len/2
        y1_sq = adj_y - min_len/2
        x2_sq, y2_sq = x1_sq + min_len, y1_sq + min_len
        
        x1_int = math.floor(x1_sq)
        y1_int = math.floor(y1_sq)
        x2_int = math.ceil(x2_sq)
        y2_int = math.ceil(y2_sq)
        
        if x1_int >= 0 and y1_int >= 0 and x2_int <= W and y2_int <= H:
            cropped = frame[y1_int:y2_int, x1_int:x2_int]
            if cropped.size > 0:
                return cv2.resize(cropped, (resize, resize)) if resize else cropped
        
        raise ValueError("Cannot crop valid region within frame")

    def _process_predictions(self, predictions, threshold):
        return [p[0] > threshold for p in predictions]

def compute_abnormality(video_paths, device, submodules_dict, **kwargs):
    # Initialize components
    detector = Detector(
        config_file=submodules_dict["detector_config"],
        weight_file=submodules_dict["detector_weights"],
        device=device
    )
    
    analyzer = Analyzer(
        model_configs=submodules_dict["analyzer_configs"],
        device=device,
        batch_size=submodules_dict["batch_size"],
        class_thresholds={k: v["threshold"] for k, v in submodules_dict["analyzer_configs"].items()}
    )

    all_results = []
    for video_path in tqdm(video_paths):

        detections = detector.detect_video(video_path)
        
        result = analyzer.analyze(
            video_path=video_path,
            detection_results=detections,
        )
        
        all_results.append({
            "video_path": video_path,
            'video_results': result['video_results'],
        })

    global_score = sum([x['video_results'] for x in all_results]) / len(all_results)
    
    return global_score, all_results

def parse_option():
    parser = argparse.ArgumentParser('training and evaluation script', add_help=False)
    # easy config modification
    parser.add_argument('--human_model', type=str, required=True, help='path to pre-trained model')
    parser.add_argument('--face_model', type=str, required=True, help='path to pre-trained model')
    parser.add_argument('--hand_model', type=str, required=True, help='path to pre-trained model')
    parser.add_argument('--detector_config', type=str, required=True, help='path to pre-trained model')
    parser.add_argument('--detector_weights', type=str, required=True, help='path to pre-trained model')
    parser.add_argument('--cfg', type=str, required=True, help='path to pre-trained model')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_option()
    submodules = {
        "detector_config": args.detector_config, #"yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py",
        "detector_weights": args.detector_weights, #"yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth",
        "analyzer_configs": {
            "human": {"cfg_path": args.cfg, "weight_path": args.human_model, "threshold": 0.4545454545454546},
            "face": {"cfg_path": args.cfg, "weight_path": args.face_model, "threshold": 0.30303030303030304},
            "hand": {"cfg_path": args.cfg, "weight_path": args.hand_model, "threshold": 0.3232}
        },
        "batch_size" : 128
    }
    
    video_paths = [
        "exmaple/people are walking.-1.mp4",
    ]
    
    final_score, detailed_results = compute_abnormality(
        video_paths=video_paths,
        device="cuda",
        submodules_dict=submodules
    )
    with open("test_results.json", "w") as f:
        json.dump(detailed_results, f)
    
    print(f"Global Abnormality Score: {final_score:.4f}")