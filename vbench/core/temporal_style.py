import os
import json
import numpy as np
import torch
import clip
from tqdm import tqdm
from vbench.utils import load_video, load_dimension_info, clip_transform, read_frames_decord_by_fps, CACHE_DIR
from vbench.third_party.ViCLIP.viclip import ViCLIP
from vbench.third_party.ViCLIP.simple_tokenizer import SimpleTokenizer
from vbench.distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)
from vbench.core import DimensionEvaluationBase, EvaluationResult, MemoryEstimate, MEMORY_USAGE_PROFILE
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TemporalStyle(DimensionEvaluationBase):
    def __init__(self, device="cuda", batch_size=1):
        super().__init__(
            memory_profile=MEMORY_USAGE_PROFILE["temporal_style"],
            device=device,
            batch_size=batch_size
        )
        
    def init_model(self, cache_folder=None):
        viclip_path = os.path.join(cache_folder, "ViCLIP", "ViClip-InternVid-10M-FLT.pth")
        ensure_download(
            viclip_path,
            "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth"
        )
        
        tokenizer_path = os.path.join(cache_folder, "ViCLIP", "bpe_simple_vocab_16e6.txt.gz")
        ensure_download(
            tokenizer_path,
            "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/bpe_simple_vocab_16e6.txt.gz"
        )
        
        tokenizer = SimpleTokenizer(tokenizer_path)
        viclip = ViCLIP(tokenizer=tokenizer, pretrain=viclip_path).to(self.device)
        
        self.model["viclip"] = viclip
        self.model["tokenizer"] = tokenizer
        
    def get_text_features(self, model, input_text, tokenizer, text_feature_dict={}):
        if input_text in text_feature_dict:
            return text_feature_dict[input_text]
        text_template = f"{input_text}"
        with torch.no_grad():
            text_features = model.encode_text(text_template).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)      
            text_feature_dict[input_text] = text_features
        return text_features

    def get_vid_features(self, model, input_frames):
        with torch.no_grad():
            clip_feat = model.encode_vision(input_frames, test=True).float()
            clip_feat /= clip_feat.norm(dim=-1, keepdim=True)    
        return clip_feat

    def get_predict_label(self, clip_feature, text_feats_tensor, top=5):
        label_probs = (100.0 * clip_feature @ text_feats_tensor.T).softmax(dim=-1)
        top_probs, top_labels = label_probs.cpu().topk(top, dim=-1)
        return top_probs, top_labels

    def _temporal_style_evaluation(self, video_dict, sample="middle"):
        sim = []
        video_results = []
        clip_model = self.model["viclip"]
        tokenizer = self.model["tokenizer"]
        image_transform = clip_transform(224)
        text_feature_dict = {}
        
        for info in tqdm(video_dict, disable=get_rank() > 0):
            query = info['prompt']
            video_list = info['video_list']
            
            for video_path in video_list:
                with torch.no_grad():
                    images = read_frames_decord_by_fps(video_path, num_frames=8, sample=sample)
                    images = image_transform(images)
                    images = images.to(self.device)
                    
                    clip_feat = self.get_vid_features(clip_model, images.unsqueeze(0))
                    text_feat = self.get_text_features(clip_model, query, tokenizer, text_feature_dict)
                    
                    logit_per_text = clip_feat @ text_feat.T
                    score_per_video = float(logit_per_text[0][0].cpu())
                    
                    sim.append(score_per_video)
                    video_results.append({
                        'video_path': video_path, 
                        'video_results': score_per_video
                    })
        
        avg_score = np.mean(sim) if sim else 0.0
        return avg_score, video_results
        
    def compute_score(self, json_dir, submodules_list, **kwargs) -> EvaluationResult:
        _, video_dict = load_dimension_info(json_dir, dimension='temporal_style', lang='en')
        video_dict = distribute_list_to_rank(video_dict)
        
        sample = kwargs.get('sample', 'middle')
        if 'sample' in submodules_list:
            sample = submodules_list['sample']
        
        if any(key in submodules_list for key in ['model_path', 'config_path', 'pretrain_path']):
            if get_rank() == 0:
                tokenizer = SimpleTokenizer(os.path.join(CACHE_DIR, "ViCLIP/bpe_simple_vocab_16e6.txt.gz"))
                viclip_params = {k: v for k, v in submodules_list.items() 
                               if k in ['model_path', 'config_path', 'pretrain_path']}
                viclip = ViCLIP(tokenizer=tokenizer, **viclip_params).to(self.device)
                viclip.eval()
            barrier()
            self.model["viclip"] = viclip
            self.model["tokenizer"] = tokenizer
        
        all_results, video_results = self._temporal_style_evaluation(video_dict, sample=sample)
        
        if get_world_size() > 1:
            video_results = gather_list_of_dict(video_results)
            all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
            
        return EvaluationResult(
            dimension="temporal_style",
            overall_score=all_results,
            per_video_scores=video_results
        )
