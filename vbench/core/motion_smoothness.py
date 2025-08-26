import os
import cv2
import glob
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from vbench.utils import load_dimension_info, CACHE_DIR, ensure_download
from vbench.third_party.amt.utils.utils import (
    img2tensor, tensor2img,
    check_dim_and_resize
)
from vbench.third_party.amt.utils.build_utils import build_from_cfg
from vbench.third_party.amt.utils.utils import InputPadder
from vbench.distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)
from vbench.core import DimensionEvaluationBase, EvaluationResult, MemoryEstimate, MEMORY_USAGE_PROFILE


class FrameProcess:
    def __init__(self):
        pass
    
    def get_frames(self, video_path):
        frame_list = []
        video = cv2.VideoCapture(video_path)
        while video.isOpened():
            success, frame = video.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to rgb
                frame_list.append(frame)
            else:
                break
        video.release()
        assert frame_list != []
        return frame_list 
    
    def get_frames_from_img_folder(self, img_folder):
        exts = ['jpg', 'png', 'jpeg', 'bmp', 'tif', 
                'tiff', 'JPG', 'PNG', 'JPEG', 'BMP', 
                'TIF', 'TIFF']
        frame_list = []
        imgs = sorted([p for p in glob.glob(os.path.join(img_folder, "*")) 
                      if os.path.splitext(p)[1][1:] in exts])
        for img in imgs:
            frame = cv2.imread(img, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_list.append(frame)
        assert frame_list != []
        return frame_list
    
    def extract_frame(self, frame_list, start_from=0):
        extract = []
        for i in range(start_from, len(frame_list), 2):
            extract.append(frame_list[i])
        return extract


class MotionSmoothness(DimensionEvaluationBase):
    def __init__(self, device="cuda", batch_size=1):
        super().__init__(
            memory_profile=MEMORY_USAGE_PROFILE["motion_smoothness"],
            device=device,
            batch_size=batch_size
        )
        self.niters = 1
        self.fp = FrameProcess()
        
    def init_model(self, cache_folder=CACHE_DIR):
        CUR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(CUR_DIR, "third_party", "amt", "cfgs", "AMT-S.yaml")
        ckpt_path = os.path.join(cache_folder, "amt_model", "amt-s.pth")
        ensure_download(ckpt_path, "https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth")
        
        network_cfg = OmegaConf.load(config_path).network
        network_name = network_cfg.name
        print(f'Loading [{network_name}] from [{ckpt_path}]...')
        
        model = build_from_cfg(network_cfg)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        
        self.model["amt"] = model
        self._initialize_memory_settings()
        
    def _initialize_memory_settings(self):
        if self.device == 'cuda':
            self.anchor_resolution = 1024 * 512
            self.anchor_memory = 1500 * 1024**2
            self.anchor_memory_bias = 2500 * 1024**2
            self.vram_avail = torch.cuda.get_device_properties(self.device).total_memory
            print("VRAM available: {:.1f} MB".format(self.vram_avail / 1024 ** 2))
        else:
            # Do not resize in cpu mode
            self.anchor_resolution = 8192*8192
            self.anchor_memory = 1
            self.anchor_memory_bias = 0
            self.vram_avail = 1
            
        self.embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(self.device)
        
    def _motion_score(self, video_path):
        iters = int(self.niters)
        
        if video_path.endswith('.mp4'):
            frames = self.fp.get_frames(video_path)
        elif os.path.isdir(video_path):
            frames = self.fp.get_frames_from_img_folder(video_path)
        else:
            raise NotImplementedError(f"Unsupported video format: {video_path}")
            
        frame_list = self.fp.extract_frame(frames, start_from=0)
        inputs = [img2tensor(frame).to(self.device) for frame in frame_list]
        
        assert len(inputs) > 1, f"The number of input should be more than one (current {len(inputs)})"
        
        inputs = check_dim_and_resize(inputs)
        h, w = inputs[0].shape[-2:]
        
        scale = self.anchor_resolution / (h * w) * np.sqrt((self.vram_avail - self.anchor_memory_bias) / self.anchor_memory)
        scale = 1 if scale > 1 else scale
        scale = 1 / np.floor(1 / np.sqrt(scale) * 16) * 16
        
        if scale < 1:
            print(f"Due to the limited VRAM, the video will be scaled by {scale:.2f}")
            
        padding = int(16 / scale)
        padder = InputPadder(inputs[0].shape, padding)
        inputs = padder.pad(*inputs)
        
        model = self.model["amt"]
        for i in range(iters):
            outputs = [inputs[0]]
            for in_0, in_1 in zip(inputs[:-1], inputs[1:]):
                in_0 = in_0.to(self.device)
                in_1 = in_1.to(self.device)
                with torch.no_grad():
                    imgt_pred = model(in_0, in_1, self.embt, scale_factor=scale, eval=True)['imgt_pred']
                outputs += [imgt_pred.cpu(), in_1.cpu()]
            inputs = outputs
            
        outputs = padder.unpad(*outputs)
        outputs = [tensor2img(out) for out in outputs]
        vfi_score = self._vfi_score(frames, outputs)
        norm = (255.0 - vfi_score) / 255.0
        
        return norm
        
    def _vfi_score(self, ori_frames, interpolate_frames):
        ori = self.fp.extract_frame(ori_frames, start_from=1)
        interpolate = self.fp.extract_frame(interpolate_frames, start_from=1)
        scores = []
        for i in range(len(interpolate)):
            scores.append(self._get_diff(ori[i], interpolate[i]))
        return np.mean(np.array(scores))
        
    def _get_diff(self, img1, img2):
        img = cv2.absdiff(img1, img2)
        return np.mean(img)
        
    def _motion_smoothness_evaluation(self, video_dict):
        sim = []
        video_results = []
        
        for info in tqdm(video_dict, disable=get_rank() > 0):
            video_list = info['video_list']
            for video_path in video_list:
                score_per_video = self._motion_score(video_path)
                video_results.append({
                    'video_path': video_path, 
                    'video_results': score_per_video
                })
                sim.append(score_per_video)
                
        avg_score = np.mean(sim) if sim else 0.0
        return avg_score, video_results
        
    def compute_score(self, json_dir, submodules_list, **kwargs) -> EvaluationResult:
        _, video_dict = load_dimension_info(json_dir, dimension='motion_smoothness', lang='en')
        video_dict = distribute_list_to_rank(video_dict)
        
        all_results, video_results = self._motion_smoothness_evaluation(video_dict)
        
        if get_world_size() > 1:
            video_results = gather_list_of_dict(video_results)
            all_results = np.mean([d['video_results'] for d in video_results])
            
        return EvaluationResult(
            dimension="motion_smoothness",
            overall_score=all_results,
            per_video_scores=video_results
        )
