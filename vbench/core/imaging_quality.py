import torch
from tqdm import tqdm
from torchvision import transforms
from pyiqa.archs.musiq_arch import MUSIQ
from vbench.utils import load_video, load_dimension_info, CACHE_DIR, ensure_download

from vbench.distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)
from vbench.core import DimensionEvaluationBase, EvaluationResult, MEMORY_USAGE_PROFILE, MemoryEstimate

class ImagingQuality(DimensionEvaluationBase):
    def __init__(self, device="cuda", batch_size=1):
        super().__init__(
            memory_profile = MEMORY_USAGE_PROFILE["imaging_quality"],
            device=device,
            batch_size=batch_size
        )

    def init_model(self, cache_folder=CACHE_DIR):
        model_path = os.path.join(cache_folder, "pyiqa_model", "musiq_spaq_ckpt-358bb6af.pth")
        ensure_download(model_path, "https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth")
        model = MUSIQ(pretrained_model_path=model_path)
        model.eval()
        self.model["musiq"] = model

    def transform(self, images, preprocess_mode='shorter'):
        if preprocess_mode.startswith('shorter'):
            _, _, h, w = images.size()
            if min(h,w) > 512:
                scale = 512./min(h,w)
                images = transforms.Resize(size=( int(scale * h), int(scale * w) ), antialias=False)(images)
                if preprocess_mode == 'shorter_centercrop':
                    images = transforms.CenterCrop(512)(images)

        elif preprocess_mode == 'longer':
            _, _, h, w = images.size()
            if max(h,w) > 512:
                scale = 512./max(h,w)
                images = transforms.Resize(size=( int(scale * h), int(scale * w) ), antialias=False)(images)

        elif preprocess_mode == 'None':
            return images / 255.

        else:
            raise ValueError("Please recheck imaging_quality_mode")
        return images / 255.

    def technical_quality(self, model, video_list, batch_size, **kwargs):
        if 'imaging_quality_preprocessing_mode' not in kwargs:
            preprocess_mode = 'longer'
        else:
            preprocess_mode = kwargs['imaging_quality_preprocessing_mode']

        model.eval()
        video_results = []

        for video_path in tqdm(video_list, disable=get_rank() > 0):
            images = load_video(video_path)
            images = self.transform(images, preprocess_mode)

            scores_list = []
            for i in range(0, len(images), batch_size):
                frame_batch = images[i:i + batch_size]
                frame_batch = frame_batch.to(self.device)

                with torch.no_grad():
                    batch_scores = model(frame_batch)

                scores_list.append(batch_scores)

            all_scores = torch.cat(scores_list, dim=0)
            acc_score_video = torch.mean(all_scores).item()

            video_results.append({'video_path': video_path, 'video_results': acc_score_video})

        average_score = sum([o['video_results'] for o in video_results]) / len(video_results)
        average_score = average_score / 100.

        return average_score, video_results


    def compute_imaging_quality(self, json_dir, submodules_list, **kwargs):
        # model_path = submodules_list['model_path']

        video_list, _ = load_dimension_info(json_dir, dimension='imaging_quality', lang='en')
        video_list = distribute_list_to_rank(video_list)
        all_results, video_results = self.technical_quality(self.model["musiq"], video_list, self.batch_size, **kwargs)
        if get_world_size() > 1:
            video_results = gather_list_of_dict(video_results)
            all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
            all_results = all_results / 100.
        return EvaluationResult(
            dimension="imaging_quality",
            overall_score=all_results,
            per_video_scores=video_results
        )
