from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import json
import os, subprocess

@dataclass
class MemoryEstimate:
    model_size_gb: int = 0
    activation_base_gb: int = 0 # normalize to 512x320x16
    temporal_scaling: bool = False
    resolution_scaling: bool = False
    max_resolution_scaling: int = 512*320
    max_temporal_scaling: int = 16

@dataclass
class EvaluationResult:
    dimension: str
    overall_score: float
    per_video_scores: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def write(cls, results: List["EvaluationResult"], output_path: str):
        output_dict = {}
        for result in results:
            output_dict[result.dimension] = [result.overall_score, result.per_video_scores]

        if results.metadata is not None:
            output_dict[result.dimension].append(result.metadata)

        json_string = json.dumps(output_dict, indent=4)
        with open(output_path, 'w') as f:
            f.write(json_string)
        return json_string


class DimensionEvaluationBase(ABC):
    def __init__(self, memory_profile: MemoryEstimate, device="cuda", batch_size=1):
        self.memory_profile = memory_profile
        self.batch_size = batch_size
        self.device = device
        self.model = {}

    def to(self, device):
        self.device = device
        for key in self.model.keys():
            self.model[key] = self.model[key].to(self.device)
            self.model[key]

    @abstractmethod
    def init_model(self, cache_folder) -> None:
        pass
    
    @abstractmethod
    def compute_score(self, json_dir, submodules_list, **kwargs) -> EvaluationResult:
        pass
    
    def estimate_memory_usage(self, resolution: tuple, timestep: int) -> float:
        assert len(resolution) == 2
        height, width = resolution
        base_height, base_width, base_frames = 320, 512, 16
        base_pixels = base_height * base_width

        if self.memory_profile.temporal_scaling and (self.memory_profile.max_temporal_scaling < timestep):
            temporal_factor = timestep / base_frames
        else:
            temporal_factor = 1.

        if self.memory_profile.resolution_scaling and (self.memory_profile.max_resolution_scaling < base_pixels):
            resolution_factor = resolution[0] * resolution[1] / base_pixels
        else:
            resolution_factor = 1.

        return self.memory_profile.model_size_gb + resolution_factor * temporal_factor * self.memory_profile.activation_base_gb

def get_dimension_evaluator(name: str):
    class_name = name.replace("_", " ").title()
    class_name = class_name.replace(" ", "")
    return class_name

MEMORY_USAGE_PROFILE = {
        "aesthetic_quality": MemoryEstimate( model_size_gb=0.9, activation_base_gb=1.1, temporal_scaling=False, resolution_scaling=False, max_resolution_scaling=224),
        "apperance_style": MemoryEstimate( model_size_gb=1.6, activation_base_gb=0.6, temporal_scaling=False, resolution_scaling=False, max_resolution_scaling=224),
        "background_consistency": MemoryEstimate( model_size_gb=1.6, activation_base_gb=0.6, temporal_scaling=False, resolution_scaling=False, max_resolution_scaling=224),
        "color": MemoryEstimate( model_size_gb=1.8, activation_base_gb=2.3, temporal_scaling=False, resolution_scaling=False, max_resolution_scaling=768),
        "dynamic_degree": MemoryEstimate( model_size_gb=1.1, activation_base_gb=1.0, temporal_scaling=False, resolution_scaling=False),
        "human_action": MemoryEstimate( model_size_gb=1.8, activation_base_gb=3.0, temporal_scaling=False, resolution_scaling=False, max_resolution_scaling=256),
        "imaging_quality": MemoryEstimate( model_size_gb=1.8, activation_base_gb=1.4, temporal_scaling=False, resolution_scaling=False, max_resolution_scaling=512),
        "motion_smoothness": MemoryEstimate( model_size_gb=1.0, activation_base_gb=1.2, temporal_scaling=False, resolution_scaling=False),
        "multiple_objects": MemoryEstimate( model_size_gb=2.7, activation_base_gb=1.8, temporal_scaling=False, resolution_scaling=False, max_resolution_scaling=768),
        "object_class": MemoryEstimate( model_size_gb=1.7, activation_base_gb=2.1, temporal_scaling=False, resolution_scaling=False, max_resolution_scaling=768),
        "overall_consistency": MemoryEstimate( model_size_gb=1.7, activation_base_gb=1.8, temporal_scaling=False, resolution_scaling=False, max_resolution_scaling=224),
        "scene": MemoryEstimate( model_size_gb=3.8, activation_base_gb=20.2, temporal_scaling=False, resolution_scaling=False, max_resolution_scaling=384),
        "spatial_relationship": MemoryEstimate( model_size_gb=2.5, activation_base_gb=2.2, temporal_scaling=False, resolution_scaling=False, max_resolution_scaling=768),
        "temporal_style": MemoryEstimate( model_size_gb=2.5, activation_base_gb=2.2, temporal_scaling=False, resolution_scaling=False, max_resolution_scaling=224),
}
