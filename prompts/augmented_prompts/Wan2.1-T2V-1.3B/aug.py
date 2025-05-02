# prompt_augmenter.py (优化版)
import json
from PIL import Image
from tqdm import tqdm
import torch
from wan.utils.prompt_extend import QwenPromptExpander
import os
import gc

class PromptAugmenter:
    def __init__(self, input_path, output_path, model_name="Qwen/Qwen-VL-Chat", device="cuda", seed=42):
        """
        Initialize the prompt augmenter with memory optimizations
        
        Args:
            max_image_size: 最大图像尺寸 (长或宽)，自动进行缩放
        """
        # 设置内存优化环境变量
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        self.input_path = input_path
        self.output_path = output_path
        self.device = device
        self.seed = seed
        
        # 延迟加载模型，只在需要时初始化
        self._expander = None
        self.model_name = model_name
        self.prompt_expander = QwenPromptExpander(
                model_name=args.model_name,
                is_vl=False,
                device=self.device)

    
    def load_data(self):
        """Load the original JSON data"""
        with open(self.input_path, 'r', encoding='utf-8') as f:
            return f.readlines()
    
    def save_data(self, data):
        """Save augmented data to JSON"""
        with open(self.output_path, 'w', encoding='utf-8') as f:
            #f.writelines(data)
            for line in data:
                f.write(line + '\n')
    
    
    def augment_prompt(self, prompt):
        """Augment a single prompt with memory management"""
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        result = self.prompt_expander(
            prompt,
            tar_lang="en",
            seed=self.seed
        )
        
        gc.collect()
        torch.cuda.empty_cache()
        
        return result.prompt if result.status else prompt
        
    
    def process_all(self,  batch_size=1):
        original_data = self.load_data()
        
        augmented_data = []
        for i in tqdm(range(0, len(original_data), batch_size), desc="Processing batches"):
            batch = original_data[i:i + batch_size]
            batch_results = []
            
            for item in batch:
                augmented_prompt = self.augment_prompt(
                    item
                )
                batch_results.append(augmented_prompt)
                
            
            augmented_data.extend(batch_results)
            
            if i % 10 == 0:
                self.save_data(augmented_data)
        
        self.save_data(augmented_data)
        return augmented_data
    


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized Prompt Augmentation Tool")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to save")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-VL-Chat",
                       help="Qwen model name or path")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run the model on (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Number of items to process at once (be careful with memory)")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # 初始化并运行
    print(f"Starting optimized prompt augmentation for {args.input}")
    augmenter = PromptAugmenter(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model_name,
        device=args.device,
        seed=args.seed
    )
    
    augmenter.process_all(batch_size=args.batch_size)
    print(f"Augmented prompts saved to {args.output}")