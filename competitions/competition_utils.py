import os
import torchvision.io as tvio
import torch

def transform_to_videos(input_path, output_path, frame_rate):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for root, dirs, files in os.walk(input_path):
        for directory in dirs:
            
            dir_path = os.path.join(root, directory)
            image_files = [f for f in os.listdir(dir_path) if f.endswith('.png')]
            if not image_files:
                continue  # Skip if there are no image files in the directory

            image_files.sort()
            
            frames = []
            for image_file in image_files:
                image_path = os.path.join(dir_path, image_file)
                frame = tvio.read_image(image_path)
                frames.append(frame)
            frames = torch.stack(frames).permute(0, 2, 3, 1)    
            
            # Write the frames to video
            video_path = os.path.join(output_path, f"{directory}.mp4")
            tvio.write_video(video_path, frames, fps=frame_rate)

    print(f"Videos are saved in '{output_path}'")


