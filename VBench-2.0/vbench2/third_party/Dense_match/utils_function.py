#############################################################################
# code for paper: Sora Generateds Video with Stunning Geometical Consistency
# arxiv: https://arxiv.org/abs/
# Author: <NAME> xuanyili
# email: xuanyili.edu@gmail.com
# github: https://github.com/meteorshowers/SoraGeoEvaluate
#############################################################################
import cv2
import csv
import os
import numpy as np
def ReadMp4FilesInFolder(folder_path):
    mp4_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
    video_list = []
    for mp4_file in mp4_files:
        video_path = os.path.join(folder_path, mp4_file)
        video_list.append(video_path)
    return video_list

def ExtractFrames(video_path, output_directory, frame_interval=5):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print("total_frames : ", total_frames)
    # print("fps : ", fps)
    # print("time(s): ", total_frames / fps)

    # Calculate frame interval based on video FPS
    # frame_interval = int(fps * frame_interval)

    # Loop through frames
    for frame_number in range(0, total_frames, frame_interval):
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        # Read the frame
        ret, frame = cap.read()
        # Check if the frame is valid
        if ret:
            # Save the processed frame to the output directory
            output_path = os.path.join(output_directory, f"frame_{frame_number}.png")
            cv2.imwrite(output_path, frame)
    # Release the video capture object
    cap.release()
    return total_frames, fps, total_frames / fps

def GetFileName(file_path):
    base_name = os.path.basename(file_path)  # 获取文件名
    file_name_without_extension = os.path.splitext(base_name)[0]  # 获取无后缀的文件名
    return file_name_without_extension

    folders = [f for f in os.listdir(folder_path)]
    print(folders)
    print(folder_path)
    folder_list = []
    for folder in folders:
        path = os.path.join(folder_path, folder)
        folder_list.append(path)
    folder_list.sort()
    return folder_list


def GetFileName(file_path):
    base_name = os.path.basename(file_path)  # 获取文件名
    file_name_without_extension = os.path.splitext(base_name)[0]  # 获取无后缀的文件名
    return file_name_without_extension


def ReadImageFolder(folder_path):
    folders = [f for f in os.listdir(folder_path)]
    # print(folders)
    # print(folder_path)
    folder_list = []
    for folder in folders:
        path = os.path.join(folder_path, folder)
        folder_list.append(path)
    folder_list.sort()
    return folder_list


def GetFileName(file_path):
    base_name = os.path.basename(file_path)  # 获取文件名
    file_name_without_extension = os.path.splitext(base_name)[0]  # 获取无后缀的文件名
    return file_name_without_extension



def read_csv_files(directory):
    data_frames = []
    filename_list =[]
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filename_list.append(filename)
    filename_list.sort()
    print(filename_list)
    # Loop through each file in the directory
    for filename in filename_list:
        file_path = os.path.join(directory, filename)
        
        # Read each CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Add the DataFrame to the list
        data_frames.append(df)
    
    return data_frames

def write_horizontal_csv(data_frames, output_file):
    # Concatenate the DataFrames horizontally
    result_df = pd.concat(data_frames, axis=1)
    
    # Write the resulting DataFrame to a new CSV file
    result_df.to_csv(output_file, index=False)