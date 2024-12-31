import io
import os
import json
import zipfile
import argparse

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from constant import *

def submission(model_name, zip_file):
    os.makedirs(model_name, exist_ok=True)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(model_name)
    upload_data = {}
    # load your score
    for file in os.listdir(model_name):
        if file.startswith('.') or file.startswith('__'):
            print(f"Skip the file: {file}")
            continue
        cur_file = os.path.join(model_name, file)
        if os.path.isdir(cur_file):
            for subfile in os.listdir(cur_file):
                if subfile.endswith(".json"):
                    with open(os.path.join(cur_file, subfile)) as ff:
                        cur_json = json.load(ff)
                        if isinstance(cur_json, dict):
                            for key in cur_json:
                                upload_data[I2VKEY[key]] = cur_json[key][0]
        elif cur_file.endswith('json'):
            with open(cur_file) as ff:
                cur_json = json.load(ff)
                if isinstance(cur_json, dict):
                    for key in cur_json:
                        upload_data[I2VKEY[key]] = cur_json[key][0]
        
        for key in TASK_INFO_I2V:
            if key not in upload_data:
                upload_data[key] = 0
    return upload_data

def get_nomalized_score(upload_data):
    # get the normalize score
    normalized_score = {}
    for key in TASK_INFO_I2V:
        min_val = NORMALIZE_DIC_I2V[key]['Min']
        max_val = NORMALIZE_DIC_I2V[key]['Max']
        normalized_score[key] = (upload_data[key] - min_val) / (max_val - min_val)
        normalized_score[key] = normalized_score[key] * DIM_WEIGHT_I2V[key]
    return normalized_score

def get_i2v_quality_score(normalized_score):
    quality_score = []
    for key in I2V_QUALITY_LIST:
        quality_score.append(normalized_score[key])
    quality_score = sum(quality_score)/sum([DIM_WEIGHT_I2V[i] for i in I2V_QUALITY_LIST])
    return quality_score

def get_i2v_score(normalized_score):
    i2v_score = []
    for key in I2V_LIST:
        i2v_score.append(normalized_score[key])
    i2v_score  = sum(i2v_score)/sum([DIM_WEIGHT_I2V[i] for i in I2V_LIST ])
    return i2v_score

def get_final_score(quality_score,i2v_score):
    return (quality_score * I2V_QUALITY_WEIGHT + i2v_score * I2V_WEIGHT) / (I2V_QUALITY_WEIGHT + I2V_WEIGHT)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Load submission file')
    parser.add_argument('--zip_file', type=str, required=True, help='Name of the zip file', default='evaluation_results.zip')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model', default='t2v_model')
    args = parser.parse_args()

    upload_dict = submission(args.model_name, args.zip_file)
    print(f"your submission info: \n{upload_dict} \n")
    normalized_score = get_nomalized_score(upload_dict)
    quality_score = get_i2v_quality_score(normalized_score)
    i2v_score = get_i2v_score(normalized_score)
    final_score = get_final_score(quality_score, i2v_score)
    print('+------------------|------------------+')
    print(f'|     Quality Score|{quality_score}|')
    print(f'|         I2V Score|{i2v_score}|')
    print(f'|       Total Score|{final_score}|')
    print('+------------------|------------------+')
