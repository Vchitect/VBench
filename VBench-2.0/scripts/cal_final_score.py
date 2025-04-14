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
                                upload_data[key.replace('_',' ')] = cur_json[key][0]
        elif cur_file.endswith('json'):
            with open(cur_file) as ff:
                cur_json = json.load(ff)
                if isinstance(cur_json, dict):
                    for key in cur_json:
                        upload_data[key.replace('_',' ')] = cur_json[key][0]
        
        for key in TASK_INFO:
            if key not in upload_data:
                upload_data[key] = 0
    return upload_data

def get_creativity_score(score):
    creativity_score = []
    for key in CREATIVITY_LIST:
        creativity_score.append(score[key])
    creativity_score = sum(creativity_score)/len(CREATIVITY_LIST)
    return creativity_score

def get_commonsense_score(score):
    commonsense_score = []
    for key in COMMONSENSE_LIST:
        commonsense_score.append(score[key])
    commonsense_score = sum(commonsense_score)/len(COMMONSENSE_LIST)
    return commonsense_score

def get_controllability_score(score):
    controllability_score = []
    for key in CONTROLLABILITY_LIST:
        controllability_score.append(score[key])
    controllability_score = sum(controllability_score)/len(CONTROLLABILITY_LIST)
    return controllability_score

def get_human_fidelity_score(score):
    human_fidelity_score = []
    for key in HUMAN_FIDELITY_LIST:
        human_fidelity_score.append(score[key])
    human_fidelity_score = sum(human_fidelity_score)/len(HUMAN_FIDELITY_LIST)
    return human_fidelity_score

def get_physics_score(score):
    physics_score = []
    for key in PHYSICS_LIST:
        physics_score.append(score[key])
    physics_score = sum(physics_score)/len(PHYSICS_LIST)
    return physics_score

def get_final_score(creativity_score, commonsense_score, controllability_score, human_fidelity_score, physics_score):
    return (creativity_score + commonsense_score + controllability_score + human_fidelity_score + physics_score) / 5

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Load submission file')
    parser.add_argument('--zip_file', type=str, required=True, help='Name of the zip file', default='evaluation_results.zip')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model', default='t2v_model')
    args = parser.parse_args()

    upload_dict = submission(args.model_name, args.zip_file)
    print(f"your submission info: \n{upload_dict} \n")
    creativity_score = get_creativity_score(upload_dict)
    commonsense_score = get_commonsense_score(upload_dict)
    controllability_score = get_controllability_score(upload_dict)
    human_fidelity_score = get_human_fidelity_score(upload_dict)
    physics_score = get_physics_score(upload_dict)
    final_score = get_final_score(creativity_score, commonsense_score, controllability_score, human_fidelity_score, physics_score)
    print('+---------------------|------------------+')
    print(f'|     creativity score|{creativity_score}|')
    print(f'|    commonsense score|{commonsense_score}|')
    print(f'|controllability score|{controllability_score}|')
    print(f'| human fidelity score|{human_fidelity_score}|')
    print(f'|        physics score|{physics_score}|')
    print(f'|          total score|{final_score}|')
    print('+---------------------|------------------+')
