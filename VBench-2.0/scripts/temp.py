import os
import json
import pandas as pd
import numpy as np

dimension_to_category = {
    "Human Anatomy": "Human Fidelity",
    "Human Identity": "Human Fidelity",
    "Human Clothes": "Human Fidelity",
    "Diversity": "Creativity",
    "Composition": "Creativity",
    "Dynamic Spatial Relationship": "Controllability",
    "Dynamic Attribute": "Controllability",
    "Motion Order Understanding": "Controllability",
    "Human Interaction": "Controllability",
    "Complex Landscape": "Controllability",
    "Complex Plot": "Controllability",
    "Camera Motion": "Controllability",
    "Motion Rationality": "Commonsense",
    "Instance Preservation": "Commonsense",
    "Mechanics": "Physics",
    "Thermotics": "Physics",
    "Material": "Physics",
    "Multi-View Consistency": "Physics"
}
file = "/mnt/petrelfs/zhengdian/zhengdian/VBench-2.0_weuse/evaluation_results"
category_scores = {category: [] for category in set(dimension_to_category.values())}
# 用来存储每个模型在每个大类下的分数
model_scores_by_category = {}

# 遍历目录
for dimension in sorted(os.listdir(file)):
    dimension_path = os.path.join(file, dimension)
    for model in sorted(os.listdir(dimension_path)):
        model_path = os.path.join(dimension_path, model)
        for jsons in os.listdir(model_path):
            if jsons.endswith('_eval_results.json'):
                json_path = os.path.join(model_path, jsons)
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 处理dimension中的下划线，替换成空格
                dimension_name = dimension.replace('_', ' ')

                # 获取对应的大类
                if dimension_name in dimension_to_category:
                    category = dimension_to_category[dimension_name]
                else:
                    continue  # 如果没有找到匹配的维度，跳过该维度

                # 提取分数
                score = data[dimension][0]

                # 将该维度的分数添加到对应大类的分数列表
                category_scores[category].append(score)

                # 将该模型在该大类下的分数存储
                if model not in model_scores_by_category:
                    model_scores_by_category[model] = {category: [] for category in category_scores}

                model_scores_by_category[model][category].append(score)

# 计算每个模型的最终得分（即五个大类的平均分）
model_final_scores = {}
model_final_large = {}
cnt=0

for model, categories in model_scores_by_category.items():
    model_scores_in_categories = []
    dim=[]
    for category in category_scores:
        dim.append(category)
        # 计算该模型在每个大类的得分（该大类下所有维度的平均分）
        category_scores_for_model = model_scores_by_category[model].get(category, [])
        if category_scores_for_model:
            category_avg = sum(category_scores_for_model) / len(category_scores_for_model)
            model_scores_in_categories.append(category_avg)
        else:
            model_scores_in_categories.append(0)
    model_final_large[model]=model_scores_in_categories

df = pd.DataFrame({
    'Dimension': dim,
    'HunyuanVideo': model_final_large['HunyuanVideo'],
    'StepVideo': model_final_large['StepVideo'],
    'Wanx': model_final_large['Wanx'],
    'Kling': model_final_large['Kling'],
    'Sora': model_final_large['Sora'],
    'CogVideo': model_final_large['CogVideo']
})
pd.set_option('display.colheader_justify', 'center')  # 设置列标题居中
print(df.to_string(index=False))  # 打印时去除索引列

for item in model_final_large:
    scores = model_final_large[item]
    np_score = np.array(scores)
    print(f'{item}: {np_score.mean()}')