import os
from PIL import Image
import json
import os.path as osp
import random
import argparse
from tqdm import tqdm

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def save_json(data, save_file):
    json.dump(data, open(save_file, "w"))


def crop(img_path, bbox, save_root):
    os.makedirs(save_root, exist_ok=True)
    img = Image.open(img_path)
    x, y, width, height = map(int, bbox)
    crop_img = img.crop((x, y, x+width, y+height))
    crop_img.save(osp.join(save_root, osp.basename(img_path)))
    
    
def get_other_ratio_crop(second_crop_info, ratio="8-5"):
    random.seed(123)
    ratio_w, ratio_h = map(int, ratio.split('-'))
    assert 1.0 <= ratio_w/ratio_h < 1.7778, "The ratio does not meet the requirements, it needs to be between 1:1 and 16:9."
    width, height = second_crop_info['width'], second_crop_info['height']
    x, y, crop_w, crop_h = second_crop_info['second_bbox']
    
    if width == height:
        target_w = int(width/ratio_w) * ratio_w
        target_h = int(width/ratio_w) * ratio_h
        assert target_h >= crop_h
        target_x = 0
        y_min = max(y - (target_h - crop_h), 0)
        y_max = min(y + target_h, height) - target_h
        assert y_max >= y_min
        target_y = random.randint(y_min, y_max)
    else:
        target_w = int(height/ratio_h) * ratio_w
        target_h = int(height/ratio_h) * ratio_h
        assert target_w >= crop_w
        target_y = 0
        x_min = max(x - (target_w - crop_w), 0)
        x_max = min(x + target_w, width) - target_w
        assert x_max >= x_min
        target_x = random.randint(x_min, x_max)
        
    return [target_x, target_y, target_w, target_h]


def transfer_bbox_to_origin_img(first_crop_info, old_bbox):
    x, y, _, _ = first_crop_info["first_bbox"]
    old_x, old_y, width, height = old_bbox
    return [x + old_x, y + old_y, width, height]



def get_target_crop(args):

    data = json.load(open(args.crop_info_path, "r"))
    target_results = []
    os.makedirs(args.result_path, exist_ok=True)
    
    ####### get target crop info ########
    for item in tqdm(data):
        second_crop_info = item["second_crop"]
        first_crop_info = item["first_crop"]
        target_crop = transfer_bbox_to_origin_img(first_crop_info, get_other_ratio_crop(second_crop_info, args.target_ratio))
        item["target_crop"] = {
            "target_ratio":args.target_ratio,
            "target_bbox":target_crop
        }
        target_results.append(item)

    target_file = os.path.join(args.result_path, f"target_crop_info_{args.target_ratio}.json")
    save_json(target_results, target_file)
    logger.info(f"Target crop info are saved in the '{target_file}' file")    
    
    ####### crop images #########
    ori_path = args.ori_image_path
    target_path = f"{args.result_path}/{args.target_ratio}"

    for sample in tqdm(target_results):
        img_path = osp.join(ori_path, sample["file_name"])
        target_bbox = sample["target_crop"]["target_bbox"]
        crop(img_path, target_bbox, target_path)
    
    logger.info(f"Cropped images are saved in the '{target_path}' path")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_info_path', type=str, default="vbench2_beta_i2v/data/i2v-bench-info.json", help="image suite meta info")
    parser.add_argument('--target_ratio', default="5-4", required=True, help="the required crop ratio")
    parser.add_argument('--ori_image_path', type=str, default="vbench2_beta_i2v/data/origin", help='the file path of the original image data')
    parser.add_argument('--result_path', type=str, default="vbench2_beta_i2v/data/target_crop", help='result save path')
    args = parser.parse_args()
    get_target_crop(args)