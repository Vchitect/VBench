# download pretrain checkpoint at  https://drive.google.com/file/d/1dJn6GYkwMIcoP3zqOEyW1_iQfpBi8UOw/view?usp=sharing


torchrun main_finetune.py --cfg 'configs/vit_base__800ep/simmim_finetune__vit_base__img224__800ep.yaml' --train-path "path-to-train-set-json" --val-path "path-to-val-set-json" --pretrained 'path-to-pretrained-vit-base' --batch-size 128 --output "output-path"