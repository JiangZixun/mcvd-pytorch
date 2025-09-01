CUDA_VISIBLE_DEVICES=0 

/home/jzx/anaconda3/envs/vid/bin/python main.py \
    --config configs/smmnist_DDPM_big5.yml \
    --data_path data \
    --exp experiments/smmnist_cat2 \
    --ni