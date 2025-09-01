CUDA_VISIBLE_DEVICES=0 

/home/jzx/anaconda3/envs/vid/bin/python main.py \
    --config configs/himawari_DDPM_big5.yml \
    --data_path data \
    --exp experiments/himawari_v1 \
    --ni