CUDA_VISIBLE_DEVICES=2 python run.py \
    --checkpoint output/Marigold_w_text/train_marigold/checkpoint/iter_020000 \
    --denoise_steps 50 \
    --ensemble_size 10 \
    --input_rgb_dir input/in-the-wild_example\
    --output_dir output/in-the-wild_example
