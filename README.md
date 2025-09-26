# Fluxy-Fine-Extractor
Mod of flux_lora_extract python script from Kohya_FLUX_v30's repo.  Compares two flux-krea ckpts and extracts lora. Multiple options for target key filtering. Flexible processing with heavy reliance on GPU. 

Designed for cmd line use within kohya_flux_v30. Runs inside or outside of venv. May also work on flux checkpoints. 

Install amd Use: 

1. Drop script in Kohya_FLUX_DreamBooth_LoRA_v30\kohya_ss\sd-scripts\networks
2. Open cmd and use -h flag for help with command usage or follow example below 

Example Command: 

python fluxkrea_lora_extractor_targeted_gpu_cpu_v0.9.py \
    --model_org base.safetensors \
    --model_tuned tuned.safetensors \
    --save_to output.safetensors \
    --dim 128 \
    --clamp_quantile 0.95 \
    --save_precision fp16 \
    --skip_double_blocks \
    --skip_kqv \
    --skip_proj \
    --skip_mlp0 \
    --skip_mlp2 \
    --skip_txt \
    --skip_img \
    --skip_single_blocks \
    --skip_linear2 \
    --skip_linear1 \
    --skip_modulation
