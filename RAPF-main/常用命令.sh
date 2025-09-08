cd Code/Research/Awesome-MMCL/RAPF-main &&
conda activate peft

nohup ./train.sh > ../results/RAPF-OpenAI_CLIP-B0-A40-1.out 2>&1 &