cd Code/Research/Awesome-MMCL/RAPF-main &&
conda activate peft

nohup ./train.sh > ../results/RAPF-OpenAI_CLIP-CIFAR-INR-CUB-SEED_1993-A40.out 2>&1 &
