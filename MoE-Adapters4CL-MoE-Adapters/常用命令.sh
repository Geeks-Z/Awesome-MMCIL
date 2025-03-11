cd Code/Research/Awesome-MMCL/MoE-Adapters4CL-MoE-Adapters &&
conda activate peft

nohup ./train.sh > ./res/MoE-Adapters-B0-cifar-inr-temp.out 2>&1 &