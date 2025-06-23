cd Code/Research/Awesome-MMCL/MoE-Adapters4CL-MoE-Adapters/cil &&
conda activate peft

nohup ./train.sh > ./res/MoE-Adapters-cifar-B0-Inc5.out 2>&1 &