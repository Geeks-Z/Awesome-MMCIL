cd Code/Research/Awesome-MMCL/PROOF &&
conda activate peft

nohup ./train.sh > ../results/PROOF-B0-3090-supp.out 2>&1 &

nohup ./train_inc.sh > ../results/OpenCLIP_LAION400M/PROOF-CIFAR-INR-CUB-3090.out 2>&1 &
