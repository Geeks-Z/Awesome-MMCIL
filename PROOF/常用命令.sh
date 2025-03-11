cd Code/Research/Awesome-MMCL/PROOF &&
conda activate peft

nohup ./train.sh > ./res/3rd-stdout.res 2> ./res/3rd-stderr.res &

nohup ./train.sh > ./res/2nd-B0-stdout.res 2> ./res/2nd-B0-stderr.res &

nohup ./train_B50.sh > ./res/3rd-B50-stdout.res 2> ./res/3rd-B50-stderr.res &

# 结果放同一文件
nohup ./train.sh > ./res/B0-alldataset-gauss-mlp.out 2>&1 &

nohup ./train.sh > ./res/B0-vote-softmax-logits-margin.out 2>&1 &

nohup ./train.sh > ./res/B0-alldataset-mlp-1layer-0001.out 2>&1 &

# temp

nohup ./train.sh > ./res/3rd-proof-B0-alldataset.out 2>&1 &