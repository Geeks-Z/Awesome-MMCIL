#!/bin/bash
set -e

# Note: This order must be consistent with the order set in the 'dataset_sequence' parameter. 'dataset_sequence' is only used to construct the 'testloader'.
python main.py -d TAIL -m clip_vit_b16 num_shots -1 dataset aircraft num_epochs 60 continue_train_first True output_dir TAIL_fullshot
python main.py -d TAIL -m clip_vit_b16 num_shots -1 dataset caltech101 num_epochs 20 output_dir TAIL_fullshot
python main.py -d TAIL -m clip_vit_b16 num_shots -1 dataset dtd num_epochs 30 output_dir TAIL_fullshot
python main.py -d TAIL -m clip_vit_b16 num_shots -1 dataset eurosat num_epochs 20 output_dir TAIL_fullshot
python main.py -d TAIL -m clip_vit_b16 num_shots -1 dataset flowers num_epochs 40 output_dir TAIL_fullshot
python main.py -d TAIL -m clip_vit_b16 num_shots -1 dataset food101 num_epochs 10 output_dir TAIL_fullshot
python main.py -d TAIL -m clip_vit_b16 num_shots -1 dataset mnist num_epochs 20 output_dir TAIL_fullshot
python main.py -d TAIL -m clip_vit_b16 num_shots -1 dataset oxford_pets num_epochs 10 output_dir TAIL_fullshot
python main.py -d TAIL -m clip_vit_b16 num_shots -1 dataset stanford_cars num_epochs 20 output_dir TAIL_fullshot
python main.py -d TAIL -m clip_vit_b16 num_shots -1 dataset sun397 num_epochs 2 output_dir TAIL_fullshot

python result_process.py -d TAIL --output_dir TAIL_fullshot
