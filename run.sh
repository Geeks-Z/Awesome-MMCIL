
conda activate mmcl

# Finetune
nohup ./scripts/run_finetune.sh > ./results/Finetune-LAION-400M.out 2>&1 &

# SimpleCIL
nohup ./scripts/run_simplecil.sh > ./results/SimpleCIL-LAION-400M.out 2>&1 &

# ZS-CLIP
nohup ./scripts/run_zs_clip.sh > ./results/ZS-CLIP-LAION-400M-Top5_10.out 2>&1 &

# L2P
nohup ./scripts/run_l2p.sh > ./results/L2P-LAION-400M-supp.out 2>&1 &

# DualPrompt
nohup ./scripts/run_dualprompt.sh > ./results/DualPrompt-LAION-400M.out 2>&1 &

# CODA-Prompt
nohup ./scripts/run_coda_prompt.sh > ./results/CODA-Prompt-LAION-400M.out 2>&1 &

# RAPF
nohup ./scripts/run_rapf.sh > ./results/RAPF-LAION-400M.out 2>&1 &

# PROOF
nohup ./scripts/run_proof.sh > ./results/PROOF-LAION-400M.out 2>&1 &

# Engine
nohup ./scripts/run_engine.sh > ./results/ENGINE-LAION-400M-supp.out 2>&1 &

# BOFA
nohup ./scripts/run_bofa.sh > ./results/BOFA-LAION-400M.out 2>&1 &

# CLG-CBM
nohup ./scripts/run_CLG-CBM.sh > ./results/CLG-CBM-LAION-400M.out 2>&1 &

# AREA
nohup ./scripts/run_area.sh > ./results/AREA-LAION-400M.out 2>&1 &
