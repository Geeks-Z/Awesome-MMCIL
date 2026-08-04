
conda activate mmcl

log_root="./logs/OpenCLIP_LAION400M_ViTB16"
mkdir -p \
  "${log_root}/FineTune/queue" \
  "${log_root}/SimpleCIL/queue" \
  "${log_root}/ZS-CLIP/queue" \
  "${log_root}/L2P/queue" \
  "${log_root}/DualPrompt/queue" \
  "${log_root}/CODA-Prompt/queue" \
  "${log_root}/RAPF/queue" \
  "${log_root}/PROOF/queue" \
  "${log_root}/ENGINE/queue" \
  "${log_root}/BOFA/queue" \
  "${log_root}/CLG-CBM/queue" \
  "${log_root}/AREA/queue"

# Finetune
nohup ./scripts/run_finetune.sh > "${log_root}/FineTune/queue/Finetune-LAION-400M.out" 2>&1 &

# SimpleCIL
nohup ./scripts/run_simplecil.sh > "${log_root}/SimpleCIL/queue/SimpleCIL-LAION-400M.out" 2>&1 &

# ZS-CLIP
nohup ./scripts/run_zs_clip.sh > "${log_root}/ZS-CLIP/queue/ZS-CLIP-LAION-400M-Top5_10.out" 2>&1 &

# L2P
nohup ./scripts/run_l2p.sh > "${log_root}/L2P/queue/L2P-LAION-400M-supp.out" 2>&1 &

# DualPrompt
nohup ./scripts/run_dualprompt.sh > "${log_root}/DualPrompt/queue/DualPrompt-LAION-400M.out" 2>&1 &

# CODA-Prompt
nohup ./scripts/run_coda_prompt.sh > "${log_root}/CODA-Prompt/queue/CODA-Prompt-LAION-400M.out" 2>&1 &

# RAPF
nohup ./scripts/run_rapf.sh > "${log_root}/RAPF/queue/RAPF-LAION-400M.out" 2>&1 &

# PROOF
nohup ./scripts/run_proof.sh > "${log_root}/PROOF/queue/PROOF-LAION-400M.out" 2>&1 &

# Engine
nohup ./scripts/run_engine.sh > "${log_root}/ENGINE/queue/ENGINE-LAION-400M-supp.out" 2>&1 &

# BOFA
nohup ./scripts/run_bofa.sh > "${log_root}/BOFA/queue/BOFA-LAION-400M.out" 2>&1 &

# CLG-CBM
nohup ./scripts/run_CLG-CBM.sh > "${log_root}/CLG-CBM/queue/CLG-CBM-LAION-400M.out" 2>&1 &

# AREA
nohup ./scripts/run_area.sh > "${log_root}/AREA/queue/AREA-LAION-400M.out" 2>&1 &
