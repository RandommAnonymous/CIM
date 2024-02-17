#!/bin/bash/
python -m torch.distributed.run --nproc_per_node=1 eval_person_retrieval.py \
--output_dir 'output/Retrieval_msrvtt' --task 're_icfg' \
--checkpoint './output/icfg/checkpoint_best.pth' 