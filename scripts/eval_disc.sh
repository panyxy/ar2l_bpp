#!/bin/bash



python evaluation.py \
--setting 1 \
--container-size 10 \
--max-item-size 5 \
--min-item-size 1 \
--num-box 80 \
--num-next-box ${1} \
--num-candidate-action 120 \
--num-processes 64 \
--log-path ./logs \
--evaluate \
--num-eval-episodes 3000 \
--load-dataset \
--dataset-path ./dataset/eval_discrete_dataset.pt \
--load-bpp-model \
--bpp-model-path ./pretrained_model/pretrained_${2}_model/nnb${1}/discrete/bpp.pt \
#--load-adv-model \
#--adv-model-path ./pretrained_model/pretrained_${2}_model/nnb${1}/discrete/adv.pt \




