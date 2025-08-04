#!/bin/bash




python evaluation.py \
--setting 1 \
--container-size 1.0 \
--max-item-size 5 \
--min-item-size 1 \
--num-box 100 \
--num-next-box ${1} \
--num-candidate-action 120 \
--continuous \
--sample-from-distribution \
--sample-left-bound 0.1 \
--sample-right-bound 0.5 \
--unit-interval 0.1 \
--num-processes 64 \
--log-path ./logs \
--evaluate \
--num-eval-episodes 3000 \
--load-dataset \
--dataset-path ./dataset/eval_continuous_dataset.pt \
--load-bpp-model \
--bpp-model-path ./pretrained_model/pretrained_${2}_model/nnb${1}/continuous/bpp.pt \
#--load-adv-model \
#--adv-model-path ./pretrained_model/pretrained_${2}_model/nnb${1}/continuous/adv.pt \



