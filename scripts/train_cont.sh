#!/bin/bash

python main.py \
--setting 1 \
--container-size 1.0 \
--max-item-size 5 \
--min-item-size 1 \
--num-box 100 \
--num-next-box $1 \
--num-candidate-action 120 \
--num-processes 64 \
--num-steps 30 \
--continuous \
--sample-from-distribution \
--sample-left-bound 0.1 \
--sample-right-bound 0.5 \
--unit-interval 0.1 \
--model-save-interval 200 \
--model-update-interval 200 \
--use-linear-lr-decay \
--max-model-num 200 \
--log-path ./logs \
--alpha $2
