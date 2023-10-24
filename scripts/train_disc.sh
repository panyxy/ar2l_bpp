#!/bin/bash


python main.py \
--setting 1 \
--container-size 10 \
--max-item-size 5 \
--min-item-size 1 \
--num-box 80 \
--num-next-box $1 \
--num-candidate-action 120 \
--num-processes 64 \
--num-steps 30 \
--model-save-interval 200 \
--model-update-interval 200 \
--use-linear-lr-decay \
--max-model-num 200 \
--log-path ./logs \
--alpha $2
