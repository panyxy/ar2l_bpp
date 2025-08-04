#!/bin/bash


python main.py \
--setting 1 \
--container-size 10 \
--max-item-size 5 \
--min-item-size 1 \
--num-box 80 \
--num-next-box ${1} \
--num-candidate-action 120 \
--num-processes 64 \
--num-steps 30 \
--model-save-interval 200 \
--use-linear-lr-decay \
--log-path ./logs \
--validate \
--validate-interval 200 \
--num-val-episodes 100 \
--dataset-path ./dataset/val_discrete_dataset.pt \
--alpha ${2} \
--use-wandb \


