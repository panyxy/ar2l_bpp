#!/bin/bash

if [[ ${4} == "load_adv" ]];
then

python evaluation.py \
--setting 1 \
--container-size 10 \
--max-item-size 5 \
--min-item-size 1 \
--num-box 80 \
--num-next-box $1 \
--num-candidate-action 120 \
--evaluate \
--log-path ./logs \
--load-dataset \
--dataset-path ./datasets/eval_discrete_dataset.pt \
--load-bpp-model \
--load-adv-model \
--bpp-model-path $2 \
--adv-model-path $3 \


else

python evaluation.py \
--setting 1 \
--container-size 10 \
--max-item-size 5 \
--min-item-size 1 \
--num-box 80 \
--num-next-box $1 \
--num-candidate-action 120 \
--evaluate \
--log-path ./logs \
--load-dataset \
--dataset-path ./datasets/eval_discrete_dataset.pt \
--load-bpp-model \
--bpp-model-path $2 \
--adv-model-path $3 \

fi