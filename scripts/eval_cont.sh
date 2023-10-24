#!/bin/bash

if [[ ${4} == "load_adv" ]];
then

python evaluation.py \
--setting 1 \
--container-size 1.0 \
--max-item-size 5 \
--min-item-size 1 \
--num-box 100 \
--num-next-box $1 \
--num-candidate-action 120 \
--continuous \
--sample-from-distribution \
--sample-left-bound 0.1 \
--sample-right-bound 0.5 \
--unit-interval 0.1 \
--evaluate \
--log-path ./logs \
--load-dataset \
--dataset-path ./datasets/eval_continuous_dataset.pt \
--load-bpp-model \
--bpp-model-path $2 \
--load-adv-model \
--adv-model-path $3 \

else

python evaluation.py \
--setting 1 \
--container-size 1.0 \
--max-item-size 5 \
--min-item-size 1 \
--num-box 100 \
--num-next-box $1 \
--num-candidate-action 120 \
--continuous \
--sample-from-distribution \
--sample-left-bound 0.1 \
--sample-right-bound 0.5 \
--unit-interval 0.1 \
--evaluate \
--log-path ./logs \
--load-dataset \
--dataset-path ./datasets/eval_continuous_dataset.pt \
--load-bpp-model \
--bpp-model-path $2 \
--adv-model-path $3 \


fi