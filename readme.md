# Adjustable Robust Reinforcement Learning for Online 3D Bin Packing

## Introduction
This is the official PyTorch implementation for the paper titled ["Adjustable Robust Reinforcement Learning for Online 3D Bin Packing"](https://arxiv.org/pdf/2310.04323.pdf).
The paper introduces the AR2L framework, which takes into account both the average performance and worst-case performance of a packing policy. 
By using this framework, the trained packing policy can be made more robust, while still maintaining acceptable performance in nominal cases.
In the AR2L framework, the training process involves alternating between training the packing policy, the permutation-based attacker, and the mixture-dynamics model in each iteration.
The [PPO algorithm](https://arxiv.org/abs/1707.06347) is utilized to train these three policies. 
Additionally, the packing policy is built on the [PCT algorithm](https://openreview.net/forum?id=bfuGjlCwAq).
The video demonstration can be found using the [YouTube Link](https://www.youtube.com/watch?v=xBxEp1fYqiU).

Importantly, we have updated our code with bug fixes to ensure more stable training and improved overall performance.
In this new version, performance may differ slightly from the old one, but overall, it is better, particularly in the continuous setting.
If you have already trained models using the old version, that is fine, as its main limitation was the slower training speed.

## Dependencies

Before executing the training process, please ensure that the necessary requirements have been installed.
```
conda create -n bpp_env python=3.7
pip install torch==1.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

## Training
The packing policy has the flexibility to observe a varying number of next boxes. 
The robustness of the policy can be adjusted by tuning the hyperparameter alpha.

**Environment: discrete**
```
bash scripts/train_disc.sh [number_of_next_boxes] [alpha]

Arguments:
[number_of_next_boxes]: 5, 10, 15, 20
[alpha]: 1.0
```

**Environment: continuous**
```
bash scripts/train_cont.sh [number_of_next_boxes] [alpha]

Arguments:
[number_of_next_boxes]: 5, 10, 15, 20
[alpha]: 1.0
```


## Evaluation
We have provided the pretrained model for you to use directly.

```
bash scripts/eval_disc.sh [number_of_next_boxes] [algorithm]

Arguments:
[number_of_next_boxes]: 5, 10, 15, 20
[algorithm]: ar2l, pct
```

```
bash scripts/eval_cont.sh [number_of_next_boxes] [algorithm]

Arguments:
[number_of_next_boxes]: 5, 10, 15, 20
[algorithm]: ar2l, pct
```


## Acknowledgement
We appreciate the anonymous reviewers, (S)ACs, and PCs of NeurIPS2023 for their insightful
comments to further improve our paper and their service to the community.
We would like to thank the authors of PCT for providing their highly valuable [implementation of PCT](https://github.com/alexfrom0815/Online-3D-BPP-PCT).
and the authors of the [PPO PyTorch Implementation](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail).


## Citation

```
@inproceedings{pan2023ar2l,
 author = {Pan, Yuxin and Chen, Yize and Lin, Fangzhen},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Neumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {51926--51954},
 publisher = {Curran Associates, Inc.},
 title = {Adjustable Robust Reinforcement Learning for Online 3D Bin Packing},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/a345ed605675c7c484e740a8ceaa6b45-Paper-Conference.pdf},
 volume = {36},
 year = {2023}
}
```

## License
This source code is provided solely for academic use. 
Please refrain from using it for commercial purposes without obtaining proper authorization from the author.

