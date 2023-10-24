# Adjustable Robust Reinforcement Learning for Online 3D Bin Packing

## Introduction
This is the official PyTorch implementation for the paper titled ["Adjustable Robust Reinforcement Learning for Online 3D Bin Packing"](https://arxiv.org/pdf/2310.04323.pdf).
The paper introduces the AR2L framework, which takes into account both the average performance and worst-case performance of a packing policy. 
By using this framework, the trained packing policy can be made more robust, while still maintaining acceptable performance in nominal cases.
In the AR2L framework, the training process involves alternating between training the packing policy, the permutation-based attacker, and the mixture-dynamics model in each iteration.
The [PPO algorithm](https://arxiv.org/abs/1707.06347) is utilized to train these three policies. 
Additionally, the packing policy is built on the [PCT algorithm](https://openreview.net/forum?id=bfuGjlCwAq).
The video demonstration can be found using the [YouTube Link](https://www.youtube.com/watch?v=xBxEp1fYqiU).

## Dependencies

Before executing the training process, please ensure that the necessary requirements have been installed.
```
pip install -r requirements.txt
```

## Training
The packing policy has the flexibility to observe a varying number of next boxes (NNB). 
The robustness of the policy can be adjusted by tuning the hyperparameter **alpha**.

**Environment: discrete, NNB=5, alpha=1.0**
```
bash scripts train_disc.sh 5 1.0
```

**Environment: discrete, NNB=10, alpha=1.0**
```
bash scripts train_disc.sh 10 1.2
```

**Environment: discrete, NNB=15, alpha=1.0**
```
bash scripts train_disc.sh 15 1.3
```

**Environment: discrete, NNB=20, alpha=1.0**
```
bash scripts train_disc.sh 20 1.0
```

**Environment: continuous, NNB=5, alpha=1.0**
```
bash scripts train_cont.sh 5 1.0
```

**Environment: continuous, NNB=10, alpha=1.0**
```
bash scripts train_cont.sh 10 1.0
```

**Environment: continuous, NNB=15, alpha=1.0**
```
bash scripts train_cont.sh 15 1.0
```

**Environment: continuous, NNB=20, alpha=1.0**
```
bash scripts train_cont.sh 20 1.0
```


## Validation
To select an effective AR2L packing policy, you can evaluate various packing policies with and without the permutation-based attacker.
```
bash val_disc.sh [NNB] [path to the parent directory where all the models are saved] load_adv

example: bash val_disc.sh 5 ./logs/experiment/timeStr load_adv
```

```
bash val_disc.sh [NNB] [path to the parent directory where all the models are saved] not_load_adv

example: bash val_disc.sh 5 ./logs/experiment/timeStr not_load_adv
```
After conducting the validation, please add the space utilization in the nominal dynamics (not_load_adv) and the space utilization in the worst-case dynamics (load_adv) for each model.
Then, you can choose the best one among them.



## Evaluation
You can evaluate the selected packing policy in various settings.
```
bash eval_disc.sh [NNB] [path to the BPP model] [path to the adv model] load_adv

example: bash eval_disc.sh 5 ./logs/experiment/timeStr/BPP-subtimeStr.pt ./logs/experiment/timeStr/Adv-subtimeStr.pt load_adv
```

```
bash eval_disc.sh [NNB] [path to the BPP model] [path to the adv model] not_load_adv

example: bash eval_disc.sh 5 ./logs/experiment/timeStr/BPP-subtimeStr.pt ./logs/experiment/timeStr/Adv-subtimeStr.pt not_load_adv
```


## Acknowledgement
We appreciate the anonymous reviewers, (S)ACs, and PCs of NeurIPS2023 for their insightful
comments to further improve our paper and their service to the community.
We would like to thank the authors of PCT for providing their highly valuable [implementation of PCT](https://github.com/alexfrom0815/Online-3D-BPP-PCT).
and the authors of the [PPO PyTorch Implementation](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail).


## Citation

```
@article{pan2023adjustable,
  title={Adjustable Robust Reinforcement Learning for Online 3D Bin Packing},
  author={Pan, Yuxin and Chen, Yize and Lin, Fangzhen},
  journal={arXiv preprint arXiv:2310.04323},
  year={2023}
}
```

## License
This source code is provided solely for academic use. 
Please refrain from using it for commercial purposes without obtaining proper authorization from the author.

