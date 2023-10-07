# MAda-SR
Modal Adaptive Super-Resolution for Medical Images via Continual Learning
> This repository is for MAda-SR introduced in the following paper.
> The code is built on HAN (PyTorch) and tested on Ubuntu 16.04/18.04 environment (Python3.9, PyTorch_1.12.0, CUDA11.7) with GeForce RTX3090 GPUs.

### Contents

> 1. [Introduction](https://github.com/wuzheng2022/MAda-SR#introduction)
> 2. [Train](https://github.com/wuzheng2022/MAda-SR#begin-to-train)
> 3. [Test](https://github.com/wuzheng2022/MAda-SR#begin-to-test)
> 4. [Acknowledgements](https://github.com/wuzheng2022/MAda-SR#Acknowledgements)

### Introduction
We proposed a multi-modal adaptive super-resolution algorithm for reconstructing CT and MRI scans, named MAda-SR, which improves the traditional Adam optimizer into an adaptive optimizer in terms of parameter updates and optimization strategies.

### Begin to train


### Begin to Test
Single task Test Cmd
> python main.py --mode mhan --data_train Medical --data_test Medical --lml icarl -- reg_lambda 0.01 --scale 4 --pre_train ../experiment1/FFMx4_icarl/task_1_PD/model/model_latest.pt --save my_test --save_results

Multiple task Test Cmd
> python multimain.py --model mhan --data_train Medical --data_test Medical --lml icarl -- reg_lambda 0.01 --scale 4 --pre_train ../experiment1/FFMx4_icarl/task_1_PD/model/model_latest.pt --save my_test --save_results 


### Acknowledgements
> This code is built on [HAN](https://github.com/rwenqi/HAN). We thank the authors for sharing their codes of HAN  [PyTorch version](https://github.com/rwenqi/HAN).
