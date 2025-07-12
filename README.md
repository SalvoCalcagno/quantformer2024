<div align="center">

# QuantFormer: Learning to quantize for neural activity forecasting in mice visual cortex

</div>

# Overview
Official PyTorch implementation of paper: <b>"QuantFormer: Learning to quantize for neural activity forecasting in mice visual cortex"</b>

# Abstract
Understanding complex animal behaviors hinges on deciphering the neural activity patterns within brain circuits, making the ability to forecast neural activity crucial for developing predictive models of brain dynamics. This capability holds immense value for neuroscience, particularly in applications such as real-time optogenetic interventions. While traditional encoding and decoding methods have been used to map external variables to neural activity and vice versa, they focus on interpreting past data. In contrast, neural forecasting aims to predict future neural activity, presenting a unique and challenging task due to the spatiotemporal sparsity and complex dependencies of neural signals.
Existing transformer-based forecasting methods, while effective in many domains, struggle to capture the distinctiveness of neural signals characterized by spatiotemporal sparsity and intricate dependencies.
To address this challenge, we here introduce *QuantFormer*, a transformer-based model specifically designed for forecasting neural activity from two-photon calcium imaging data. Unlike conventional regression-based approaches, *QuantFormer* reframes the forecasting task as a classification problem via dynamic signal quantization, enabling more effective learning of sparse neural activation patterns. Additionally, *QuantFormer* tackles the challenge of analyzing multivariate signals from an arbitrary number of neurons by incorporating neuron-specific tokens, allowing scalability across diverse neuronal populations.
Trained with unsupervised quantization on the Allen dataset, *QuantFormer* sets a new benchmark in forecasting mouse visual cortex activity. It demonstrates robust performance and generalization across various stimuli and individuals, paving the way for a foundational model in neural signal prediction.

# How to run

## Pre-requisites
- NVIDIA GPU (Tested on Nvidia A6000 GPUs )
- Wandb account (change entity and project name in scripts)
- Conda environment

## Train QuantFormer

### **Pretrain a quantizer**

```
python train_quantformer.py --dataset pretrain --stimulus drifting_gratings
```

### **Finetune on classification**

```
python train_quantcoder.py --container-id 511507650 --stimulus drifting_gratings --use-neuron-embedding --quantizer <your-pretrained-quantformer> --cls
```

### **Finetune on forecasting**

```
 python train_quantcoder.py --container-id 511507650 --stimulus drifting_gratings --use-neuron-embedding --quantizer <your-pretrained-quantformer> --criterion focal
```

## **Train baselines**

### LSTM

```
python train.py --model autoregressive_lstm --container-id 511507650 --stimulus drifting_gratings
```
```
python train.py --model lstm_classification --container-id 511507650 --stimulus drifting_gratings 
```

### Informer

```
python train.py --model informer --container-id 511507650 --stimulus drifting_gratings --cls
```
```
python train.py --model informer --container-id 511507650 --stimulus drifting_gratings
```
### Autoformer

```
python train.py --model autoformer --container-id 511507650 --stimulus drifting_gratings --cls
```
```
python train.py --model autoformer --container-id 511507650 --stimulus drifting_gratings
```
### Crossformer

```
python train.py --model crossformer --container-id 511507650 --stimulus drifting_gratings --model crossformer --cls
```
```
python train.py --model crossformer --container-id 511507650 --stimulus drifting_gratings --model crossformer
```

# References
[1] Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. CoRR, abs/2106.13008, 2021.

[2] Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pages 11106–11115, 2021.

[3] YunhaoZhangandJunchiYan. Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting. In The eleventh international conference on learning representations, 2022.

[4] Josue Ortega Caro, Antonio Henrique Oliveira Fonseca, Christopher Averill, Syed A Rizvi, Matteo Rosati, James L Cross, Prateek Mittal, Emanuele Zappala, Daniel Levine, Rahul M Dhodapkar, et al. Brainlm: A foundation model for brain activity recordings. bioRxiv, pages 2023–09, 2023.

[5] Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In International Conference on Learning Representations.

[6] Minyoung Huh, Brian Cheung, Pulkit Agrawal, and Phillip Isola. Straightening out the straight through estimator: Overcoming optimization challenges in vector quantized networks. In International Conference on Machine Learning. PMLR, 2023.
