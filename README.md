## Deep Self-Enhancement Hashing For Robust Multi-label Cross-modal Retrieval -- Tensorflow Implementation

### Deep Self-Enhancement Hashing (DSEH) 

The details can be found in the paper Deep Self-Enhancement Hashing For Robust Multi-label Cross-modal Retrieval (submitted to Pattern Recognition)

#### Implementation platform: 
* python 3.7  
* tensorflow 2.5.0 
* matlab 2016b

#### Datasets
We use the following datasets.

* MIRFlickr-25k  
* MSCOCO
* NUSWIDE

Pre-extracted features by ResNet-152:
* MIRFlickr-25k [download](https://pan.baidu.com/s/1THON8aDr0q4HLFzaTmkbfw) lmt3 


#### Training
The command for training is
* python3 train_DSEH.py
* our trained model (on MIRFlickr-25k) can be [download](https://pan.baidu.com/s/1lMCdFcyctM2fiarKi1oI7g) ve27
#### Evaluation
The command for evaluation is
* extract hash codes: python3 DSEH_encoding.py
* run evaluation.m