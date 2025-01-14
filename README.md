# PromptReader

Code and data of the paper "A Multi-Round MRC Framework incorporating Prompt Learning for Aspect Sentiment Triple Extraction" 

Authors:  Yuyao Zhang,Zhiyuan Yan,Xiaodian Zhang.

## Requirements

```
   python==3.7.2
   torch==1.8.0+cu111
   transformers==4.21.2
   spacy==2.3.8
```

## Datastes

| Data                                                         | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [ASTE-Data-V1](https://github.com/xuuuluuu/SemEval-Triplet-data/tree/master/ASTE-Data-V1-AAAI2020) | These data are originally used in the [AAAI-2020 paper](https://arxiv.org/pdf/1911.01616.pdf). |
| [ASTE-Data-V2](https://github.com/xuuuluuu/SemEval-Triplet-data/tree/master/ASTE-Data-V2-EMNLP2020) | These data are originally used in the [EMNLP-2020 paper](https://arxiv.org/abs/2010.02609). |

## Reproduction

### Data Preprocess

```
  python ./scripts/DataProcessV1.py  #The results of data preprocessing will be placed in the ./data/preprocess/.
```

## Train and Test the PromptReader model:

```
  python ./scripts/Main.py --mode train 
  python ./scripts/Main.py --mode test 
```

The parameters of model is saved in ./scripts/model/, and the log records are stored in ./scripts/log/.
