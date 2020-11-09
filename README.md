# CogLTX

CogLTX is a framework to apply current BERT-like pretrained language models to long texts. CogLTX does not need new Transformer structures or pretraining, but want to put forward a solution in finetuning and inference. See the paper (http://keg.cs.tsinghua.edu.cn/jietang/publications/NIPS20-Ding-et-al-CogLTX.pdf) for details. 

This repo is a preview version, we will work out a more easy-to-use version in the future. You need to wait for some time due to another project for the first author, and be assured that we are not forgetting!

## Environment
The script `setup_env.sh` describe how to set up an environment by conda. The most important part is 
```
pip install torch==1.3.1 torchvision==0.4.2 transformers==2.4.1 pytorch-lightning==0.6 gensim ujson fuzzywuzzy
``` 
Note that the backward compatiblity of pytorch-lightning is not good, and some bugs in the 0.6 version are fixed manually in the code. We will change the framework in the later version of CogLTX.

## Preprocess
The preprocess is different for different types of data, but in general can be split into 3 phases:
1. Read and tokenize the data.
2. Split the long text into blocks by `Buffer.split_document_into_blocks(document, tokenizer, cnt=0, hard=True, properties=None)`. The `hard` parameter switches the mode, i.e. whether to use dynamic programming to decide the separation. `properties` are associated information with the document, usually labels.
3. Save them as a list of samples by pickle.

See the folders (`newsqa  hotpotqa  20news`) for details.
## Running
To train CogLTX on a specific task, we need write an entry like `run_newsqa.py, run_hotpotqa.py, run_20news.py`, which includes the configuration, postprocess and evaluation. 

## Others
The data of NewsQA, hotpotQA and 20news can be found in the original dataset paper, but we do not release the codes and data about Alibaba due to commercial reasons. 
