# Knowledgeable Parameter Efficient Tuning Network for Commonsense Question Answering

Codes for ACL2023 paper: [Knowledgeable Parameter Efficient Tuning Network for Commonsense Question Answering](https://aclanthology.org/2023.acl-long.503/).

## Overview

![](./image/img1.png)

## Usage

#### Installation

```shell
pip install -r requirements.txt
```

#### Runing

- Download conceptnet
  
  ```shell
  cd ./data
  wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
  gzip -d conceptnet-assertions-5.7.0.csv.gz
  ```

- Generate knowledge
  
  ```shell
  cd ../
  python preprocess/construct_graph.py
  python preprocess/find_triples.py csqa2
  python preprocess/find_triples_knowledge.py csqa2
  ```

- Run
  
  ```shell
  python -m torch.distributed.launch --nproc_per_node 4 train_ddp.py
  ```

### Citation

```latex
@inproceedings{zhao-etal-2023-knowledgeable,
    title = "Knowledgeable Parameter Efficient Tuning Network for Commonsense Question Answering",
    author = "Zhao, Ziwang  and
      Hu, Linmei  and
      Zhao, Hanyu  and
      Shao, Yingxia  and
      Wang, Yequan",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023"
}
```