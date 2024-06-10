
# Source-Context-KNN-MT

## Introduction
This repository includes the implementation of the paper "Revisiting Source Context in Nearest Neighbor Machine Translation" (EMNLP 2023). The code is implemented based on [kNN-box](https://github.com/NJUNLP/knn-box), and we are very grateful to the developers of kNN-box.

## Requirements
Following the settings of kNN-box, below are the environment requirements:
* python >= 3.7
* pytorch >= 1.10.0
* faiss-gpu >= 1.7.3
* sacremoses == 0.0.41
* sacrebleu == 1.5.1
* fastBPE == 0.1.0
* streamlit >= 1.13.0
* scikit-learn >= 1.0.2
* seaborn >= 0.12.1

After cloning this project:
```bash
cd source-context-knn-mt
pip install --editable ./
```

Installing faiss with the following commands:

```bash
# CPU version only:
conda install faiss-cpu -c pytorch

# GPU version:
conda install faiss-gpu -c pytorch # For CUDA
```

## Data Preparation
Obtain the data using the following command:
```bash
cd knnbox-scripts
bash prepare_dataset_and_model.sh
```

## Datastore Construction
Before training and inference, you need to build the datastore first. The following command will construct both the target and source side datastore:
```bash
cd knnbox-scripts/vanilla-knn-mt
bash build_datastore.sh
```

## Adaptive kNN-MT + Ours
Use the following commands to train and infer using the "Adaptive kNN-MT + Ours" method described in the paper:

```bash
# Ensure the datastore is constructed before training
cd knnbox-scripts/adaptive-knn-mt
bash train_metak.sh
bash inference.sh
```

## Robust kNN-MT + Ours
Use the following commands to train and infer using the "Robust kNN-MT + Ours" method described in the paper:

```bash
# Ensure the datastore is constructed before training
cd knnbox-scripts/robust-knn-mt
bash train_metak.sh
bash inference.sh
```

## Contacts
If you have any questions, please create an issue or contact me via email at xuanhong.li@mails.ccnu.edu.cn.

## Citation

```bibtex
@inproceedings{li-etal-2023-revisiting,
    title = "Revisiting Source Context in Nearest Neighbor Machine Translation",
    author = "Li, Xuanhong and Li, Peng and Hu, Po",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    year = "2023",
    url = "https://aclanthology.org/2023.emnlp-main.503",
    pages = "8087--8098",
}
