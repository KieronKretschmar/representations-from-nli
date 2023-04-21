# Overview
This project is a partial reimplementation of the paper "Supervised learning of universal sentence representations from natural language inference data" from Conneau et al. 2017. 

In this project we train models to generate sentence embeddings utilizing GloVe and the SNLI corpus, and analyze their performance on downstream tasks using the SentEval package by Conneau et al. (2018).

Repository structure:
```
    .
    ├── SentEval/                       # a partial copy of the SentEval repository
    ├── data/                           # folder containing data classes, where datasets will be placed
    │   └── __init__.py
    │   └── snli_data_module.py         # a LightningDataModule for the SNLI dataset
    │   └── snli_dataset.py             # a dataset class for the SNLI dataset
    │   └── vocabulary.py               # a vocabulary class for holding word embeddings
    ├── images/                         # folder containing image
    ├── models/                         # folder containing model classes
    │   └── __init__.py
    │   └── batcher.py                  # a wrapper class for evaluating encoders with SentEval
    │   └── classifier.py               # a classifier for the NLI task
    │   └── nli_model.py                # a LightningModule containing a sentence encoder and a classifier
    │   └──  sentence_encoders.py       # classes for multiple sentence encoders
    ├── senteval_results/               # folder containing detailed senteval evaluation results
    ├── slurm_jobs/                     # jobs for running code on LISA or Snellius cluster
    ├── .gitignore                      # .gitignore
    ├── README.md                       # README
    ├── LICENSE                         # MIT License
    ├── analysis.ipynb                  # jupyter notebook containing results and analysis
    ├── environment.yml                 # the conda environment specification
    ├── eval.py                         # script for evaluating models on SNLI and SentEval
    └── train.py                        # script for training models on SNLI
```

# Setup
Install the packages referenced in `environment.yml`. For example by creating a conda environment with `conda env create --name myenv -f environment.yml`.

## SentEval
For evaluation, we further have to set up [SentEval](https://github.com/facebookresearch/SentEval). A copy of the necessary parts for evaluation is included in this repository. However, the package needs to be installed and the datasets, which are excluded, and need to be downloaded seperately.

Users of Lisa or Snellius can use the `install_senteval.job` script that is part of this repository. It will apply the steps described below.

1. To install SentEval, navigate into the `SentEval/` directory, make sure to activate the environment created in the previous step with `conda activate myenv`, and install SentEval as a module with `python setup.py install`.

2. The datasets have to be downloaded by executing the bash script located at `SentEval/data/downstream/get_transfer_data.bash`.

## GloVe
The 840B-token 300-dimensional GloVe embeddings are required to train and evaluate models. If the `glove.840B.300d.txt` is already available, it can be used by placing it in the `data/glove/` directory. Otherwise, it will be automatically downloaded and placed there.

## SNLI
The SNLI dataset will automatically be downloaded from the Hugging Face Hub with the `datasets` library. If a cached version already exists, it will be used. 

## Pretrained models and vocabulary
At the time of writing, checkpoints of 8 pretrained sentence encoders, one classifier for the natural language inference task for one of these encoders, and 2 vocabularies, which contain the embeddings, are avilable on [Google Drive](https://drive.google.com/drive/folders/1XMYTyJigxU4JLDfhy-Y7VIWd5HOlr6u0?usp=sharing). These can be downloaded and evaluated. For more info on how to evaluate them, see sections below as well as `demo.ipynb` and `analysis.ipynb`.

# Training models
Models can be trained using the `train.py` script. For more information about the arguments, please look at `train.py`.

An example call would look like this:
```terminal
cd representations-from-nli
conda activate myenv

python train.py --logger wandb --batch-size 64 --max-epochs 25 --num-workers 3 --save-name my_unilstm --encoder-name unilstm
```

At the end of training, the best checkpoint, according to validation accuracy, will be saved to `checkpoints/best/<save_name>.ckpt`.

Users of Lisa or Snellius can train models with the `run_training.job` script that is part of this repository.

# Evaluating models
Checkpoints of models created with `train.py` can be evaluated using the `eval.py` script. For more information about the arguments, please look at `eval.py`.

An example call would look like this:
```terminal
cd representations-from-nli
conda activate myenv

python -u eval.py --encoder-checkpoint-path ./checkpoints/best/my_unilstm.ckpt --results-save-name my_unilstm_results --eval-task senteval
```

If `--eval-task senteval` was specified, then an object holding the detailed results will be stored in `senteval_results/<results-save-name>.pkl` at the end of evaluation.

Users of Lisa or Snellius can use the `run_eval.job` script that is part of this repository.

# Demo
This repository contains a demo.ipynb with examples of how to use the trained models.

# Results and Analysis
This repository contains a analysis.ipynb, where test results for SNLI and SentEval are shown and discussed. 

# Design choices
The parameters in `train.py` and `eval.py` allow for some flexibility in altering training and evaluation.

One decision that is currently not configurable is the alignment of the vocabulary. We have chosen to align the word embedding layer with the words from the SNLI corpus for which GloVe embeddings are available. We have chosen not to include additional words appearing in downstream-tasks because this project is about evaluating *general* sentence representations for which the downstream tasks (e.g. SentEval) are unknown.

Note that we would expect the performance on SentEval to go up if the test vocabulary was known during training, although the encoder would never see those exact tokens during training, because it might learn useful information from having seen *similar* tokens' embeddings.

# References
- S. R. Bowman, G. Angeli, C. Potts, and C. D. Manning. A large annotated corpus for learning natural language inference. arXiv preprint arXiv:1508.05326, 2015.
- A. Conneau and D. Kiela. Senteval: An evaluation toolkit for universal sentence representations. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC-2018), 2018.
- A. Conneau, D. Kiela, H. Schwenk, L. Barrault, and A. Bordes. Supervised learning of universal sentence representations from natural language inference data. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 670–680, Copenhagen, Denmark, September 2017. Association for Computational Linguistics.
- Philipp Lippe's Deep Learning tutorial on pytorch lightning (https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html)