# Specialized Embedding Approximation Training Pipeline
This repository contains the training code for Specialized Embedding Approximation in the context of urban noise classification as described in our IEEE ICASSP 2021 paper [Specialized Embedding Approximation for Edge Intelligence: A Case Study in Urban Sound Classification](https://ieeexplore.ieee.org/document/9414287).

## Overview of Training Pipeline
The pipeline for solving Specialized Embedding Approximation (SEA) involves two components: (i) Dimensionality Reduction and (ii) Knowledge Distillation. The first step corresponds to learning a dimensionality reduction function <img src="https://render.githubusercontent.com/render/math?math=\phi"> from the teacher embedding model's data space <img src="https://render.githubusercontent.com/render/math?math=\mathbb{R}^n"> to the student's subspace <img src="https://render.githubusercontent.com/render/math?math=\mathbb{R}^d">. In order to reduce the memory and compute complexity associated with learning <img src="https://render.githubusercontent.com/render/math?math=\phi"> without compromising the geometric interpretation in the <img src="https://render.githubusercontent.com/render/math?math=\mathbb{R}^d"> subspace, we use a sampling technique to choose a representative subset of data points from the student's training data. The second step is to transform the teacher's embeddings in the student's data space using <img src="https://render.githubusercontent.com/render/math?math=\phi">, and then train the student to learn the resulting embeddings using Mean Squared Error (MSE) loss. The overall pipeline is outlined below:

<img src="EA_Pipeline-1.png" alt="EA-Pipeline"/>

## Knowledge Distillation Pipeline
### Prerequisites
1. [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. `libsndfile`: On Windows and macOS, these will be installed via `pip`. However, on Linux this must be installed manually via your platform's package manager. For Debian-based distributions (such as Ubuntu), this can be done by running:

    ```
    apt-get install libsndfile1
    ```
3. [Gooogle Sheets API](https://developers.google.com/sheets/api/quickstart/python#prerequisites): Complete the _Prerequisites_ step of [creating a project with the API enabled](https://developers.google.com/workspace/guides/create-project).

### Setup
Create a virtual environment `l3embedding-tf-12-gpu` using the `l3embedding-tf-12-gpu.yml` file in this repository:
```
conda env create -f l3embedding-tf-12-gpu.yml
```

### Knowledge Distillation
Knowledge Distillation training involves submitting the following job 

## How to Cite
Kindly cite our work as:

```
@inproceedings{srivastava2021specialized,
  title={Specialized Embedding Approximation for Edge Intelligence: A Case Study in Urban Sound Classification},
  author={Srivastava, Sangeeta and Roy, Dhrubojyoti and Cartwright, Mark and Bello, Juan P and Arora, Anish},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={8378--8382},
  year={2021},
  organization={IEEE}
}
```
