# Marius and MariusGNN #

This repository contains the code for the Marius and MariusGNN papers. 
We have combined the two works into one unified system for training 
graph embeddings and graph neural networks over large-scale graphs 
on a single machine using the entire memory hierarchy.

Marius ([OSDI '21 Paper](https://www.usenix.org/conference/osdi21/presentation/mohoney)) is designed to mitigate/reduce data movement overheads for graph embeddings using:
- Pipelined training and IO
- Partition caching and a buffer-aware data ordering to minimize IO for disk-based training (called BETA)

MariusGNN ([arxiv](https://arxiv.org/abs/2202.02365), to appear in EuroSys '23) 
utilizes the data movement optimizations from Marius and adds support for scalable graph neural network training through:
- An optimized data structure for neighbor sampling and GNN aggregation (called DENSE)
- An improved data ordering for disk-based training (called COMET) which minimizes IO and maximizes model accuracy (note that COMET subsumes BETA)

## Build and Install ##

### Requirements ###

* CUDA >= 10.1
* CuDNN >= 7 
* pytorch >= 1.8
* python >= 3.6
* GCC >= 7 (On Linux) or Clang >= 11.0 (On MacOS)
* cmake >= 3.12
* make >= 3.8

### Pip Installation ###

```
git clone https://github.com/marius-team/marius.git
pip3 install .
```



The Python API can be accessed with ``import marius``

The following command line tools will be installed:
- marius_train: Train models using configuration files and the command line
- marius_eval: Command line model evaluation
- marius_preprocess: Built-in dataset downloading and preprocessing
- marius_predict: Batch inference tool for link prediction or node classification

## Command Line Interface ##

The command line interface supports performant in-memory and out-of-core 
training and evaluation of graph learning models. Experimental results 
from our papers can be reproduced using this interface (we also provide
an exact experiment artifact for each paper in separate branches).

### Quick Start: ###

First make sure Marius is installed with `pip3 install .` 

Preprocess the FB15K_237 dataset with `marius_preprocess --dataset fb15k_237 --output_dir datasets/fb15k_237_example/`

Train using the example configuration file (assuming we are in the root directory of the repository) `marius_train examples/configuration/fb15k_237.yaml`

After running this configuration file, the MRR output by the system should be about .25 after 10 epochs.

Perform batch inference on the test set with `marius_predict --config examples/configuration/fb15k_237.yaml --metrics mrr --save_scores --save_ranks`

See the [full example](http://marius-project.org/marius/examples/config/lp_fb15k237.html#small-scale-link-prediction-fb15k-237) for details.

## Python API ##

The Python API is currently experimental and can be used to perform in-memory training and evaluation of graph learning models. 

See the [documentation](http://marius-project.org/marius/examples/python/index.html#) for Python API usage and examples.


## Citing Marius ##
Marius (out-of-core graph embeddings)
```
@inproceedings {273733,
    author = {Jason Mohoney and Roger Waleffe and Henry Xu and Theodoros Rekatsinas and Shivaram Venkataraman},
    title = {Marius: Learning Massive Graph Embeddings on a Single Machine},
    booktitle = {15th {USENIX} Symposium on Operating Systems Design and Implementation ({OSDI} 21)},
    year = {2021},
    isbn = {978-1-939133-22-9},
    pages = {533--549},
    url = {https://www.usenix.org/conference/osdi21/presentation/mohoney},
    publisher = {{USENIX} Association},
    month = jul,
}
```

MariusGNN (out-of-core GNN training)
```
@misc{waleffe2022marius,
  doi = {10.48550/ARXIV.2202.02365},
  url = {https://arxiv.org/abs/2202.02365},
  author = {Waleffe, Roger and Mohoney, Jason and Rekatsinas, Theodoros and Venkataraman, Shivaram},
  keywords = {Machine Learning (cs.LG), Databases (cs.DB), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {MariusGNN: Resource-Efficient Out-of-Core Training of Graph Neural Networks},
  publisher = {arXiv},
  year = {2022},
```
