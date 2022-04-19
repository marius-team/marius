# Marius #

Marius is a system for training graph neural networks and embeddings for large-scale graphs on a single machine.

Marius ([OSDI '21 Paper](https://www.usenix.org/conference/osdi21/presentation/mohoney)) is designed to mitigate/reduce data movement overheads using:
- Pipelined training and IO
- Partition caching and buffer-aware data orderings

We scale graph neural network training ([preprint](https://arxiv.org/abs/2202.02365)) through:
- Optimized datastructures for neighbor sampling and GNN aggregation
- Out-of-core GNN training 

## Build and Install ##

### Requirements ###

* CUDA >= 10.1
* CuDNN >= 7 
* pytorch >= 1.8
* python >= 3.6
* GCC >= 7 (On Linux) or Clang 12.0 (On MacOS)
* cmake >= 3.12
* make >= 3.8

### Pip Installation ###

```
git clone https://github.com/marius-team/marius.git
pip3 install .
```



The Python API can be accessed with ``import marius``

The following commands will be installed:
- marius_train: Train models using configuration files and the command line
- marius_eval: Command line model evaluation
- marius_preprocess: Built-in dataset downloading and preprocessing
- marius_predict: Batch inference tool for link prediction or node classification

## Command Line Training ##

First make sure marius is installed with `pip3 install .` 

Preprocess dataset the FB15K_237 dataset with `marius_preprocess --dataset fb15k_237 --output_dir datasets/fb15k_237_example/`

Train example configuration file (assuming we are in the repo root directory) `marius_train examples/configuration/fb15k_237.yaml`

After running this configuration, the MRR output by the system should be about .25 after 10 epochs.

Perform batch inference on the test set with `marius_predict --config examples/configuration/fb15k_237.yaml --metrics mrr --save_scores --save_ranks`

See the [full example](http://marius-project.org/marius/examples/config/lp_fb15k237.html#small-scale-link-prediction-fb15k-237) for details.

## Python API ##


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

Marius++ (out-of-core GNN training)
```
@misc{waleffe2022marius,
  doi = {10.48550/ARXIV.2202.02365},
  url = {https://arxiv.org/abs/2202.02365},
  author = {Waleffe, Roger and Mohoney, Jason and Rekatsinas, Theodoros and Venkataraman, Shivaram},
  keywords = {Machine Learning (cs.LG), Databases (cs.DB), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Marius++: Large-Scale Training of Graph Neural Networks on a Single Machine},
  publisher = {arXiv},
  year = {2022},
```
