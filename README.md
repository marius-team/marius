# Marius #

Marius is a system for large-scale graph learning. The system is currently in the alpha phase and is under active development.

Details on how Marius works can be found in our [OSDI '21 Paper](https://arxiv.org/abs/2101.08358), where experiment scripts and configurations can be found in the `osdi2021` branch.

Currently we support:
- Large-scale link prediction training
- Preprocessing and training of datasets in CSV format (single-file)
- Configuration file based API
- Single GPU training and evaluation
- Dataset sizes that fit in: GPU memory, CPU memory, and Disk.

See `docs/user_guide` for more details.

We are working on expanding the functionality of Marius to include:
- Graph neural network support
- Multi-GPU training
- Node classification
- Python API for user defined models, sampling and training procedures

## Requirements ##
(Other versions may work, but are untested)
* Ubuntu 18.04 or MacOS 10.15 
* CUDA >= 10 (If using GPU training)
* 1.7 <= pytorch < 1.10
* python >= 3.6
* pip >= 21
* GCC >= 9 (On Linux) 
* Clang >= 11 (On MacOS)
* cmake >= 3.12
* make >= 3.8


## Installation from source with Pip ##

1. Install latest version of PyTorch for your CUDA version: https://pytorch.org/get-started/locally/

2. Clone the repository `git clone https://github.com/marius-team/marius.git`

3. Build and install Marius `cd marius; python3 -m pip install .`

#### Full script (without torch install) ####

```
git clone https://github.com/marius-team/marius.git
cd marius
python3 -m pip install .
```

## Training a graph ##

Training embeddings on a graph requires three steps. 

1. Define a configuration file. This example will use the config already defined in `examples/training/configs/fb15k_gpu.ini`

   See `docs/configuration.rst` for full details on the configuration options.

2. Preprocess the dataset `marius_preprocess output_dir/ --dataset fb15k`

   This command will download the freebase15k dataset and preprocess it for training, storing files in `output_dir/`. If a different output directory is used, the configuration file's path options will need to be updated accordingly.

3. Run the training executable with the config file `marius_train examples/training/configs/fb15k_gpu.ini`. 

The output of the first epoch should be similar to the following.
```[info] [03/18/21 01:33:16.173] Start preprocessing
[info] [03/18/21 01:33:18.778] Metadata initialized
[info] [03/18/21 01:33:18.778] Training set initialized
[info] [03/18/21 01:33:18.779] Evaluation set initialized
[info] [03/18/21 01:33:18.779] Preprocessing Complete: 2.605s
[info] [03/18/21 01:33:18.791] ################ Starting training epoch 1 ################
[info] [03/18/21 01:33:18.836] Total Edges Processed: 40000, Percent Complete: 0.082
[info] [03/18/21 01:33:18.862] Total Edges Processed: 80000, Percent Complete: 0.163
[info] [03/18/21 01:33:18.892] Total Edges Processed: 120000, Percent Complete: 0.245
[info] [03/18/21 01:33:18.918] Total Edges Processed: 160000, Percent Complete: 0.327
[info] [03/18/21 01:33:18.944] Total Edges Processed: 200000, Percent Complete: 0.408
[info] [03/18/21 01:33:18.970] Total Edges Processed: 240000, Percent Complete: 0.490
[info] [03/18/21 01:33:18.996] Total Edges Processed: 280000, Percent Complete: 0.571
[info] [03/18/21 01:33:19.021] Total Edges Processed: 320000, Percent Complete: 0.653
[info] [03/18/21 01:33:19.046] Total Edges Processed: 360000, Percent Complete: 0.735
[info] [03/18/21 01:33:19.071] Total Edges Processed: 400000, Percent Complete: 0.816
[info] [03/18/21 01:33:19.096] Total Edges Processed: 440000, Percent Complete: 0.898
[info] [03/18/21 01:33:19.122] Total Edges Processed: 480000, Percent Complete: 0.980
[info] [03/18/21 01:33:19.130] ################ Finished training epoch 1 ################
[info] [03/18/21 01:33:19.130] Epoch Runtime (Before shuffle/sync): 339ms
[info] [03/18/21 01:33:19.130] Edges per Second (Before shuffle/sync): 1425197.8
[info] [03/18/21 01:33:19.130] Edges Shuffled
[info] [03/18/21 01:33:19.130] Epoch Runtime (Including shuffle/sync): 339ms
[info] [03/18/21 01:33:19.130] Edges per Second (Including shuffle/sync): 1425197.8
[info] [03/18/21 01:33:19.148] Starting evaluating
[info] [03/18/21 01:33:19.254] Pipeline flush complete
[info] [03/18/21 01:33:19.271] Num Eval Edges: 50000
[info] [03/18/21 01:33:19.271] Num Eval Batches: 50
[info] [03/18/21 01:33:19.271] Auc: 0.973, Avg Ranks: 24.477, MRR: 0.491, Hits@1: 0.357, Hits@5: 0.651, Hits@10: 0.733, Hits@20: 0.806, Hits@50: 0.895, Hits@100: 0.943
```

To train using CPUs only, use the `examples/training/configs/fb15k_cpu.ini` configuration file instead.

## Using the Python API ##

### Sample Code ###

Below is a sample python script which trains a single epoch of embeddings on fb15k.
```
import marius as m
from marius.tools import preprocess

def fb15k_example():

    preprocess.fb15k(output_dir="output_dir/")
    
    config_path = "examples/training/configs/fb15k_cpu.ini"
    config = m.parseConfig(config_path)

    train_set, eval_set = m.initializeDatasets(config)

    model = m.initializeModel(config.model.encoder_model, config.model.decoder_model)

    trainer = m.SynchronousTrainer(train_set, model)
    evaluator = m.SynchronousEvaluator(eval_set, model)

    trainer.train(1)
    evaluator.evaluate(True)


if __name__ == "__main__":
    fb15k_example()
```

## Marius in Docker ##

Marius can be deployed within a docker container. Here is a sample ubuntu dockerfile (located at `examples/docker/dockerfile`) which contains the necessary dependencies preinstalled for GPU training.

### Building and running the container ###

Build an image with the name `marius` and the tag `example`:  
`docker build -t marius:example -f examples/docker/dockerfile examples/docker`

Create and start a new container instance named `gaius` with:  
`docker run --name gaius -itd marius:example`

Run `docker ps` to verify the container is running

Start a bash session inside the container:  
`docker exec -it gaius bash`


### Sample Dockerfile ###
See `examples/docker/dockerfile`
```
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
RUN apt update

RUN apt install -y g++ \ 
         make \
         wget \
         unzip \
         vim \
         git \
         python3-pip

# install gcc-9
RUN apt install -y software-properties-common
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt update
RUN apt install -y gcc-9 g++-9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9

# install cmake 3.20
RUN wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0-linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh cmake-3.20.0-linux-x86_64.sh --skip-license --prefix=/opt/cmake/
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake

# install pytorch
RUN python3 -m pip install torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

## Citing Marius ##
Arxiv Version:
```
@misc{mohoney2021marius,
      title={Marius: Learning Massive Graph Embeddings on a Single Machine}, 
      author={Jason Mohoney and Roger Waleffe and Yiheng Xu and Theodoros Rekatsinas and Shivaram Venkataraman},
      year={2021},
      eprint={2101.08358},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
OSDI Version:
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
