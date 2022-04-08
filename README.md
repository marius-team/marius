# Marius #

Marius is a system under active development for training embeddings for large-scale graphs on a single machine.

Training on large scale graphs requires a large amount of data movement to get embedding parameters from storage to the computational device. 
Marius is designed to mitigate/reduce data movement overheads using:
- Pipelined training and IO
- Partition caching and buffer-aware data orderings

Details on how Marius works can be found in our [OSDI '21 Paper](https://arxiv.org/abs/2101.08358), where experiment scripts and configurations can be found in the `osdi2021` branch.

## Requirements ##
(Other versions may work, but are untested)
* Ubuntu 18.04 or MacOS 10.15 
* CUDA >= 10.1
* CuDNN >= 7
* 1.7 <= pytorch < 1.10
* python >= 3.6
* pip >= 21
* GCC >= 9 (On Linux) or Clang 12.0 (On MacOS)
* cmake >= 3.12
* make >= 3.8


## Installation from source with Pip (currently broken, see development build instructions) ##

1. Install latest version of PyTorch for your CUDA version:

2. Clone the repository `git clone https://github.com/marius-team/marius-internal.git`

3. Build and install Marius `cd marius; python3 -m pip install .`


## Installation from source with CMake ##

1. Clone the repository `git clone https://github.com/marius-team/marius-internal.git`

2. Install dependencies `cd marius; python3 -m pip install -r requirements.txt`

3. Create build directory `mkdir build; cd build`

4. Run cmake in the build directory `cmake ../ -DUSE_OMP=1` (CPU-only build) or `cmake ../ -DUSE_OMP=1 -DUSE_CUDA=1` (GPU build)

5. Make the marius executable. `make marius_train -j`


## Development build ## 

For development, it is best to use CMake to build the c++ sources, and use pip to install the python tools seperately. The build process is as follows (assuming working from the cloned repository root directory):

1. Install python tools with `MARIUS_NO_BINDINGS=1 python3 -m pip install .`, the `MARIUS_NO_BINDINGS=1` environment variable tells pip not to build and install the C++ sources. So after this command you will have access to the marius_prepreprocess, marius_postprocess, and other python only modules. The `marius_train` and `marius_eval` executables will not be available and need to be built seperately.
2. To build the `marius_train` executable, follow the CMake instructions in the previous section. After following those instructions you can use the `marius_train` command by invoking `build/marius_train <your_config.yaml>`

When making modifications to the python, you only need to rerun `MARIUS_NO_BINDINGS=1 python3 -m pip install .`, and for the C++ you need to rerun the CMake build.

NOTE: You must supply the path to the executable built by CMake: `<build_dir>/marius_train <your_config.yaml>`. If you run command without supplying the path to the build, then the `marius_train` command installed by pip will be used, which will not work when pip is run with the `MARIUS_NO_BINDINGS=1` flag.

## Common development tasks ## 

### Adding a new Python command line tool ### 

1. Place the sources for your command line tool under `src/python/tools/<your_tool_name>`, make sure it has a main method as an entrypoint.
2. Modify the `[options.entry_points]` in setup.cfg to let pip know that it should install your tool as a new command. https://github.com/marius-team/marius-internal/blob/main/setup.cfg#L47

Use the existing command line tools as a guide if stuck.

Install your tool by using `MARIUS_NO_BINDINGS=1 python3 -m pip install .`. After any edits to your tools sources, you will have to rerun the pip install. To avoid reinstalling each time, you can to use the `-e` editable flag with pip: `MARIUS_NO_BINDINGS=1 python3 -m pip install -e .`. However, this flag hasn't been tested with our build so it's possible it will not work.

### Debugging with GDB ###

C++ error messages are usually not very helpful. For debugging mysterious errors, you can use GDB to run Marius and get a stack trace for where the program failed.

To use GDB, you will need to build Marius in Debug mode. You can do this by passing the `-DCMAKE_BUILD_TYPE=Debug` option to `cmake` when building. 

If you are using Docker to run Marius, you will have to run your container using `--cap-add=SYS_PTRACE` for GDB to work. 

Use `gdb <debug_build_dir>/marius_train` to enter GDB.

Once within the GDB console use `run <your_config.yaml>` to start execution. When your error is hit, gdb will stop at the line which threw the error.

## Training a graph ##

Before training you will need to preprocess your input dataset. Using the FB15K_237 knowledge graph as an example, you can do this by running `marius_preprocess --dataset fb15k_237 <output_directory>`. 

The <output_directory> specifies where the preprocessed graph will be output and is set by the user. In this example assume we have already have directory created called `datasets/fb15k_237_example/` which we set as the <output_directory>. 

NOTE: If the above command fails due to any missing directory errors, please create the `<output_directory>/edges` and `<output_directory>/nodes` directories as a workaround.

Training is run with `marius_train <your_config.yaml>` or `<build_dir>/marius_train <your_config.yaml>` if CMake was used to build the system. 

Example YAML config for the FB15K_237 dataset:

Note that the `base_directory` is set to the preprocessing output directory, `datasets/fb15k_237_example/`.
```
model:
  learning_task: LINK_PREDICTION
  encoder:
    layers:
      - - type: EMBEDDING
          output_dim: 50
  decoder:
    type: DISTMULT
  loss:
    type: SOFTMAX
  sparse_optimizer:
    type: ADAGRAD
    options:
      learning_rate: 0.1
storage:
  device_type: cuda
  dataset:
    base_directory: datasets/fb15k_237_example/
    num_edges: 272115
    num_train: 272115
    num_nodes: 14541
    num_relations: 237
    num_valid: 17535
    num_test: 20466
  edges:
    type: DEVICE_MEMORY
  embeddings:
    type: DEVICE_MEMORY
  save_model: true
training:
  batch_size: 1000
  negative_sampling:
    num_chunks: 10
    negatives_per_positive: 500
    degree_fraction: 0.0
    filtered: false
  num_epochs: 10
  pipeline:
    sync: true
  epochs_per_shuffle: 1
evaluation:
  batch_size: 1000
  negative_sampling:
    filtered: true
  pipeline:
    sync: true
```

After running this configuration, the MRR output by the system should be about .25 after 10 epochs.

## Using the Python API ##

TODO, Update Python API

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
```
FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu18.04
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
RUN python3 -m pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

RUN python3 -m pip install networkx scipy
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
OSDI Version (not yet available):
