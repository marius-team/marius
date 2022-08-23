# EuroSys 23' MariusGNN Artifact #

MariusGNN is a system for resource-efficient training of graph neural networks (GNNs) over large-scale 
graphs on a single machine. To support such training, MariusGNN uses two main techniques as described in our
[EuroSys '23 Paper]():

- Optimized data structures for neighborhood sampling and GNN aggregation
- Partition replacement policies for out-of-core GNN training

This branch contains the artifact used to produce the experiments reported in the paper. **Note that Marius/MariusGNN
are under active development. Thus, the main branch has changed and will continue to do so from the code supplied
here.** In particular, the configuration file format has changed. The configs used here will not run directly using the
main branch (or vice versa). For the most up-to-date Marius/MariusGNN see [here](https://github.com/marius-team/marius).

This artifact is licensed under the Apache 2.0 License as described in the LICENSE file.



## Getting Started ##

### Build Information and Environment ###
The following denotes the dependencies and versions used for this artifact and how to build this version of MariusGNN. 
**We highly recommend using Docker for installation as described below**.

### Artifact Dependencies and Versions Used for Paper ###
* Ubuntu 18.04
* CUDA 11.4
* CuDNN 8
* PyTorch 1.9.1
* Python 3.6.9
* Pip 9.0.1
* GCC 9
* CMake 3.20
* Make 4.1
* DGL 0.7.0
* PyTorch Geometric (PyG) 2.03
* dstat

Note that PyTorch/DGL/PyG need to be installed with CUDA support for GPU training. See `examples/docker/Dockerfile` for
installation commands.

**Strictly speaking, this artifact requires GPU support. However, we provide a minimal working example which runs
MariusGNN and baseline systems on the CPU for those who do not have a GPU machine**. 
CPU only installation does not require CUDA/CuDNN or CUDA installations of PyTorch/DGL/PyG. **Even for CPU only support
we recommend the Docker installation described below.**

### End-to-End Docker Installation (Recommended) ###
The following Docker installation installs the necessary dependencies for this artifact and builds MariusGNN. It
requires Docker to be installed (Docker can generally be installed easily using your favorite package
manager), as well as the NVIDIA drivers for GPU support
(check by running `nvidia-smi` and verify it can detect the GPUs).

**CPU Only Installation**: If your machine does not have a GPU, remove the `--gpus all` from the `docker run` command 
in the GPU installation instructions.

**GPU Installation**:

```
git clone https://github.com/marius-team/marius.git marius_artifact
cd marius_artifact
git checkout eurosys_2023_artifact
cd examples/docker/
docker build -t marius:artifact .
docker run --gpus all -it --ipc=host -v <path to directory containing marius_artifact>:/working_dir/ marius:artifact bash
cd marius_artifact
pip3 install .
python3 experiment_manager/run_experiment.py --experiment setup_dgl
sed -i 's/device=eth/device=th/g' /usr/local/lib/python3.6/dist-packages/dgl/optim/pytorch/sparse_optim.py
```

### Pip Installation ###
Assuming installation of the above dependencies, this artifact can also be installed directly without Docker:

```
git clone https://github.com/marius-team/marius.git marius_artifact
cd marius_artifact
git checkout eurosys_2023_artifact
pip3 install .
python3 experiment_manager/run_experiment.py --experiment setup_dgl
sed -i 's/device=eth/device=th/g' /usr/local/lib/python3.6/dist-packages/dgl/optim/pytorch/sparse_optim.py
```

### Installation Notes ###

The latter two commands in both of the above installation instructions fix a typo in DGL's ```sparse_optim.py``` file. 
Depending on your DGL installation, this file may be located in a different location.

This artifact installation installs the following command line tools:

- marius_preprocess: Built-in dataset downloading and preprocessing
- marius_train: Train GNN models using configuration files and the command line



## How To Use This Artifact ##

MariusGNN can preprocess graphs and train GNNs from the command line by supplying a configuration file which defines 
the training procedure.

For example, the following command will preprocess the FB15k-237 graph.

`marius_preprocess --dataset fb15k_237 datasets/fb15k237/`

Given the preprocessed graph, we can then run MariusGNN according to a configuration file. For example:

CPU: `marius_train experiment_manager/example/configs/marius_gs_mem_cpu.yaml`

GPU: `marius_train experiment_manager/example/configs/marius_gs_mem_gpu.yaml`

**However, instead of directly running the MariusGNN executable on configuration files, for this artifact we have 
provided Python scripts which handle running each experiment (MariusGNN and baselines DGL/PyG). 
These Python scripts are contained in the `experiment_manager` and are described in the following sections.** The
Python scripts automatically pass the corresponding configuration files for each experiment to the corresponding system.



## Artifact Minimal Working Example (Functionality) ##

### Link Prediction ###
We provide a minimal working example which trains a GraphSage GNN on the (small) FB15k-237 graph for the task of link 
prediction using MariusGNN and our two baselines DGL and PyG. MariusGNN training contains two configurations: 
one with the graph stored in memory 
(GPU or CPU memory depending on GPU support) and one with the graph data stored on disk 
(as an example of MariusGNN out-of-core support). DGL and PyG only support in-memory training. Note that we use the 
small FB15k-237 graph to ensure this example is fast to run and doesn't require specialized hardware, but that the paper
experiments report results over significantly larger graphs.

For CPU based training, the minimal working example can be run as follows:
```
python3 experiment_manager/run_experiment.py --experiment fb15k237_mem_cpu --show_output
python3 experiment_manager/run_experiment.py --experiment fb15k237_disk_cpu --show_output
```

For GPU based training, the minimal working example can be run as follows:
```
python3 experiment_manager/run_experiment.py --experiment fb15k237_mem_gpu --show_output
python3 experiment_manager/run_experiment.py --experiment fb15k237_disk_gpu --show_output
```

In both cases, the first script will run MariusGNN, DGL, and PyG with graph data stored in memory (either CPU or GPU). 
The output should contain the results of running each system for five epochs. 
Epoch runtimes and MRR (a measure of accuracy for link prediction)
are reported for all systems. A summary of the five epochs for each system is presented at the end.
The second script runs MariusGNN with graph data stored on disk (again for five epochs). 
Disk-based training support is a key property of MariusGNN which allows for up to 64x cheaper GNN training over
large-scale graphs compared to DGL and PyG.

### Node Classification ###
We also provide a minimal working example for the task of node classification. We train a three layer GraphSage GNN
on the ogbn-arxiv graph. The format for running the experiments is the same as in the above, 
except the `fb15k237` in the experiment name is replaced by `arxiv`:
```
python3 experiment_manager/run_experiment.py --experiment arxiv_mem_cpu --show_output
python3 experiment_manager/run_experiment.py --experiment arxiv_disk_cpu --show_output

python3 experiment_manager/run_experiment.py --experiment arxiv_mem_gpu --show_output
python3 experiment_manager/run_experiment.py --experiment arxiv_disk_gpu --show_output
```



## Artifact Documentation: Running Experiments ##
Here we provide documentation with respect to using and running the experiment manager. 
**For documentation regarding the MariusGNN source code contained in this artifact see the DOCS.md file.**

This artifact is organized as follows:

```
datasets/                       // directory created to download and store preprocessed datasets

examples/docker/                // contains the example Dockerfile for installation

experiment_manager/             // suit of python scripts for running paper experiments and baseline systems
   baselines/
      dgl/                      // contains the code for running GNNs using the DGL baseline
      pyg/                      // contains the code for running GNNs using the PyG baseline
   disk/                        // contains python scripts and configs for out-of-core microbenchmarks (Table 7, Figure 8)
   example/                     // contains the python scripts and configs for the minimal working example
   sampling/                    // contains the python scripts and configs for neighborhood sampling experiments (Table 6)
   setup_dgl/                   // contains a single experiment to import dgl and allow for typo fixing (used during install)
   system_comparisons/          // contains the python scripts and configs for the system comparisons (Tables 3-5, Figure 7)
   
   executor.py                  // executes training for each system and starts/stop results collection
   parsing.py                   // parses the output of each system, dstat, and nvidia-smi
   reporting.py                 // prints experiment summaries
   run_experiment.py            // entry script which runs each experiment with desired flags
   tracing.py                   // runs dstat and nvidia-smi during experiments for tracing

results/                        // directory created to store experiment results and outpur files

src/                            // MariusGNN artifact source code
```


To reproduce the experiments we have provided Python scripts to run each experiment in the paper. 
**Experiments are run from the repository root directory with**:

`python3 experiment_manager/run_experiment.py --experiment <experiment_name>`

To change which experiment is run simply change the `<experiment_name>`. Below we provide a table with the experiment
name for each result reported in the paper as well as the machine needed and any additional notes. 
**Note that re-running the experiments from the paper can take many hours and require access to AWS P3 GPU 
machines leading to significant monetary cost**. We report the estimated cost to reproduce each experiment in 
the table below.

Experiments cannot be run in parallel on the same machine and must be run one at a time. 
This is because MariusGNN utilizes the same paths to store intermediate program data across experiments. 

By running experiments with the above command, results will be output to the terminal and in the corresponding 
`results/` directory for the experiment. See the artifact structure section (above) for the locations of the 
experiment directories.

### Experiment Runner Flags ###
The following are additional flags for the `experiment_manager/run_experiment.py` script 
(in addition to `--experiment`):

`--overwrite`: Will overwrite previous experiment results. Can be used in case the experiment results get in an 
inconsistent state.

`--show_output`: By default, the output of each program is redirected to a file and stored in the experiment `results/` 
directory. This flag will show the output in stdout AND redirect the output. 
Useful for monitoring the experiment, but may print out a lot of info. 

`--num_runs`: Number of times to run each experiment.

`--enable_dstat`: Will run dstat tracing for the experiment.

`--enable_nvidia_smi`: Will run nvidia-smi tracing for the experiment.



## Reproducing Experimental Results ##
[comment]: <> (| Experiment | Corresponding Figure | Est. Runtime | Est. Runtime with `--short` |)

[comment]: <> (| --- | ----------- | -------- | ---------- |)

[comment]: <> (| fb15k | Table 2 | 20 mins | - |)

[comment]: <> (| livejournal | Table 3 | 2 hours | - |)

[comment]: <> (| twitter | Table 4 | 10 hours | - |)

[comment]: <> (| freebase86m | Table 5 | 12 hours | - |)

[comment]: <> (| buffer_simulator | Figure 7 | ~ 1 minute | - |)

[comment]: <> (| utilization | Figure 8 | 2 hours | - |)

[comment]: <> (| orderings_total_io | Figure 9 | 2 hours | - |)

[comment]: <> (| orderings_freebase86m | Figure 10 | ~1 day | 2 hours |)

[comment]: <> (| orderings_twitter | Figure 11 | ~2 days | 4 hours |)

[comment]: <> (| staleness_bound | Figure 12 | ~2 days | 8 hours |)

[comment]: <> (| prefetching | Figure 13 | 30 mins | - |)

[comment]: <> (| big_embeddings | Table 6 | 5 hours | - |)



## Hit An Issue? ##
If you have hit an issue with the system, the scripts, or the results, please let us know 
(contact: waleffe@wisc.edu, or open an issue) and we will investigate and fix the issue if needed.



[comment]: <> (## Citing MariusGNN ##)

