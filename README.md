# OSDI 21' Artifact Evaluation #

## Getting Started ##

### How to use Marius ### 

Marius is run from the command line by supplying a .ini format configuration file which defines the training procedure. 

For example, the following command will train marius on the configuration used to obtain the ComplEx entry in Table 2 of the paper.

`marius_train osdi2021/system_comparisons/fb15k/marius/complex.ini`

However, instead of directly running the Marius executable on configuration files, we have provided a python script which handles running each experiment, collecting results, and plotting results. This is described in the following section.

### Reproducing experiments ###

To reproduce the experiments we have provided python scripts to run each experiment in the paper. **Nearly all experiments can be run on a single Amazon EC2 p3.2xlarge Ubuntu 18.04 instance.** The few exceptions to this are described in the detailed instructions. Also please refer to to [Requirements](#requirements) below for software packages and dependencies required. 

Experiments are run from the repository root directory with `python3 osdi2021/run_experiment.py --experiment <experiment_name>`

**Experiments cannot be run in parallel on the same machine and must be run one at a time. This is because Marius utilizes the same paths to store intermediate program data accross experiments.** 

By running experiments with this command, results and generated plots will be output to the terminal and in the corresponding directory for the experiment. See the artifact structure section for the locations of the experiment directories. For example, the above command will output results to `osdi2021/system_comparisons/fb15k`.

The list of experiments and their corresponding figures/tables are:


| Experiment | Corresponding Figure | Est. Runtime | Est. Runtime with `--short` |
| --- | ----------- | -------- | ---------- |
| fb15k | Table 2 | 20 mins | - |
| livejournal | Table 3 | 2 hours | - |
| twitter | Table 4 | 10 hours | - |
| freebase86m | Table 5 | 12 hours | - |
| buffer_simulator | Figure 7 | ~ 1 minute | - |
| utilization | Figure 8 | 2 hours | - |
| orderings_total_io | Figure 9 | 2 hours | - |
| orderings_freebase86m | Figure 10 | ~1 day | 2 hours |
| orderings_twitter | Figure 11 | ~2 days | 4 hours |
| staleness_bound | Figure 12 | ~2 days | 8 hours |
| prefetching | Figure 13 | 30 mins | - |
| big_embeddings | Table 6 | 5 hours | - |

#### Experiment runner flags ####

The following are flags for the `osdi2021/run_experiment.py` script

`--overwrite`: Will overwrite previous experiment results. Can be used in case the experiment results get in an inconsistent state.

`--show_output`: By default output of each program is redirected to a file and stored in the experiment directory. This flag will show the output in stdout AND redirect the output. Useful for checking how far you are in the experiment, but will print out a lot of info. 

`--short`: For the long experiments: orderings_freebase86m, orderings_twitter, and staleness_bound, this will run a shortened version of them. It has no effect for the other experiments.

Using short will have the following effects:
- orderings_freebase86m configurations will be trained to 1 epoch instead of 10 epochs 
- orderings_twitter configurations will be trained to 1 epoch instead of 10 epochs
- staleness_bound configurations will be trained to 3 epochs instead of 10 epochs and only staleness bounds of [1, 4, 16, 64, 256] will be evaluated.

`--collect_tracing_metrics`: Will run dstat and nvidia-smi tracing for the experiment. The following experiments have this enabled by default: utilization, orderings_total_io. The rest have it disable. 

### Hit an issue? ###

If you have hit an issue with the system, the scripts, or the results, please let us know (contact: mohoney2@wisc.edu) and we will investigate and fix the issue if needed.


## Detailed Instructions ##

### Artifact claims ###

Since submitting the paper, we have made changes to the system to get it ready for open sourcing. The changes involved a full refactor of the system to simplify its design and usage. These changes have also resulted in fixing of some bugs that are relevant to the paper's results. The code in this repository is the most up to date version and will be used to rerun experiments for the final version of the paper. 

#### Accuracy differences in Table 5 ####
In the original submission of the paper we had differences in accuracy between PyTorch Big-Graph and Marius when running on Freebase86m, where the systems reached a peak MRR of .73 and ~.685 respectively. This issue was known by us and also pointed out by the reviewers and we aimed to fix it for the final submission of the paper. 

As a result of the full refactor, the accuracy differences are no longer as large. With Marius now achieving a peak MRR of .728 on the same configuration.

#### Accuracy differences in Table 4 ####
We also had differences in accuracy between PyTorch Big-Graph and Marius when running on Twitter, where the systems reached a peak MRR of .313 and .383 respectively. 

After the refactor Marius achieves an accuracy of about .310.

#### Updates to this artifact ####
While the submission of this artifact for evaluation points to a specific commit, we will keep this branch up to date with our latest experiment configuration, scripts and plotting code. Such that for the final version of the paper, all produced results will correspond exactly to the scripts and configuration in the most up to date version of this branch. 

Additionally, if the reviewers of this artifact encounter any issues/bugs, please contact us and we will fix the issue and push an update to the branch.

#### Experiments that require extra effort to reproduce ####
Some experiments will not run on the single P3.2xLarge instance due to memory requirements and need to be run on a machine with a large amount of memory.

Experiments that have large memory requirements: 
- DGL-KE Twitter
  - Will run into an out of memory error on a P3.2xLarge machine, requires > 64 GB of CPU memory to run.
- DGL-KE Freebase86m
  - Will run into an out of memory error on a P3.2xLarge machine, requires > 64 GB of CPU memory to run.
- Evaluating D=400 sized embeddings on Marius for Table 6
  - The embeddings for this configuration can be trained with limited memory, but to evaluate the MRR of the embeddings requires that the embeddings fit in memory. For the paper, we transferred the embeddings trained on the P3.2xLarge machine to a machine with 500 GB of memory for evaluation.
- Evaluating D=800 sized embeddings on Marius for Table 6
  - Same as above
   
If the reviewers of this artifact wish to reproduce these experiments, we will be happy to provide a machine which can accommodate these experiments.


### Artifact structure ###

Experiment scripts and results are located in the `osdi2021` directory. The directory structure is as follows:
```
osdi2021/
   preprocessors/
      pbg/
      dgl-ke/ 
   buffer_simulator/                         // contains the code for the buffer simulator to reproduce figure 7.
   large_embeddings/                         // configuration and results for Table 6
      cpu_memory.ini                            // d=50 
      disk.ini                                  // d=100 
      gpu_memory.ini                            // d=20
      large_scale.ini                           // d=400 and d=800
   microbenchmarks/                          // configuration and results for figures 12 and 13
      bounded_staleness/                        // configuration and results for figure 12 
         all_async.ini                             // async relations           
         all_sync.ini                              // staleness bound = 1
         sync_relations_async_nodes.ini            // sync relations 
      prefetching/                              // configuration and results for figure 13
         no_prefetching.ini                        // prefetching off
         prefetching.ini                           // prefetching on
   partition_orderings/                      // configuration and results for figures 9, 10 and 11
      freebase86m/                              // figures 9 and 10
         elimination.ini                           // elimination ordering d=50 and d=100
         hilbert.ini                               // hilbert ordering d=50 and d=100
         hilbert_symmetric.ini                     // hilbert symmetric ordering d=50 and d=100
         memory.ini                                // in memory configuration d=50
      twitter/                                  // figure 11
         elimination.ini                           // elimination ordering d=100 and d=200
         hilbert.ini                               // hilbert ordering d=100 and d=200
         hilbert_symmetric.ini                     // hilbert symmetric ordering d=100 and d=200
         memory.ini                                // in memory configuration d=100
   system_comparisons/                       // configuration and results for tables 2, 3, 4, 5 and figure 8
      fb15k/                                    // table 2
         dgl-ke/                                   // commands to run dgl-ke   
            complex.txt                               // complex cmd
            distmult.txt                              // distmult cmd
         marius/                                   // configuration files for marius
            complex.ini                               // complex config 
            distmult.ini                              // distmult config
         pbg/                                      // configuration files for pbg
            fb15k_complex_config.py                   // complex config 
            fb15k_distmult_config.py                  // distmult config
      freebase86m/                              // table 5 and figure 8
         dgl-ke/                                   // commands to run dgl-ke
            d50.txt                                   // d=50 for figure 8
         marius/                                   // configuration files for marius
            d50.ini                                   // d=50 in memory for figure 8
            d50_8.ini                                 // d=50 on disk for figure 8
            d100.ini                                  // d=100 for table 5
         pbg/                                      // configuration files for pbg
            d50_p8.py                                 // d=50 on disk for figure 8
            d100_p16.py                               // d=100 on disk for table 5 
      livejournal/                              // table 3
         dgl-ke/                                   // command to run dgl-ke
            dot.txt                                   // dot cmd
         marius/                                   // configuration file for marius
            dot.ini                                   // dot config
         pbg/                                      // configuration file for marius
            dot.py                                    // dot config
      twitter/                                  // table 4
         dgl-ke/                                   // command to run dgl-ke
            dot.txt                                   // dot cmd
         marius/                                   // configuration file for marius
            dot.ini                                   // dot config
         pbg/                                      // configuration file for marius
            dot.py                                    // dot config
   execute.py                                // executes training for each system and starts/stop results collection
   parse_output.py                           // parses the output of each systems output and dstat and nvidia-smi
   run_experiment.py                         // run each experiment 
   utils.py                                  // handles special routines for preprocessing datasets for other systems
   plotting.py                               // produces plots after experiments have been run 
   osdi21-paper143.pdf                       // the orignal submission of the paper
```

## Build Information and Environment ##

For the artifact reviewers, we will set up an AWS P3.2xLarge instance with the system pre-built and provide access to them. But if you wishe to build the system from scratch, the following instructions denote dependencies and how to build Marius.

### Requirements ###
* Ubuntu 18.04
* CUDA 10.1 or 10.2 
* CuDNN 7
* 1.7 >= pytorch 
* python >= 3.6
* pip >= 21
* GCC >= 9
* cmake >= 3.12
* make >= 3.8
* dstat 

Please also see instructions below.

### Installation ###

1. Install latest version of PyTorch for your CUDA version:

    - CUDA 10.1: `python3 -m pip install torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html`
    - CUDA 10.2: `python3 -m pip install torch==1.7.1`

2. Clone the repository `git clone https://github.com/marius-team/marius.git`

3. Checkout the artifact evaluation branch `cd marius; git checkout osdi2021`

4. Build and install Marius `python3 -m pip install .`

5. Download and build PyTorch-BigGraph `git clone https://github.com/facebookresearch/PyTorch-BigGraph.git; cd PyTorch-BigGraph; export PBG_INSTALL_CPP=1; python3 -m pip install .; cd ..`

#### Full script (without torch install) ####
```
git clone https://github.com/marius-team/marius.git
cd marius
git checkout osdi2021
python3 -m pip install -r requirements.txt
python3 -m pip install .
git clone https://github.com/facebookresearch/PyTorch-BigGraph.git
export PBG_INSTALL_CPP=1
cd PyTorch-BigGraph 
python3 -m pip install .
cd ..
```

