# OSDI 21' Artifact Evaluation #

## Getting Started ##

Experiment configuration, scripts and a copy of the paper are located in the `osdi2021` directory. The directory is structured in the following format.


### How Marius works ### 

Marius is run from the command line by supplying a .ini format configuration file which defines the training procedure. 

For example, the following command will train marius on the configuration used to obtain the ComplEx entry in Table 2.

`marius_train osdi2021/system_comparisons/fb15k/marius/complex.ini`

### Reproducing experiments ###

To reproduce the experiments we have provided python scripts to run each experiment in the paper.

Experiments are run with `python3 osdi2021/run_experiment.py --experiment <experiment_name>`

The list of experiments and their corresponding figures/tables are:
```
fb15k                      // Table 2
livejournal                // Table 3
twitter                    // Table 4
freebase86m                // Table 5
buffer_simulator           // Figure 7
utilization                // Figure 8 
ordering_total_io          // Figure 9 
orderings_freebase86m      // Figure 10 
orderings_twitter          // Figure 11
staleness_bound            // Figure 12
prefatching                // Figure 13
big_embeddings             // Table 6
all                        // Will run all of the above
```

### Reproducing plots ###

### Differences from the original version ###

## Detailed Instructions ##

### Artifact claims ###



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
```


## Requirements ##
(Other versions may work, but are untested)
* Ubuntu 18.04 or MacOS 10.15 
* CUDA 10.1 or 10.2 (If using GPU training)
* CuDNN 7 (If using GPU training)
* 1.7 >= pytorch 
* python >= 3.6
* pip >= 21
* GCC >= 9 (On Linux) or Clang 12.0 (On MacOS)
* cmake >= 3.12
* make >= 3.8


## Installation ##

1. Install latest version of PyTorch for your CUDA version:

    Linux:
    - CUDA 10.1: `python3 -m pip install torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html`
    - CUDA 10.2: `python3 -m pip install torch==1.7.1`
    - CPU Only: `python3 -m pip install torch==1.7.1+cpu -f https://download.pytorch.org/whl/torch_stable.html`

    MacOS:
    - CPU Only: `python3 -m pip install torch==1.7.1`

2. Clone the repository `git clone https://github.com/marius-team/marius.git`

3. Install dependencies `cd marius; python3 -m pip install -r requirements.txt`

4. Create build directory `mkdir build; cd build`

5. Run cmake in the build directory `cmake ../` (CPU-only build) or `cmake ../ -DUSE_CUDA=1` (GPU build)

6. Make the marius executable. `make marius_train -j`

#### Full script (without torch install) ####

```
git clone https://github.com/marius-team/marius.git
cd marius
python3 -m pip install -r requirements.txt
mkdir build
cd build
cmake ../ -DUSE_CUDA=1
make marius_train -j
```

