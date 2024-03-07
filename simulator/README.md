# Simulator

Before running the simulator, install the required dependencies using the command:
```
$ pip3 install -r requirements.txt
```

The following directory contains code to simulate GNN inference. The main file in this directory is `main.py` which takes in the arguments of:
```
usage: main.py [-h] [--config_file CONFIG_FILE] --save_path SAVE_PATH --graph_title GRAPH_TITLE [--log_rate LOG_RATE]

optional arguments:
  -h, --help            show this help message and exit
  --config_file CONFIG_FILE
                        The config file containing the details for the simulation (default: None)
  --save_path SAVE_PATH
                        The path to save the resulting image to (default: None)
  --graph_title GRAPH_TITLE
                        The title of the saved graph (default: None)
  --log_rate LOG_RATE   Log rate of the nodes processed (default: 20)
```

Thus, to run the simulator on the archive dataset with 1-hop sampling, you can run the command:
```
$ python3 main.py --config_file configs/arvix_linear.json --save_path results/arvix_linear.png --graph_title "CDF for sequentially arranged features"
```