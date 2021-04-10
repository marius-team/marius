from pathlib import Path
import argparse

DEFAULT_CONFIG_FILE = "./tools/config_templates/default_configs.txt"


def output_config(config_dict, output_dir):
    device = config_dict.get("device")
    ds_name = config_dict.get("dataset")

    file = Path(output_dir) / Path(str(ds_name) + "_" +
                                   device.lower() + ".ini")
    sections = ["model", "storage", "training", "training_pipeline",
                "evaluation", "evaluation_pipeline", "path", "reporting"]
    opts = list(config_dict.keys())

    with open(file, "w+") as f:
        f.write("[general]\n")
        f.write("device=" + config_dict.get("general.device") + "\n")
        f.write("gpu_ids=" + config_dict.get("general.gpu_ids") + "\n")
        f.write("num_train=" + config_dict.get("num_train") + "\n")
        f.write("num_nodes=" + config_dict.get("num_nodes") + "\n")
        f.write("num_relations=" + config_dict.get("num_relations") + "\n")
        f.write("num_valid=" + config_dict.get("num_valid") + "\n")
        f.write("num_test=" + config_dict.get("num_test") + "\n")
        f.write("experiment_name=" +
                config_dict.get("general.experiment_name") + "\n")

        for sec in sections:
            f.write("\n[" + sec + "]\n")
            for key in opts:
                if key.split(".")[0] == sec:
                    f.write(key.split(".")[1] +
                            "=" + config_dict.get(key) + "\n")


def read_template(file):
    with open(file, "r") as f:
        lines = f.readlines()

    keys = []
    values = []
    valid_dict = {}
    for line in lines:
        line = line.split("=")
        line[1] = line[1].rstrip()
        keys.append(line[0])
        sub_line = line[1].split("*")
        values.append(sub_line[0])
        if len(sub_line) > 1:
            valid_dict.update({line[0]: [sub_line[1:]]})
    config_dict = dict(zip(keys, values))

    return config_dict, valid_dict


def update_param(config_dict, arg_dict):
    if arg_dict.get("generate_config") is None:
        for key in config_dict:
            if arg_dict.get(key) is not None:
                raise RuntimeError(
                    "Please specify --generate_config when " +
                    "specifying generating options"
                )
    else:
        if arg_dict.get("generate_config") is None:
            config_dict.update({"device": "GPU"})
            config_dict.update({"general.device": "GPU"})
        elif arg_dict.get("generate_config") == "multi-GPU":
            config_dict.update({"device": "multi_GPU"})
            config_dict.update({"general.device": "multi-GPU"})
        else:
            config_dict.update({"general.device":
                                arg_dict.get("generate_config")})
            config_dict.update({"device":
                                arg_dict.get("generate_config")})

        for key in config_dict.keys():
            if arg_dict.get(key) is not None:
                config_dict.update({key: arg_dict.get(key)})

    if config_dict.get("general.random_seed") == "*":
        del config_dict["general.random_seed"]

    return config_dict


if __name__ == "__main__":
    print("This is a config generator.")
