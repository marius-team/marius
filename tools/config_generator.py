from pathlib import Path
import argparse

def output_config(device_dict, output_dir):
    device = device_dict.get("device")
    ds_name = device_dict.get("dataset")

    if device == "GPU":
        opts = readOpts("./tools/cpu_default_config.txt")
        gpu(device_dict, opts, ds_name, output_dir)
    elif device == "CPU":
        opts = readOpts("./tools/gpu_default_config.txt")
        cpu(device_dict, opts, ds_name, output_dir)
    else:
        opts = readOpts("./tools/mult_gpu_default_config.txt")
        multi_gpu(device_dict, opts, ds_name, output_dir)

    
def readOpts(file):
    with open(file, "r") as f:
        lines = f.readlines()
    
    opts = []
    for l in lines:
        l = l.split("=")[0].split(".")
        opts.append(l)

    return opts

def cpu(d, opts, ds_name, output_dir):  
    file = Path(output_dir) / Path(str(ds_name) + "_cpu.ini")
    modules = ["model", "storage", "training", "training_pipeline", 
            "evaluation_pipeline", "evaluation", "path", "reporting"]

    with open(file, "w+") as f:
        f.write("[general]\n")
        f.write("device=CPU\n")
        f.write("random_seed=" + d.get("general.random_seed") + "\n")
        f.write("num_train=" + d.get("num_train") + "\n")
        f.write("num_nodes=" + d.get("num_nodes") + "\n")
        f.write("num_relations=" + d.get("num_relations") + "\n")
        f.write("num_valid=" + d.get("num_valid") + "\n")
        f.write("num_test=" + d.get("num_test") + "\n")

        for m in modules:
            f.write("\n[" + m + "]\n")
            for opt in opts:
                if opt[0] == m:
                    f.write(opt[1] + "=" + d.get(".".join(opt)) + "\n")


def gpu(d, opts, ds_name, output_dir):
    file = Path(output_dir) / Path(str(ds_name) + "_gpu.ini")
    modules = ["model", "storage", "training", "training_pipeline",
                "evaluation_pipeline", "evaluation", "path", "reporting"]
    with open(file, "w+") as f:
        f.write("[general]\n")
        f.write("device=GPU\n")
        f.write("random_seed=" + d.get("general.random_seed") + "\n")
        f.write("num_train=" + d.get("num_train") + "\n")
        f.write("num_nodes=" + d.get("num_nodes") + "\n")
        f.write("num_relations=" + d.get("num_relations") + "\n")
        f.write("num_valid=" + d.get("num_valid") + "\n")
        f.write("num_test=" + d.get("num_test") + "\n")
    
        for m in modules:
            f.write("\n[" + m + "]\n")
            for opt in opts:
                if opt[0] == m:
                    f.write(opt[1] + "=" + d.get(".".join(opt)) + "\n")


def multi_gpu(d, opts, ds_name, output_dir):
    file = Path(output_dir) / Path(str(ds_name) + "_multi_gpu.ini")
    modules = ["storage", "training", "training_pipeline", "evaluation_pipeline",
                "evaluation", "path"]

    with open(file, "w+") as f:
        f.write("[general]\n")
        f.write("scale_factor=" + d.get("general.scale_factor") + "\n")
        f.write("embedding_size=" + d.get("general.embedding_size") + "\n")
        f.write("device=GPU\n")
        f.write("gpu_ids=" + d.get("general.gpu_ids") + "\n")
        f.write("comparator_type=" + d.get("general.comparator_type") + "\n")
        f.write("relation_type=" + d.get("general.relation_type") + "\n")
        f.write("random_seed=" + d.get("general.random_seed") + "\n")
        f.write("num_train=" + d.get("num_train") + "\n")
        f.write("num_nodes=" + d.get("num_nodes") + "\n")
        f.write("num_relations=" + d.get("num_relations") + "\n")
        f.write("num_valid=" + d.get("num_valid") + "\n")
        f.write("num_test=" + d.get("num_test") + "\n")

        for m in modules:
            f.write("\n[" + m + "]\n")
            for opt in opts:
                if opt[0] == m:
                    f.write(opt[1] + "=" + d.get(".".join(opt)) + "\n")



def output_bash_cmds(output_dir, dataset_name):
    cpu_file = Path(output_dir) / Path(dataset_name + "_cpu.sh")
    gpu_file = Path(output_dir) / Path(dataset_name + "_gpu.sh")
    mgpu_file = Path(output_dir) / Path(dataset_name + "_multi_gpu.sh")
    with open(cpu_file, "w+") as f:
        f.write("# preprocess the " + dataset_name + " graph and put preprocessed graph into output dir\n")
        f.write("python3 tools/preprocess.py " +  dataset_name + " output_dir/ \n\n")
        f.write("# run marius on the preprocessed input\n")
        f.write("build/marius_train examples/training/configs/" + dataset_name + "_cpu.ini info")
    with open(gpu_file, "w+") as f:
        f.write("# preprocess the " + dataset_name + " graph and put preprocessed graph into output dir\n")
        f.write("python3 tools/preprocess.py " +  dataset_name + " output_dir/ \n\n")
        f.write("# run marius on the preprocessed input\n")
        f.write("build/marius_train examples/training/configs/" + dataset_name + "_gpu.ini info")
    with open(mgpu_file, "w+") as f:
        f.write("# preprocess the " + dataset_name + " graph and put preprocessed graph into output dir\n")
        f.write("python3 tools/preprocess.py " +  dataset_name + " output_dir/ \n\n")
        f.write("# run marius on the preprocessed input\n")
        f.write("build/marius_train examples/training/configs/" + dataset_name + "_multi_gpu.ini info")

def readTemplate(file):
    with open(file, "r") as f:
        lines = f.readlines()

    keys = []   
    values = []
    for l in lines:
        l = l.split("=")
        l[1] = l[1].rstrip()
        keys.append(l[0])
        values.append(l[1])

    d = dict(zip(keys, values))
    
    return d                       

def updateParam(devices, dicts, args, opts, arg_dict):
    device_idx = -1
    if args.generate_config == None:
        for opt in opts:
            if arg_dict.get(opt) != None:
                raise RuntimeError(
                    "Please specify --generate_config when specifying generating options"
                )

    for i in range(3):
        if (args.generate_config == devices[i]):
            device_idx = i
            for opt in opts:
                if arg_dict.get(opt) != None:
                    if dicts[i].get(opt) != None:                    
                        dicts[i].update({opt: arg_dict.get(opt)})
                    else:
                        raise RuntimeError(
                            str("Unmatching parameter for " + devices[i] + " config: " + opt))

    return device_idx, dicts

if __name__=="__main__":
    print("This is a config generator.")