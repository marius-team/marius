from pathlib import Path

def output_config(stats, num_nodes, num_relations, output_dir, name, device = "cpu"):
    if device == "gpu":
        gpu(stats, num_nodes, num_relations, output_dir, name)
    elif device == "cpu":
        cpu(stats, num_nodes, num_relations, output_dir, name)
    else:
        multi_gpu(stats, num_nodes, num_relations, output_dir, name)
    
def cpu(stats, num_nodes, num_relations, output_dir, name):  
    file = Path(output_dir) / Path(str(name) + "_cpu.ini")
    with open(file, "w+") as f:
        f.write("[general]\n")
        f.write("device=CPU\n\
random_seed=0\n\
num_train=" + str(int(stats[0])) + "\n\
num_nodes=" + str(int(num_nodes)) + "\n\
num_relations=" + str(int(num_relations)) + "\n\
num_valid=" + str(int(stats[1])) + "\n\
num_test=" + str(int(stats[2])) + "\n\n")
        
        f.write("\
[model]\n\
embedding_size=100\n\
decoder=ComplEx\n\n")

        f.write("\
[storage]\n\
edges_backend=HostMemory\n\
embeddings_backend=HostMemory\n\
relations_backend=HostMemory\n\n")

        f.write("\
[training]\n\
batch_size=1000\n\
number_of_chunks=16\n\
negatives=512\n\
degree_fraction=.5\n\
num_epochs=50\n\
learning_rate=.1\n\
regularization_coef=0\n\
regularization_norm=2\n\
synchronous=false\n\
shuffle_interval=1\n\n")
        
        f.write("\
[training_pipeline]\n\
max_batches_in_flight=16\n\
num_embedding_loader_threads=4\n\
num_embedding_transfer_threads=2\n\
num_compute_threads=4\n\
num_gradient_transfer_threads=2\n\
num_embedding_update_threads=4\n\n")
        
        f.write("\
[evaluation_pipeline]\n\
max_batches_in_flight=16\n\
num_embedding_loader_threads=2\n\
num_embedding_transfer_threads=2\n\
num_evaluate_threads=4\n\n")
        
        f.write("\
[evaluation]\n\
epochs_per_eval=1\n\
batch_size=1000\n\
number_of_chunks=1\n\
negatives=1000\n\
degree_fraction=0\n\
negative_sampling_access=Uniform\n\
evaluation_method=LinkPrediction\n\
filtered_evaluation=false\n\n")
        
        f.write("\
[path]\n\
base_directory=training_data/\n\
train_edges=output_dir/train_edges.pt\n\
validation_edges=output_dir/valid_edges.pt\n\
test_edges=output_dir/test_edges.pt\n\n")
        
        f.write("\
[reporting] \n\
log_level=info\n\n")


def gpu(stats, num_nodes, num_relations, output_dir, name):
    file = Path(output_dir) / Path(str(name) + "_gpu.ini")
    with open(file, "w+") as f:
        f.write("[general]\n\
device=GPU\n\
random_seed=0\n\
num_train=" + str(int(stats[0])) + "\n\
num_nodes=" + str(int(num_nodes)) + "\n\
num_relations=" + str(int(num_relations)) + "\n\
num_valid=" + str(int(stats[1])) + "\n\
num_test=" + str(int(stats[2])) + "\n\n\
[model]\n\
embedding_size=100\n\
decoder=ComplEx\n\n\
[storage]\n\
edges_backend=DeviceMemory\n\
embeddings_backend=DeviceMemory\n\
relations_backend=DeviceMemory\n\n\
[training]\n\
batch_size=10000\n\
number_of_chunks=16\n\
negatives=512\n\
degree_fraction=0\n\
num_epochs=5\n\
learning_rate=.1\n\
regularization_coef=0\n\
regularization_norm=2\n\
synchronous=true\n\
shuffle_interval=1\n\n\
[training_pipeline]\n\
max_batches_in_flight=16\n\
num_embedding_loader_threads=4\n\
num_embedding_transfer_threads=2\n\
num_compute_threads=1\n\
num_gradient_transfer_threads=2\n\
num_embedding_update_threads=4\n\n\
[evaluation_pipeline]\n\
max_batches_in_flight=16\n\
num_embedding_loader_threads=2\n\
num_embedding_transfer_threads=2\n\
num_evaluate_threads=1\n\n\
[evaluation]\n\
epochs_per_eval=1\n\
batch_size=1000\n\
number_of_chunks=1\n\
negatives=1000\n\
degree_fraction=0\n\
negative_sampling_access=Uniform\n\
evaluation_method=LinkPrediction\n\
filtered_evaluation=false\n\n\
[path]\n\
base_directory=training_data/\n\
train_edges=output_dir/train_edges.pt\n\
validation_edges=output_dir/valid_edges.pt\n\
test_edges=output_dir/test_edges.pt\n\n\
[reporting]\n\
log_level=info")

def multi_gpu(stats, num_nodes, num_relations, output_dir, name):
    file = Path(output_dir) / Path(str(name) + "_multi_gpu.ini")
    with open(file, "w+") as f:
        f.write("[general]\n\
scale_factor=.001\n\
embedding_size=100\n\
device=GPU\n\
gpu_ids=0 1\n\
comparator_type=Dot\n\
relation_type=ComplexHadamard\n\
random_seed=0\n\
num_train=" + str(int(stats[0])) + "\n\
num_nodes=" + str(int(num_nodes)) + "\n\
num_relations=" + str(int(num_relations)) + "\n\
num_valid=" + str(int(stats[1])) + "\n\
num_test=" + str(int(stats[2])) + "\n\n\
[storage]\n\
edges_backend=HostMemory\n\
embeddings_backend=HostMemory\n\
relations_backend=HostMemory\n\n\
[training]\n\
batch_size=10000\n\
number_of_chunks=10\n\
negatives=256\n\
degree_fraction=256\n\
num_epochs=5\n\
optimizer_type=Adagrad\n\
loss=SoftMax\n\
epsilon=1e-8\n\
learning_rate=.1\n\
negative_sampling_access=Uniform\n\
negative_sampling_policy=DegreeBased\n\
edge_bucket_ordering=Shuffle\n\
synchronous=false\n\
shuffle_epochs=1\n\n\
[training_pipeline]\n\
max_batches_in_flight=8\n\
update_in_flight=false\n\
embeddings_host_queue_size=4\n\
embeddings_device_queue_size=4\n\
gradients_host_queue_size=4\n\
gradients_device_queue_size=4\n\
num_embedding_loader_threads=2\n\
num_embedding_transfer_threads=1\n\
num_compute_threads=2\n\
num_gradient_transfer_threads=1\n\
num_embedding_update_threads=2\n\n\
[evaluation_pipeline]\n\
max_batches_in_flight=16\n\
embeddings_host_queue_size=4\n\
embeddings_device_queue_size=4\n\
num_embedding_loader_threads=4\n\
num_embedding_transfer_threads=1\n\
num_evaluate_threads=2\n\n\
[evaluation]\n\
epochs_per_eval=1\n\
batch_size=1000\n\
number_of_chunks=1\n\
negatives=1000\n\
degree_fraction=1000\n\
valid_fraction=0\n\
test_fraction=0\n\
negative_sampling_access=Uniform\n\
negative_sampling_policy=DegreeBased\n\
evaluation_method=MRR\n\n\
[path]\n\
base_directory=training_data/\n\
train_edges=output_dir/train_edges.pt\n\
validation_edges=output_dir/valid_edges.pt\n\
test_edges=output_dir/test_edges.pt")


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
if __name__=="__main__":
    print("This is a configuration file generator.")
    ds_name = [
        "fb15k",
        "live_journal",
        "freebase86m",
        "wn18",
        "fb15k_237",
        "wn18rr",
        "codex_s",
        "codex_m",
        "codex_l",
        "drkg",
        "hetionet",
        "kinships",
        "openbiolink_hq",
        "openbiolink_lq",
        "ogbl_biokg",
        "ogbl_ppa",
        "ogbl_ddi",
        "ogbl_collab",
        "ogbn_arxiv",
        "ogbn_proteins",
        "ogbn_products",
    ]

    for n in ds_name:
        output_bash_cmds("./examples/training/scripts/", n)