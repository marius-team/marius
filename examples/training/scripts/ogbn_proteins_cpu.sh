# preprocess the ogbn_proteins graph and put preprocessed graph into output dir
python3 tools/preprocess.py ogbn_proteins output_dir/ 

# run marius on the preprocessed input
build/marius_train examples/training/configs/ogbn_proteins_cpu.ini info