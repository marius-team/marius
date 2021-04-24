# preprocess the ogbn_proteins graph and put preprocessed graph into output dir
marius_preprocess ogbn_proteins output_dir/

# run marius on the preprocessed input
marius_train examples/training/configs/ogbn_proteins_cpu.ini info