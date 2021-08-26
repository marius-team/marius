# preprocess the ogbn_proteins graph and put preprocessed graph into output dir
marius_preprocess output_dir/ --dataset ogbn_proteins

# run marius on the preprocessed input
marius_train examples/training/configs/ogbn_proteins_cpu.ini