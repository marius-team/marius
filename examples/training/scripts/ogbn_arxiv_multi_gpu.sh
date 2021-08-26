# preprocess the ogbn_arxiv graph and put preprocessed graph into output dir
marius_preprocess output_dir/ --dataset ogbn_arxiv

# run marius on the preprocessed input
marius_train examples/training/configs/ogbn_arxiv_multi_gpu.ini