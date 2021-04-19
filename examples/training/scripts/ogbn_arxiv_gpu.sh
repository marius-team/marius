# preprocess the ogbn_arxiv graph and put preprocessed graph into output dir
marius_preprocess ogbn_arxiv output_dir/

# run marius on the preprocessed input
marius_train examples/training/configs/ogbn_arxiv_gpu.ini info