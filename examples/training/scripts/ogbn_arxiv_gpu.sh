# preprocess the ogbn_arxiv graph and put preprocessed graph into output dir
python3 tools/preprocess.py ogbn_arxiv output_dir/ 

# run marius on the preprocessed input
build/marius_train examples/training/configs/ogbn_arxiv_gpu.ini info