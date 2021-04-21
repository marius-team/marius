# preprocess the dbpedia_50 graph and put preprocessed graph into output dir
marius_preprocess dbpedia_50 output_dir/

# run marius on the preprocessed input
marius_train examples/training/configs/dbpedia_50_multi_gpu.ini info