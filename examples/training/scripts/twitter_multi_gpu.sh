# preprocess the twitter graph and put preprocessed graph into output dir
marius_preprocess twitter output_dir/

# run marius on the preprocessed input
marius_train examples/training/configs/twitter_multi_gpu.ini info