# preprocess the wn18 graph and put preprocessed graph into output dir
marius_preprocess output_dir/ --dataset wn18

# run marius on the preprocessed input
marius_train examples/training/configs/wn18_gpu.ini