# preprocess the wn18 graph and put preprocessed graph into output dir
marius_preprocess wn18 output_dir/

# run marius on the preprocessed input
marius_train examples/training/configs/wn18_cpu.ini info