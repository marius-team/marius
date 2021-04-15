# preprocess the freebase86m graph and put preprocessed graph into output dir
marius_preprocess freebase86m output_dir/

# run marius on the preprocessed input
marius_train examples/training/configs/freebase86m_multi_gpu.ini info