# preprocess the live_journal graph and put preprocessed graph into output dir
marius_preprocess output_dir/ --dataset live_journal

# run marius on the preprocessed input
marius_train examples/training/configs/live_journal_multi_gpu.ini