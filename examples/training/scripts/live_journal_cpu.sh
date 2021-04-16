# preprocess the live_journal graph and put preprocessed graph into output dir
marius_preprocess live_journal output_dir/

# run marius on the preprocessed input
marius_train examples/training/configs/live_journal_cpu.ini info