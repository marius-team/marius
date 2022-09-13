.. _marius_postprocess

Model exporting tool (marius_postprocess)
==================================================

This page describes the marius_postprocess tool, which can convert and export trained models to csv or parquet formats.

Currently this tool is in memory only, thus the embedding table(s) must fit in CPU memory to perform the export.

Example Usage
##############################

Train the fb15k_237 example model:

    .. code-block:: bash

        marius_preprocess --dataset fb15k_237 --output_directory datasets/fb15k_237_example/
        marius_train examples/configuration/fb15k_237.yaml

The trained model is located at: ``datasets/fb15k_237_example/model_0``

Export the model to CSV format:

    .. code-block:: bash

        marius_postprocess --model_dir datasets/fb15k_237_example/model_0 --format csv --output_dir my_output_dir

The output of the command should look like:

    .. code-block:: text

        Wrote my_output_dir/embeddings.csv: shape (14541, 2)
        Wrote my_output_dir/relation_embeddings.csv: shape (237, 2)
        Wrote my_output_dir/inverse_relation_embeddings.csv: shape (237, 2)
        Wrote my_output_dir/model.pt

From the above we can see that the node and edge-type (relation) embeddings have been written to CSV files. model.pt is a pytorch model file which may contain additional model parameters (GNN weights).

The output files contain two columns (id, embedding). The number of rows corresponds to the number of nodes or number of edge-types.

Note that the CSV format may not be ideal for exporting the embedding table(s), as the embedding vectors are converted to text representations. The model can be exported in parquet format by using ``--format parquet``, or copied in raw format with ``--format bin``.

Command line arguments
##############################

Below is the help message for the tool, containing an overview of the tools arguments and usage.

    .. code-block:: text

        $ marius_postprocess --help
        usage: postprocess [-h] [--model_dir model_dir] [--format format] [--delim delim] [--output_dir output_dir] [--overwrite]

        Convert trained embeddings to desired output format and output to specified directory.

        Example usage:
        marius_postprocess --model_dir foo --format csv --output_dir bar

        optional arguments:
          -h, --help            show this help message and exit
          --model_dir model_dir
                                Directory of the trained model
          --format format, -f format
                                Format of output embeddings. Choices are [csv, parquet, binary]
          --delim delim         Delimiter to use for the output CSV
          --output_dir output_dir
                                Output directory, if not provided the model directory will be used.
          --overwrite           If enabled, the output directory will be overwritten

