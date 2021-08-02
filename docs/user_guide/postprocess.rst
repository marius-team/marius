.. _postprocessing:

***************
Postprocessing
***************

Here we cover how to export the trained embeddings for use.

Supported formats for exporting embeddings:
-------------------------------------------

Numpy array
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyTorch tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CSV/TSV file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Accessing embeddings by original IDs
-------------------------------------------

During preprocessing, a unique integer id was assigned to each node and edge-type. The embedding table is ordered accorded to the mapped integer id, such that the first row (embedding) in the embedding table corresponds to the embedding for node ID 0.

We store the mapping of the original ids to the integer ids in ``<preprocess_dir>/node_mapping.txt``, where this file is two column TSV. The first column is the original ID (can be a string, integer, etc.) and the second column is the corresponding row index in the embedding table. By looking up the ID in this mapping the index in the embedding table can be found.