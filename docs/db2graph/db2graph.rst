Db2Graph: Database to Graph conversion tool
============================================

Introduction
""""""""""""""""""""

**Db2Graph** converts **relational databases** into **graphs as sets of triples** which can be used as **input datasets for Marius**, allowing streamlined preprocessing from database to Marius. Db2Graph comes with Marius but can be used as a standalone tool. Conversion with Db2Graph is achieved in the following steps: 

#. Users import/create the database locally

#. Users define the configuration file and edge SQL SELECT queries

#. Db2Graph executes the SQL SELECT queries

#. Db2Graph transforms the result set of queries into sets of triples

Below we lay out the requirements, definitions, and steps for using Db2Graph, and a real example use case:

Requirements
""""""""""""""""""""

Db2Graph currently supports graph conversion from three relational database management systems: **MySQL**, **MariaDB**, and **PostgreSQL**. Db2Graph requires no additional installation as all the required python packages are part of Marius installation. Please refer to `mairus installation <https://github.com/marius-team/marius/blob/main/README.md>`_ for installing the required packages.

System Design
""""""""""""""""""""

Db2Graph classifies a graph into the following two types:

* Entity Nodes: Nodes that are globally unique. Global uniqueness is ensured by appending ``table-name_col-name_val`` to the literal. In a graph, entity nodes either point to other entity nodes or are pointed to by other entity nodes.
* Edges of Entity Node to Entity Node: Directed edges where both source and destination are entity nodes.

During the conversion, we assume that all nodes are **case insensitive**. We ignore the following set of **invalid nodes names**: ``"0", None, "", 0, "not reported", "None", "none"``.

Db2Graph outputs a set of triplets in the format of ``[source node] [edge] [destination node]`` where each element in the triplets is delimited by a single tab. This output format aligns with the input format of Marius, allowing streamlined preprocessing from database to using Marius.

How to Use
""""""""""""""""""""

First, make sure marius is installed with the optional db2graph dependencies: `python3 -m pip install .[db2graph]`.

Assuming that a database has already been created, graph conversion with Db2Graph can be achieved in the following steps:

#. | First, create a YAML configuration file ``config.yaml`` and a query definition files to contain SQL SELECT queries of type ``edges_queries``. Assume that the config file and query file are placed in a ``./conf/`` directory. 

    .. code-block:: bash
    
       $ ls -l .
       conf/  
         config.yaml                         # config file
         edges_queries.txt             # defines edges_queries

   | Define the configuration file in ``config.yaml``. Below is a sample configuration file. Note that all fields are required. An error would be thrown if the query files do not exist.
    
        .. code-block:: yaml
        
            db_server: postgre-sql
            db_name: sample_db
            db_user: sample_user
            db_password: sample_password
            db_host: localhost
            edges_queries: conf/edges_queries.txt

    .. list-table::
       :widths: 15 10 50 15
       :header-rows: 1
    
       * - Key
         - Type
         - Description
         - Required
       * - db_server
         - String
         - Denotes the RDBMS to use. Options: [“maria-db”, “postgre-sql”, "my-sql"].
         - Yes
       * - db_name
         - String
         - Denotes the name of the database.
         - Yes
       * - db_user
         - String
         - Denotes the user name to access the database.
         - Yes
       * - db_password
         - String
         - Password to access the database.
         - Yes
       * - db_host
         - String
         - Denotes the hostname of the database.
         - Yes
       * - edges_queries
         - String
         - Path to the text file that contains the SQL SELECT queries fetching edges from entity nodes to entity nodes.
         - Yes

#. | Next, define SQL SELECT queries. Assume the file ``conf/edges_queries.txt`` has been created. In it, define queries with the following format with no empty lines in-between lines. Each edge consists of two rows: A single ``relation_name`` followed by another row of SQL SELECT query. Note that you can include any SQL keyword after WHERE clause.
    
    .. code-block:: sql
           
           relation_name_A_to_B -- this is the name of the edge from A to B
           SELECT table1_name.column_name_A, table2_name.column_name_B FROM table1_name, table1_name WHERE ...; -- this row represents an edge from source entity node A to destination entity node B
           relation_name_B_to_C -- this is the name of the edge from B to C
           SELECT table1_name.column_name_B, table2_name.column_name_C FROM table1_name, table2_name WHERE ...; -- this row represents an edge from source entity node B to destination entity node C

   | The user can expand or shorten the list of queries in the above query definition file to query a certain subset of data from the database.

   .. note:: 
       Db2Graph validates the correctness of format of each query. However, it does not validate the correctness of the queries. That is, it assumes that all column names and table names exist in the given database schema provided by the user. An error will be thrown in the event that the validation check fails.
    
   .. note:: 
       There cannot be ``AS`` alias within the queries. Any alias violates the correctness of the queries in Db2Graph.
    
#. | Lastly, execute Db2Graph with the following commands. Two flags are required. Note that prints will include both errors and general information, and those are also logged to ``./output_dir/output.log``:

    .. code-block:: bash
        
           $ marius_db2graph --config_path conf/config.yaml --output_directory output_dir/
           Starting marius_db2graph conversion tool for config: conf/config.yaml
           ...
           Edge file written to output_dir/edges.txt

   | The  ``--config_path`` flag specifies where the configuration file created by the user is.

   | The  ``--output_directory`` flag specifies where the data will be output and is set by the user. In this example, assume we have not created the output_dir directory. ``db2graph`` will create it for us. 

   | The conversion result will be written to ``edges.txt`` in a newly created directory named ``./output_dir``:
    
    .. code-block:: bash
        
           $ ls -l .
           output_dir/
             edges.txt                       # generated file with sets of triples
             output.log                          # output log file
           conf/  
             config.yaml                         # config file
             edges_queries.txt             # defines edges_queries    
          $ cat output_dir/edges.txt
          column_name_A    relation_name_A_to_B    column_name_B
          column_name_B    relation_name_B_to_C    column_name_C
    
End-to-end Example Use Case
""""""""""""""""""""

We use `the Sakila DVD store database <https://dev.mysql.com/doc/sakila/en/>`_ from MySQL to demonstrate an end-to-end example from converting a database into a graph using Db2Graph to preprocessing and training the dataset using Marius. For simplicity, we have provided a dockerfile and a bash script which install Marius along with Db2Graph and initialize the Sakila database for you. 

#. | First, download an place the provided ``dockerfile`` and ``run.sh`` in the current working directory. Create and run a docker container using the dockerfile. This dockerfile pre-installs Marius and all dependencies needed for using Marius in this end-to-end example. It also copies ``run.sh`` into the container. 

    .. code-block:: bash
    
       $ docker build -t db2graph_image . # Builds a docker image named db2graph_image
       $ docker run --name db2graph_container -itd db2graph_image # Create the container named db2graph_container
       $ docker exec -it db2graph_container bash # Run the container in interactive mode in bash

   | In the root directory of the container, execute ``run.sh``. This script downloads and initializes the Sakila database. Note that the username is set to ``root``, the database name is set to ``sakila_user``, and the password is set to ``sakila_password``.
    
       .. code-block:: bash
    
        $ run.sh
        $ cd marius/

   | To verify that the database has been install correctly:
    
       .. code-block:: bash
    
        $ mysql
        mysql> USE sakila;
        mysql> SHOW FULL tables;
        +----------------------------+------------+
        | Tables_in_sakila           | Table_type |
        +----------------------------+------------+
        | actor                      | BASE TABLE |
        | actor_info                 | VIEW       |
         ...
        23 rows in set (0.01 sec)    

    .. note::
       
       If you see any error of type ``ERROR 2002 (HY000): Can't' connect to local MySQL server through socket '/var/run/mysqld/mysqld.sock' (111)``, run the command ``systemctl start mysql`` and retry.

#. | Next, create the configuration file for using Db2Graph. Assuming we are in the ``marius/`` root directory, create & navigate to the ``datasets/sakila`` directory. Create the ``conf/config.yaml`` and ``conf/edges_queries.txt`` files if they have not been created. 

    .. code-block:: bash 
       
       $ mkdir -p datasets/sakila/conf/
       $ vi datasets/sakila/conf/config.yaml
       $ vi datasets/sakila/conf/edges_queries.txt

   | In ``datasets/sakila/conf/config.yaml``, define the following fields:
    
    .. code-block:: yaml
        
            db_server: my-sql
            db_name: sakila
            db_user: sakila_user
            db_password: sakila_password
            db_host: 127.0.0.1
            edges_queries: datasets/sakila/conf/edges_queries.txt

   | In ``datasets/sakila/conf/edges_queries.txt``, define the following queries. Note that we create three edges/relationships: An actor acted in a film; A film sold by a store; A film categorized as a category.
    
    .. code-block:: sql
           
           acted_in
           SELECT actor.first_name, film.title FROM actor, film_actor, film WHERE actor.actor_id = film_actor.actor_id AND film_actor.film_id = film.film_id ORDER BY film.title ASC;
           sold_by
           SELECT film.title, address.address FROM film, inventory, store, address WHERE film.film_id = inventory.film_id AND inventory.store_id = store.store_id AND store.address_id = address.address_id ORDER BY film.title ASC;
           categorized_as
           SELECT film.title, category.name FROM film, film_category, category WHERE film.film_id = film_category.film_id AND film_category.category_id = category.category_id ORDER BY film.title ASC;  

   | For simplicity, we limit the queries to focus on the film table. The user can expand or shorten the list of queries in each of the above query definition files to query a certain subset of data from the database. For the Sakila database structure, please refer to `this MySQL documentation <https://dev.mysql.com/doc/sakila/en/sakila-structure.html>`_.

    .. note::
       
       The queries above have ``ORDER BY`` clause at the end, which is not compulsory (and can have performance impact). We have kept it for the example because it will ensure same output across multiple runs. For optimal performance remove the ``ORDER BY`` clause.
   
#. | Lastly, execute Db2Graph with the following script:

    .. code-block:: bash
        
           $ marius_db2graph --config_path datasets/sakila/conf/config.yaml --output_directory datasets/sakila/
           Starting marius_db2graph conversion tool for config: datasets/sakila/conf/config.yaml
           ...
           Total execution time: 0.382 seconds
           Edge file written to datasets/sakila/edges.txt

   | The conversion result was written to ``edges.txt`` in the specified directory ``datasets/sakila/``. In ``edges.txt``, there should be 7915 edges representing the three relationships we defined earlier:
    
    .. code-block:: bash
        
           $ ls -1 datasets/sakila/
           edges.txt                       # generated file with sets of triples
           marius_db2graph.log             # output log file
           conf/  
             ...    
          $ cat datasets/sakila/edges.txt
          actor_first_name_rock   acted_in        film_title_academy dinosaur
          actor_first_name_mary   acted_in        film_title_academy dinosaur
          actor_first_name_oprah  acted_in        film_title_academy dinosaur
          ...

    .. note::
       
       This concludes the example for using Db2Graph. For an end-to-end example of using Db2Graph with Marius, continue through the sections below. For example, for a custom link prediction example, follow `Custom Link Prediction example <https://github.com/marius-team/marius/blob/main/docs/examples/python/lp_custom.rst>`_ from the docs. Please refer to docs/examples to see all the examples.
   
#. | Preprocessing and training a custom dataset like the Sakila database is straightforward with the ``marius_preprocess`` and ``marius_train`` commands. These commands come with ``marius`` when ``marius`` is installed.

    .. code-block:: bash
        
           $  marius_preprocess --output_dir datasets/sakila/ --edges datasets/sakila/edges.txt --dataset_split 0.8 0.1 0.1 --delim="\t"
           Preprocess custom dataset
           Reading edges
           /usr/local/lib/python3.8/dist-packages/marius/tools/preprocess/converters/readers/pandas_readers.py:55: ParserWarning: Falling back to the 'python' engine because the 'c' engine does not support regex separators (separators > 1 char and different from '\s+' are interpreted as regex); you can avoid this warning by specifying engine='python'.
             train_edges_df = pd.read_csv(self.train_edges, delimiter=self.delim, skiprows=self.header_length, header=None)
           Remapping Edges
           Node mapping written to: datasets/sakila/nodes/node_mapping.txt
           Relation mapping written to: datasets/sakila/edges/relation_mapping.txt
           Splitting into: 0.8/0.1/0.1 fractions
           Dataset statistics written to: datasets/sakila/dataset.yaml

   | In the above command, we set ``dataset_split`` to a list of ``0.8 0.1 0.1``. Under the hood, this splits ``edge.txt`` into ``edges/train_edges.bin``, ``edges/validation_edges.bin`` and ``edges/test_edges.bin`` based on the given list of fractions.

   | Note that ``edge.txt`` contains three columns delimited by tabs, so we set ``--delim="\t"``.

   | The  ``--edges`` flag specifies the raw edge list file that ``marius_preprocess`` will preprocess (and train later).

   | The  ``--output_directory`` flag specifies where the preprocessed graph will be output and is set by the user. In this example, assume we have not created the datasets/fb15k_237_example repository. ``marius_preprocess`` will create it for us. 

   | For detailed usages of  ``marius_preprocess``, please execute the following command:

    .. code-block:: bash

        $ marius_preprocess -h

   | Let's check again what was created inside the ``datasets/sakila/`` directory:

    .. code-block:: bash

      $ ls -1 datasets/sakila/ 
      dataset.yaml                       # input dataset statistics                                
      nodes/  
        node_mapping.txt                 # mapping of raw node ids to integer uuids
      edges/   
        relation_mapping.txt             # mapping of relations to integer uuids
        test_edges.bin                   # preprocessed testing edge list 
        train_edges.bin                  # preprocessed training edge list 
        validation_edges.bin             # preprocessed validation edge list 
      conf/                              # directory containing config files
        ...  

   | Let's check what is inside the generated ``dataset.yaml`` file:

    .. code-block:: bash

      $ cat datasets/sakila/dataset.yaml
        dataset_dir: /marius/datasets/sakila/
        num_edges: 6332
        num_nodes: 1146
        num_relations: 3
        num_train: 6332
        num_valid: 791
        num_test: 792
        node_feature_dim: -1
        rel_feature_dim: -1
        num_classes: -1
        initialized: false

    .. note:: 
      If the above ``marius_preprocess`` command fails due to any missing directory errors, please create the ``<output_directory>/edges`` and ``<output_directory>/nodes`` directories as a workaround.

   | To train a model, we need to define a YAML configuration file based on information created from ``marius_preprocess``. An example YAML configuration file for the Sakila database (link prediction model with DistMult) is given in ``examples/configuration/sakila.yaml``. Note that the ``dataset_dir`` is set to the preprocessing output directory, in our example, ``datasets/sakila/``.
   
   | Let's create the same YAML configuration file for the Sakila database from scratch. We follow the structure of the configuration file and create each of the four sections one by one. In a YAML file, indentation is used to denote nesting and all parameters are in the format of key-value pairs. 
  
    .. code-block:: bash

      $ vi datasets/sakila/sakila.yaml 

    .. note:: 
      String values in the configuration file are case insensitive but we use capital letters for convention.

   | First, we define the **model**. We begin by setting all required parameters. This includes ``learning_task``, ``encoder``, ``decoder``, and ``loss``. The rest of the configurations can be fine-tuned by the user.

    .. code-block:: yaml
    
        model:
          learning_task: LINK_PREDICTION # set the learning task to link prediction
          encoder:
            layers:
              - - type: EMBEDDING # set the encoder to be an embedding table with 50-dimensional embeddings
                  output_dim: 50
          decoder:
            type: DISTMULT # set the decoder to DistMult
            options:
              input_dim: 50
          loss:
            type: SOFTMAX_CE
            options:
              reduction: SUM
          dense_optimizer: # optimizer to use for dense model parameters. In this case these are the DistMult relation (edge-type) embeddings
              type: ADAM
              options:
                learning_rate: 0.1
          sparse_optimizer: # optimizer to use for node embedding table
              type: ADAGRAD
              options:
                learning_rate: 0.1
        storage:
          # omit
        training:
          # omit
        evaluation:
          # omit
      
   | Next, we set the **storage** and **dataset**. We begin by setting all required parameters. This includes ``dataset``. Here, the ``dataset_dir`` is set to ``datasets/sakila/``, which is the preprocessing output directory.

    .. code-block:: yaml
    
        model:
          # omit
        storage:
          device_type: cuda
          dataset:
            dataset_dir: /marius/datasets/sakila/
          edges:
            type: DEVICE_MEMORY
          embeddings:
            type: DEVICE_MEMORY
          save_model: true
        training:
          # omit
        evaluation:
          # omit

   | Lastly, we configure **training** and **evaluation**. We begin by setting all required parameters. We begin by setting all required parameters. This includes ``num_epochs`` and ``negative_sampling``. We set ``num_epochs=10`` (10 epochs to train) to demonstrate this example. Note that ``negative_sampling`` is required for link prediction.

    .. code-block:: yaml
    
        model:
          # omit
        storage:
          # omit
        training:
          batch_size: 1000
          negative_sampling:
            num_chunks: 10
            negatives_per_positive: 500
            degree_fraction: 0.0
            filtered: false
          num_epochs: 10
          pipeline:
            sync: true
          epochs_per_shuffle: 1        
        evaluation:
          batch_size: 1000
          negative_sampling:
            filtered: true
          pipeline:
            sync: true   

   | After defining our configuration file, training is run with ``marius_train <your_config.yaml>``.

   | We can now train our example using the configuration file we just created by running the following command (assuming we are in the ``marius`` root directory):

    .. code-block:: bash

      $ marius_train datasets/sakila/sakila.yaml  
      [2022-06-19 07:01:39.828] [info] [marius.cpp:44] Start initialization
      [06/19/22 07:01:44.287] Initialization Complete: 4.458s
      [06/19/22 07:01:44.292] ################ Starting training epoch 1 ################
      [06/19/22 07:01:44.308] Edges processed: [1000/6332], 15.79%
      [06/19/22 07:01:44.311] Edges processed: [2000/6332], 31.59%
      [06/19/22 07:01:44.313] Edges processed: [3000/6332], 47.38%
      [06/19/22 07:01:44.315] Edges processed: [4000/6332], 63.17%
      [06/19/22 07:01:44.317] Edges processed: [5000/6332], 78.96%
      [06/19/22 07:01:44.320] Edges processed: [6000/6332], 94.76%
      [06/19/22 07:01:44.322] Edges processed: [6332/6332], 100.00%
      [06/19/22 07:01:44.322] ################ Finished training epoch 1 ################
      [06/19/22 07:01:44.322] Epoch Runtime: 29ms
      [06/19/22 07:01:44.322] Edges per Second: 218344.83
      [06/19/22 07:01:44.322] Evaluating validation set
      [06/19/22 07:01:44.329]
      =================================
      Link Prediction: 1582 edges evaluated
      Mean Rank: 548.639697
      MRR: 0.005009
      Hits@1: 0.000632
      Hits@3: 0.001264
      Hits@5: 0.001264
      Hits@10: 0.001896
      Hits@50: 0.034766
      Hits@100: 0.075221
      =================================
      [06/19/22 07:01:44.330] Evaluating test set
      [06/19/22 07:01:44.333]
      =================================
      Link Prediction: 1584 edges evaluated
      Mean Rank: 525.809343
      MRR: 0.006225
      Hits@1: 0.000000
      Hits@3: 0.001263
      Hits@5: 0.004419
      Hits@10: 0.005682
      Hits@50: 0.046086
      Hits@100: 0.107323
      =================================

   | After running this configuration for 10 epochs, we should see a result similar to below:

    .. code-block:: bash

      [06/19/22 07:01:44.524] ################ Starting training epoch 10 ################
      [06/19/22 07:01:44.527] Edges processed: [1000/6332], 15.79%
      [06/19/22 07:01:44.529] Edges processed: [2000/6332], 31.59%
      [06/19/22 07:01:44.531] Edges processed: [3000/6332], 47.38%
      [06/19/22 07:01:44.533] Edges processed: [4000/6332], 63.17%
      [06/19/22 07:01:44.536] Edges processed: [5000/6332], 78.96%
      [06/19/22 07:01:44.538] Edges processed: [6000/6332], 94.76%
      [06/19/22 07:01:44.540] Edges processed: [6332/6332], 100.00%
      [06/19/22 07:01:44.540] ################ Finished training epoch 10 ################
      [06/19/22 07:01:44.540] Epoch Runtime: 16ms
      [06/19/22 07:01:44.540] Edges per Second: 395749.97
      [06/19/22 07:01:44.540] Evaluating validation set
      [06/19/22 07:01:44.544]
      =================================
      Link Prediction: 1582 edges evaluated
      Mean Rank: 469.225664
      MRR: 0.047117
      Hits@1: 0.030973
      Hits@3: 0.044880
      Hits@5: 0.051833
      Hits@10: 0.071429
      Hits@50: 0.136536
      Hits@100: 0.197219
      =================================
      [06/19/22 07:01:44.544] Evaluating test set
      [06/19/22 07:01:44.547]
      =================================
      Link Prediction: 1584 edges evaluated
      Mean Rank: 456.828283
      MRR: 0.041465
      Hits@1: 0.023990
      Hits@3: 0.040404
      Hits@5: 0.051768
      Hits@10: 0.068813
      Hits@50: 0.147096
      Hits@100: 0.210227
      =================================
   
   | Let's check again what was added in the ``datasets/sakila/`` directory. For clarity, we only list the files that were created in training. Notice that several files have been created, including the trained model, the embedding table, a full configuration file, and output logs:

    .. code-block:: bash

      $ ls datasets/sakila/ 
      model_0/
        embeddings.bin                   # trained node embeddings of the graph
        embeddings_state.bin             # node embedding optimizer state
        model.pt                         # contains the dense model parameters, embeddings of the edge-types
        model_stlsate.pt                 # optimizer state of the trained model parameters
        node_mapping.txt                 # mapping of raw node ids to integer uuids
        relation_mapping.txt             # mapping of relations to integer uuids
        full_config.yaml                 # detailed config generated based on user-defined config
        metadata.csv                     # information about metadata
        logs/                            # logs containing output, error, debug information, and etc.
      nodes/  
        ...
      edges/   
        ...
      ...

    .. note:: 
        ``model.pt`` contains the dense model parameters. For DistMult, this is the embeddings of the edge-types. For GNN encoders, this file will include the GNN parameters.
      