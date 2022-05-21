Db2Graph: Database to Graph conversion tool
============================================

Introduction
""""""""""""""""""""

**Db2Graph** converts **relational databases** into **graphs as sets of triples** which can be used as **input datasets for Marius**, allowing streamlined preprocessing from database to Marius. Db2Graph comes with Marius but can be used as a standalone tool. Conversion with Db2Graph is achieved in the following steps: 

#. Users import/create the database locally

#. Users define the configuration file and entity/edge SQL SELECT queries

#. Db2Graph executes the SQL SELECT queries

#. Db2Graph transforms the result set of queries into sets of triples

Below we lay out the requirements, definitions, and steps for using Db2Graph, and a real example use case:

Requirements
""""""""""""""""""""

Db2Graph currently supports graph conversion from three relational database management systems: **MySQL**, **MariaDB**, and **PostgreSQL**. Db2Graph requires no installation and all the required python packages are part of Marius installation. Following are the packages required:

* python >= 3.6  (included in Marius installation)
* pandas  (included in Marius installation)
* hydra  (included in Marius installation)
* psutil  (included in Marius installation)
* MySQL connector >= 8.0 (included in Marius installation)
* Psycopg2 >= 2.9 (included in Marius installation)

System Design
""""""""""""""""""""

Db2Graph classifies a graph into the following two types:

* Entity Nodes: Nodes that are globally unique. Global uniqueness is ensured by appending ``table-name_col-name_val`` to the literal. In a graph, entity nodes either point to other entity nodes or are pointed to by other entity nodes.
* Edges of Entity Node to Entity Node: Directed edges where both source and destination are entity nodes.

During the conversion, we assume that all nodes are **case insensitive**. We ignore the following set of **invalid nodes names**: ``"0", None, "", 0, "not reported", "None", "none"``.

Db2Graph outputs a set of triplets in the format of ``[source node] [edge] [destination node]`` where each element in the triplets is delimited by a single tab. This output format aligns with the input format of Marius, allowing streamlined preprocessing from database to using Marius.

How to Use
""""""""""""""""""""

Assuming a database has been created locally and ``marius`` has been installed successfully, database to graph conversion with Db2Graph can be achieved in the following steps: 

#. | First, create a YAML configuration file ``config.yaml`` and a query definition files to contain SQL SELECT queries of type ``edges_entity_entity_queries``. Assume that the config file and query file are placed in a ``./conf/`` directory. 

    .. code-block:: bash
    
       $ ls -l .
       conf/  
         config.yaml                         # config file
         edges_entity_entity.txt             # defines edges_entity_entity_queries

   | Define the configuration file in ``config.yaml``. Below is a sample configuration file. Note that all fields are required. An error would be thrown if the query files do not exist.
    
        .. code-block:: yaml
        
            db_server: postgre-sql
            db_name: sample_db
            db_user: sample_user
            db_password: sample_password
            db_host: localhost
            edges_entity_entity_queries: conf/edges_entity_entity.txt

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
       * - edges_entity_entity_queries
         - String
         - Path to the text file that contains the SQL SELECT queries fetching edges from entity nodes to entity nodes.
         - Yes

#. | Next, define SQL SELECT queries. Assume the file ``conf/edges_entity_entity.txt`` has been created. In it, define queries with the following format. Each edge consists of two rows: A single ``relation_name`` followed by another row of SQL SELECT query. Note that you can any SQL keyword after WHERE clause.
    
    .. code-block:: sql
           
           relation_name_A_to_B -- this is the name of the edge from A to B
           SELECT table1_name.column_name_A, table2_name.column_name_B FROM table1_name, table1_name WHERE ...; -- this row represents an edge from source entity node A to destination entity node B
           relation_name_B_to_C -- this is the name of the edge from B to C
           SELECT table1_name.column_name_B, table2_name.column_name_C FROM table1_name, table2_name WHERE ...; -- this row represents an edge from source entity node B to destination entity node C

   | The user can expand or shorten the list of queries in the above query definition file to query a certain subset of data from the database.

   .. note:: 
       Db2Graph validates the correctness of format of each query. However, it does not validate the correctness of the queries. That is, it assumes that all column names and table names exist in the given database schema provided by the user. An error will be thrown in the event that the validation check fails.
    
#. | Lastly, execute Db2Graph with the following commands. Two flags are required. Note that prints will include both errors and general information, and those are also logged to ``./output_dir/output.log``:

    .. code-block:: bash
        
           $ MARIUS_NO_BINDINGS=1 marius_db2graph --config_path conf/config.yaml --output_directory output_dir/
           Starting a new run!!!
           ...
           Edge file written to output_dir/all_edges.txt

   | The  ``--config_path`` flag specifies where the configuration file created by the user is.

   | The  ``--output_directory`` flag specifies where the data will be output and is set by the user. In this example, assume we have not created the output_dir directory. ``db2graph`` will create it for us. 

   | The conversion result will be written to ``all_edges.txt`` in a newly created directory named ``./output_dir``:
    
    .. code-block:: bash
        
           $ ls -l .
           output_dir/
             all_edges.txt                       # generated file with sets of triples
             output.log                          # output log file
           conf/  
             config.yaml                         # config file
             edges_entity_entity.txt             # defines edges_entity_entity_queries    
          $ cat output_dir/all_edges.txt
          column_name_A    relation_name_A_to_B    column_name_B
          column_name_B    relation_name_B_to_C    column_name_C
    
Example Use Case
""""""""""""""""""""

We use `The Movie Dataset <https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset>`_ from Kaggle as an example to demonstrate a step-by-step walkthrough from loading a dataset into a PostgreSQL database to defining the edge queries and to converting the database into a graph using Db2Graph. Note the following steps assume the database has not been created and Marius has not been installed.

#. | First, create a docker container from the PostgreSQL image. This container will contain all of our work in this example. Note that the password of this container, ``password``, will be the password of the database we create.

    .. code-block:: bash
    
       $ docker run --name movies_dataset -e POSTGRES_PASSWORD=password -d postgres:12  
       $ docker exec -it movies_dataset bash # Attach to the container in interactive mode in bash

   | Create a PostgreSQL database ``test_db`` with the username set to ``postgres`` and the password being ``password``. (Assuming in the root directory)
    
       .. code-block:: bash
    
        $ psql -U postgres
        > postgres=# create database test_db; 
        > postgres=# \q

   | Download `The Movie Dataset <https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset>`_ from Kaggle and load it using `the-movie-database-import <https://github.com/guenthermi/the-movie-database-import.git>`_ script. The script allows for easy import into the PostgreSQL database created in prior steps. Note that we place the downloaded ``archive.zip`` dataset from Kaggle in the ``dataset/`` directory. We skip the downloading step as different ways can be used. 
    
       .. code-block:: bash
    
        $ git clone https://github.com/guenthermi/the-movie-database-import.git 
        $ cd the-movie-database-import
        $ vi db_config.json # update the username, password, host, and db_name fields if applicable. Here, password is changed to 'password' and db_name is 'test_db'
        $ mkdir dataset/ # create a dataset directory and place the downloaded dataset file in it
        $ unzip archive.zip # unzip the downloaded dataset file
        $ python3 loader.py dataset/ # load the dataset files from the path to your dataset folder
        $ psql -U postgres -d test_db # check what is inside the database now
        > postgres=# \d
                                 List of relations
         Schema |                Name                |   Type   |  Owner
        --------+------------------------------------+----------+----------
         public | actors                             | table    | postgres
         public | actors_id_seq                      | sequence | postgres
         ...
        (30 rows)    
   
   | This creates 15 tables containing information about actors, movies, keywords, production companies, production countries, as well as credits data.
   
   | Install ``marius_db2graph`` and the required dependencies.
   
   .. code-block:: bash 
       
       $ cd / # back to root directory
       $ apt-get update
       $ apt-get install vim
       $ apt-get install git
       $ apt-get install python3
       $ apt-get install python3-pip
       $ git clone https://github.com/marius-team/marius.git
       $ cd marius
       $ MARIUS_NO_BINDINGS=1 python3 -m pip install . 

#. | Next, create the configuration files. From the root directory, create & navigate to an empty directory and create the ``conf/config.yaml`` and ``conf/edges_entity_entity.txt`` files if they have not been created. 

    .. code-block:: bash 
       
       $ mkdir empty_dir
       $ cd empty_dir
       $ vi conf/config.yaml

   | In ``conf/config.yaml``, define the following fields:
    
    .. code-block:: yaml
        
            db_server: postgre-sql
            db_name: test_db
            db_user: postgres
            db_password: password
            db_host: 127.0.0.1
            edges_entity_entity_queries: conf/edges_entity_entity.txt

   | In ``conf/edges_entity_entity.txt``, define the following queries. Note that we create three edges/relationships: An actor acted in a movie; A movie directed by a director; A movie produced by a production company.
    
    .. code-block:: sql
           
           acted_in
           SELECT persons.name, movies.title FROM persons, actors, movies WHERE persons.id = actors.person_id AND actors.movie_id = movies.id ORDER BY persons.name ASC;
           directed_by
           SELECT movies.title, persons.name FROM persons, directors, movies WHERE persons.id = directors.director_id AND directors.movie_id = movies.id ORDER BY movies.title ASC;
           produced_by
           SELECT movies.title, production_companies.name FROM production_companies, movies_production_companies, movies WHERE production_companies.id = movies_production_companies.production_company_id AND movies_production_companies.movie_id = movies.id ORDER BY movies.title ASC;  

   | For simplicity, we limit the queries to focus on the movies table. The user can expand or shorten the list of queries in each of the above query definition files to query a certain subset of data from the database.

   .. note::
       
       The queries above have ``ORDER BY`` clause at the end, which is not compulsory (and can have performance impact). We have kept it for the example because it will ensure same output across multiple runs. For optimal performance remove the ``ORDER BY`` clause.

#. | Lastly, execute Db2Graph with the following script:

    .. code-block:: bash
        
           $ MARIUS_NO_BINDINGS=1 marius_db2graph --config_path conf/config.yaml --output_directory output_dir/
           Starting a new run!!!
           ...
           Edge file written to output_dir/all_edges.txt

   | The conversion result was written to ``all_edges.txt`` in a newly created directory ``./output_dir``. In ``all_edges.txt``, there should be 679923 edges representing the three relationships we defined earlier:
    
    .. code-block:: bash
        
           $ ls -l .
           output_dir/
             all_edges.txt                       # generated file with sets of triples
             output.log                          # output log file
           conf/  
             ...    
          $ cat output_dir/all_edges.txt
          persons_name_강계열	acted_in	movies_title_님아, 그 강을 건너지 마오
          persons_name_조병만	acted_in	movies_title_님아, 그 강을 건너지 마오
          persons_name_2 chainz	acted_in	movies_title_the art of organized noize
          ...