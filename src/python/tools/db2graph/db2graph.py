import argparse
import pandas as pd
import mysql.connector
from mysql.connector import errorcode
import psycopg2
from pathlib import Path
import uuid
import re
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import time
import logging
import psutil

INVALID_ENTRY_LIST = ["0", None, "", 0, "not reported", "None", "none"]

def set_args():
    parser = argparse.ArgumentParser(
                description='Db2Graph', prog='db2graph')

    parser.add_argument('--config_path',
                        metavar='config_path',
                        type=str,
                        default="",
                        help='Path to the config file')

    parser.add_argument('--output_directory',
                        metavar='output_directory',
                        type=str,
                        default="./",
                        help='Directory to put output data and log file')
    return parser

def config_parser_fn(config_name):
    """
    Takes the input yaml config file's name (& relative path). Returns all the extracted data

    :param config_name: file name (& relative path) for the YAML config file
    :returns:
        - db_server: string denoting database server (initial support only for mariadb)
        - db_name: name of the database you need to pull from
        - db_user: user name used to access the database
        - db_password: password used to access the database
        - db_host: hostname of the database
        - entity_node_sql_queries: list of sql queries used to define entity nodes
        - edge_entity_entity_sql_queries: list of sql queries to define edges of type entity nodes to entity nodes 
            & the names of edges
        - edge_entity_feature_values_sql_queries: list of sql queries to define edges of type entity node to feature 
            values & also the names of edges
    """
    input_cfg = None
    input_config_path = Path(config_name).absolute()

    config_name = input_config_path.name
    config_dir = input_config_path.parent

    with hydra.initialize_config_dir(config_dir=config_dir.__str__()):
        input_cfg = hydra.compose(config_name=config_name)

    # db_server used to distinguish between different databases
    db_server = None
    if "db_server" in input_cfg.keys():
        db_server = input_cfg["db_server"]
    else:
        print("ERROR: db_server is not defined")
        exit(1)

    # db_name is the name of the database to pull the data from
    db_name = None
    if "db_name" in input_cfg.keys():
        db_name = input_cfg["db_name"]
    else:
        print("ERROR: db_name is not defined")
        exit(1)

    # db_user is the user name used to access the database
    db_user = None
    if "db_user" in input_cfg.keys():
        db_user = input_cfg["db_user"]
    else:
        print("ERROR: db_user is not defined")
    
    # db_password is the password used to access the database
    db_password = None
    if "db_password" in input_cfg.keys():
        db_password = input_cfg["db_password"]
    else:
        print("ERROR: db_password is not defined")
    
    # db_host is the hostname of the database
    db_host = None
    if "db_host" in input_cfg.keys():
        db_host = input_cfg["db_host"]
    else:
        print("ERROR: db_host is not defined")

    # generate_uuid is used to identify whether to generate the uuid using entity nodes or 
    # use the table_name-col_name-value as the unique identifier
    generate_uuid = None
    if "generate_uuid" in input_cfg.keys():
        generate_uuid = input_cfg["generate_uuid"]
    else:
        # default is assumed as ture
        generate_uuid = True

    # Getting all the entity nodes sql queries in a list
    entity_node_sql_queries = list()
    if "entity_node_queries" in input_cfg.keys():
        query_filepath = input_cfg["entity_node_queries"]

        if not Path(query_filepath).exists():
            raise ValueError("{} does not exist".format(str(query_filepath)))

        file = open(query_filepath, 'r')
        entity_node_sql_queries = file.readlines()
        for i in range(len(entity_node_sql_queries)):
            # Removing the last '\n' character
            if (entity_node_sql_queries[i][-1] == '\n'):
                entity_node_sql_queries[i] = entity_node_sql_queries[i][:-1]
    else:
        print("ERROR: entity_node_queries is not defined")
        exit(1)

    # Getting all edge queries for edge type entity node to entity node
    edge_entity_entity_sql_queries = list()
    edge_entity_entity_rel_list = list()
    if "edges_entity_entity_queries" in input_cfg.keys():
        query_filepath = input_cfg["edges_entity_entity_queries"]

        if not Path(query_filepath).exists():
            raise ValueError("{} does not exist".format(str(query_filepath)))

        file = open(query_filepath, 'r')
        # edge_entity_entity_sql_queries = file.readlines()
        read_lines = file.readlines()
        for i in range(len(read_lines)):
            # Removing the last '\n' character
            if (read_lines[i][-1] == '\n'):
                read_lines[i] = read_lines[i][:-1]
            
            # Adding the line to rel_list if even else its a query
            if (i % 2 == 0):
                edge_entity_entity_rel_list.append(read_lines[i])
            else:
                edge_entity_entity_sql_queries.append(read_lines[i])
    else:
        print("ERROR: edges_entity_entity_queries is not defined")
        exit(1)

    # Gettting all edge queries for edge type entity node to feature values
    edge_entity_feature_values_sql_queries = list()
    edge_entity_feature_values_rel_list = list()
    if "edges_entity_feature_values_queries" in input_cfg.keys():
        query_filepath = input_cfg["edges_entity_feature_values_queries"]

        if not Path(query_filepath).exists():
            raise ValueError("{} does not exist".format(str(query_filepath)))

        file = open(query_filepath, 'r')
        read_lines = file.readlines()
        for i in range(len(read_lines)):
            # Removing the last '\n' character
            if (read_lines[i][-1] == '\n'):
                read_lines[i] = read_lines[i][:-1]
            
            # Adding the line to rel_list if even else its a query
            if (i % 2 == 0):
                edge_entity_feature_values_rel_list.append(read_lines[i])
            else:
                edge_entity_feature_values_sql_queries.append(read_lines[i])
    else:
        print("ERROR: edges_entity_feature_values_queries is not defined")
        exit(1)

    return db_server, db_name, db_user, db_password, db_host, generate_uuid, entity_node_sql_queries, edge_entity_entity_sql_queries,\
     edge_entity_entity_rel_list, edge_entity_feature_values_sql_queries, edge_entity_feature_values_rel_list

def connect_to_db(db_server, db_name, db_user, db_password, db_host):
    """
    Function takes db_server and db_name as the input. Tries to connect to the database and returns an object
    which can be used to execute queries.
    Assumption: default user: root, host: 127.0.0.1 and password:"". You will need to change code if otherwise

    :param db_server: The name of the backend database application used for accessing data
    :param db_name: The name of the database where the data resides
    :param db_user: The user name used to access the database
    :param db_password: The password used to access the database
    :param db_host: The hostname of the database
    :return cnx: cursor object that can be used to execute the database queries
    """
    if db_server == 'maria-db':
        try:
            cnx = mysql.connector.connect(user=db_user,
                                        password=db_password,  # change password to your own
                                        host=db_host,
                                        database=db_name)
            # cursor = cnx.cursor(name="my_cursor") # Make cursor only when you need it
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print(f"Incorrect user name or password\n{err}")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print(f"Non-existing database\n{err}")
            else:
                print(err)

    elif db_server == 'postgre-sql':
        try:
            cnx = psycopg2.connect(user=db_user,
                                 password=db_password,  # change password to your own
                                 host=db_host,
                                 database=db_name)
            # Note: It is important to have cursor name in psycopg else all the data will be transfered 
            # to client application and lead to out-of-memory error with just cursor.execute()
            # Reference: https://www.psycopg.org/docs/usage.html#server-side-cursors
            # Or: https://stackoverflow.com/questions/28343240/psycopg2-uses-up-memory-on-large-select-query
            
            # Make cursor only when you need it - As named cursor can only execute one query so need to make new ones
            # cursor = cnx.cursor(name="my_cursor")

        except psycopg2.Error as err:
            print(f"Error\n{err}")

    else:
        print('Other databases are currently not supported.')
    
    return cnx

def validation_check_entity_queries(entity_query_list):
    """
    Ensures that the entity queries are correctly formatted.

    :param entity_query_list: List of entity queries and each will be checked and validated
    :return new_query_list: List of new queries with necessary updates
    """
    # Format: SELECT DISTINCT table_name.col_name FROM ____ WHERE ____;
    new_query_list = list()
    for q in range(len(entity_query_list)):
        qry_split = entity_query_list[q].split()
        
        check_var = qry_split[0].lower() # To ensure no case sensitivity issues
        if (check_var != "select"):
            print("Error: Incorrect entity query formatting, not starting with SELECT")
            exit(1)
        
        check_var = qry_split[1].lower()
        if (check_var != "distinct"):
            print("Adding distinct to the entity query " + str(q) +" (0 indexed position)")
            qry_split.insert(1,"distinct")
        
        check_split = qry_split[2].split('.')
        if (len(check_split) != 2):
            print("Error: Incorrect entity query formatting, table_name.col_name should in the SELECT line")
            exit(1) 
        
        check_var = qry_split[3].lower()
        if (check_var != "from"):
            print("Error: Incorrect entity query formatting, FROM not at correct position")
            exit(1)
        
        # # We have rigid stop at table name because we are using this structure to extract table name
        # # Update: Table name extraction logic updated so no longer need this check 
        # if (qry_split[4][-1] != ";"):
        #     print("Error: Incorrect entity query formatting, there should be nothing after table name")
        #     exit(1)
        
        new_query_list.append(' '.join(qry_split))
    
    return new_query_list

def validation_check_edge_entity_entity_queries(edge_entity_entity_queries_list):
    """
    Ensures that the edge_entity_entity_queries are correctly formatted.

    :param edge_entity_entity_queries_list: List of all the queries defining edges from entity nodes to entity nodes
    :return new_query_list: These are updated queries with necessary changes if any
    """
    # Format: SELECT table1_name.col1_name, table2_name.col2_name FROM ____ WHERE ____ (and so on);
    new_query_list = list()
    for q in range(len(edge_entity_entity_queries_list)):
        qry_split = edge_entity_entity_queries_list[q].split()
        
        check_var = qry_split[0].lower()
        if (check_var != "select"):
            print("Error: Incorrect edge entity node - entity node formatting, " +
                "not starting with SELECT")
            exit(1)
        
        check_split = qry_split[1].split('.')
        if (len(check_split) != 2):
            print("Error: Incorrect edge entity node - entity node formatting, " +
                "table1_name.col1_name not correctly formatted")
            exit(1)
        if (check_split[1][-1] != ','):
            print("Error: Incorrect edge entity node - entity node formatting, " +
                "missing ',' at the end of table1_name.col1_name")
            exit(1)
        
        check_split = qry_split[2].split('.')
        if (len(check_split) != 2):
            print("Error: Incorrect edge entity node - entity node formatting, " +
                "table2_name.col2_name not correctly formatted")
            exit(1)
        
        check_var = qry_split[3].lower()
        if (check_var != "from"):
            print("Error: Incorrect edge entity node - entity node formatting, " +
                "extra elements after table2_name.col2_name")
            exit(1)
        
        new_query_list.append(edge_entity_entity_queries_list[q])
    
    return new_query_list

def validation_check_edge_entity_feature_val_queries(edge_entity_feature_val_queries_list):
    """
    Ensures that the edge_entity_feature_val_queries_list are correctly formatted.

    :param edge_entity_feature_val_queries_list: List of all the queries defining edges from entity node to feature values
    :return new_query_list: These are updated queries with necessary changes if any
    """
    # Format: SELECT table1_name.col1_name, ____ FROM ____ WHERE ____ (and so on);
    new_query_list = list()
    for q in range(len(edge_entity_feature_val_queries_list)):
        qry_split = edge_entity_feature_val_queries_list[q].split(' ')
        
        check_var = qry_split[0].lower()
        if (check_var != "select"):
            print("Error: Incorrect edge entity node - feature value formatting, " +
                "not starting with SELECT")
            exit(1)
        
        check_split = qry_split[1].split('.')
        if (len(check_split) != 2):
            print("Error: Incorrect edge entity node - feature value formatting, " +
                "table1_name.col1_name not correctly formatted")
            exit(1)
        if (check_split[1][-1] != ','):
            print("Error: Incorrect edge entity node - feature value formatting, " +
                "missing ',' at the end of table1_name.col1_name")
            exit(1)
        
        new_query_list.append(edge_entity_feature_val_queries_list[q])
    
    return new_query_list

def clean_token(token):
    """
    Helper to clean a dataframe, can be used by applying this function to a dataframe

    :param token: elements to clean
    :return token: cleaned token
    """
    token = str(token)
    token = token.strip().strip("\t.\'\" ")
    return token.lower()

def get_uuid(row):
    """
    Gets the UUID associated with entity_node on this row consisting of a three-tuple

    :param row: a row in the dataframe
    :return uuid: the UUID associated with entity_node on this row
    """
    return uuid.uuid5(uuid.NAMESPACE_DNS, row['entity_node'])

def entity_node_to_uuids(output_dir, cnx, entity_queries_list, db_server):
    """
    Takes entity node queries as inputs, execute the queries, store the results in temp dataframes,
    then concatenate each entity node with its respective table name & column name, 
    convert each into uuid, and store the mapping in a dictionary
    Assumption: Entries are case insentitive, i.e. 'BOB' and 'bob' are considered as duplicates

    :param output_dir: Directory to put output file
    :param cnx: Cursor object
    :param entity_queries_list: List of all the entity queries 
    :param db_server: Database server name
    :return entity_mapping: dictionary of Entity_nodes mapping to UUIDs
    """
    logging.info('starting entity_node_to_uuids')
    entity_mapping = pd.DataFrame()


    fetchSize = 10000
    # New post processing code for edges entity node to entity nodes
    for i in range(len(entity_queries_list)):
        start_time2 = time.time()
        first_pass = True

        # Executing the query and timing it
        add_time1 = time.time()
        query = entity_queries_list[i]
        cursor_name = "entity_queries_list" + str(i)  # Name imp because: https://www.psycopg.org/docs/usage.html#server-side-cursors
        cursor = []
        if db_server == 'maria-db':
            cursor = cnx.cursor()
        elif db_server == 'postgre-sql':
            cursor = cnx.cursor(name = cursor_name)
        cursor.execute(query)
        logging.info(f'Cursor.execute time is: {time.time() - add_time1}')

        # Getting Basic Details
        # extracting table and column names
        table_name = query.split()[2].split('.')[0]  # table name of the query to execute
        col_name = str(query.split()[2].split('.')[1]) # column name of the query
        
        # Processing each batch of cursor on client
        rowsCompleted = 0

        # In an initial sample pass, estimates the optimal maximum possible fetchSize for given query based on memory usage report of virtual_memory() 
        # process data with fetchSize=10000, record the amount of memory used,
        # increase fetchSize if the amount of memory used is less than half of machine's total available memory, 
        # Note: all unit size are in bytes, fetchSize limited between 10000 and 100000000 bytes
        if first_pass:
            mem_copy = psutil.virtual_memory()
            mem_copy_used = mem_copy.used
            limit_fetchSize = min(mem_copy.available / 2, 1000000000) # max limit 1 billion

        # Potential issue: There might be duplicates now possible as drop_duplicates over smaller range
        # expected that user db does not have dupliacted
        while (True): # Looping till all rows are completed and processed
            result = cursor.fetchmany(fetchSize)
            result = pd.DataFrame(result)
            if (result.shape[0] == 0):
                break

            # Cleaning Part
            result = result.applymap(clean_token)  # strip tokens and lower case strings
            result = result[~result.iloc[:, 0].isin(INVALID_ENTRY_LIST)] # clean invalid data
            result = result.drop_duplicates()  # remove invalid row

            # concatenate each entity node with its respective table nam
            result[result.columns[0]] = table_name + '_' + col_name + '_' + result[result.columns[0]].map(str)

            result['uuid'] = ''
            result.columns = ['entity_node', 'uuid']
            result['entity_node'] = result['entity_node'].str.lower() # entries in lower case
            result = result.drop_duplicates() # removing duplicates
            result['uuid'] = result.apply(lambda row: get_uuid(row), axis=1) # convert each entity node to uuid
            entity_mapping = pd.concat([entity_mapping, result]).drop_duplicates()

            result.to_csv(output_dir / Path("entity_mapping.txt"), sep='\t',\
                header=False, index=False, mode='a') # Appending the output to disk
            del result
            rowsCompleted += fetchSize

            if first_pass:
                delta = psutil.virtual_memory().used - mem_copy_used # delta between two virtual_memory(), i.e. mem used curr fetchSize
                est_fetchSize = limit_fetchSize / (delta + 1) * fetchSize # estimated optimal fetchSize for 
                if est_fetchSize > limit_fetchSize:
                    fetchSize = int(limit_fetchSize)
                elif 10000 < est_fetchSize and est_fetchSize <= limit_fetchSize:
                    fetchSize = int(est_fetchSize)
                else:
                    fetchSize = 10000
                first_pass = False  # executing get_fetchSize means we are in 
        logging.info(f'finishing converting entity nodes to uuid, execution time: {time.time() - start_time2}')
    return entity_mapping.set_index('entity_node').to_dict()['uuid']

def post_processing(output_dir, cnx, edge_entity_entity_queries_list, edge_entity_entity_rel_list, 
    edge_entity_feature_val_queries_list, edge_entity_feature_val_rel_list, entity_mapping,
    generate_uuid, db_server):
    """
    Executes the given queries_list one by one, cleanses the data by removing duplicates,
    then replace the entity nodes with their respective UUIDs, and store the final result in a dataframe/.txt file

    :param output_dir: Directory to put output file
    :param cnx: Cursor object
    :param edge_entity_entity_queries_list: List of all the queries defining edges from entity nodes to entity nodes
    :param edge_entity_entity_rel_list: List of all the relationships defining edges from entity nodes to entity nodes
    :param edge_entity_feature_val_queries_list: List of all the queries defining edges from feature nodes to feature nodes
    :param edge_entity_feature_val_rel_list: List of all the relationships defining edges from feature nodes to feature nodes
    :param entity_mapping: dictionary of Entity_nodes mapping to UUIDs
    :param generate_uuid: boolean value, if ture converts entity nodes to UUID, else skip the conversion
    :param db_server: database server name
    :return 0: 0 for success, exit code 1 for failure
    """
    if (len(edge_entity_entity_queries_list) != len(edge_entity_entity_rel_list)):
        print("Number of queries in edge_entity_entity_queries_list must match number of edges in edge_entity_entity_rel_list")
        logging.error("Number of queries in edge_entity_entity_queries_list must match number of edges in edge_entity_entity_rel_list")
        exit(1)
    
    if (len(edge_entity_feature_val_queries_list) != len(edge_entity_feature_val_rel_list)):
        print("Number of queries in edge_entity_feature_val_queries_list must match number of edges in edge_entity_feature_val_rel_list")
        logging.error("Number of queries in edge_entity_feature_val_queries_list must match number of edges in edge_entity_feature_val_rel_list")
        exit(1)

    src_rel_dst = pd.DataFrame()
    open(output_dir / Path("all_edges.txt"), 'w').close() # Clearing the output file
    logging.info('in postprocessing')

    # These are just for metrics - Only correct when not batch processing
    num_uniq = []  # number of entities
    num_edge_type = []  # number of edges
    

    fetchSize = 10000
    # New post processing code for edges entity node to entity nodes
    for i in range(len(edge_entity_entity_queries_list)):
        start_time2 = time.time()
        first_pass = True

        # Executing the query and timing it
        add_time1 = time.time()
        query = edge_entity_entity_queries_list[i]
        cursor_name = "edge_entity_entity_cursor" + str(i)  # Name imp because: https://www.psycopg.org/docs/usage.html#server-side-cursors
        cursor = []
        if db_server == 'maria-db':
            cursor = cnx.cursor()
        elif db_server == 'postgre-sql':
            cursor = cnx.cursor(name = cursor_name)
        cursor.execute(query)
        logging.info(f'Cursor.execute time is: {time.time() - add_time1}')

        # Getting Basic Details
        table_name_list = re.split(' ', query)  # table name of the query to execute
        table_name1 = table_name_list[1].split('.')[0] # src table
        col_name1 = table_name_list[1].split('.')[1][:-1] # src column, (note last character ',' is removed)
        table_name2 = table_name_list[2].split('.')[0] # dst/target table
        col_name2 = table_name_list[2].split('.')[1] # dst/target column
        
        # Processing each batch of cursor on client
        rowsCompleted = 0

        # In an initial sample pass, estimates the optimal maximum possible fetchSize for given query based on memory usage report of virtual_memory() 
        # process data with fetchSize=10000, record the amount of memory used,
        # increase fetchSize if the amount of memory used is less than half of machine's total available memory, 
        # Note: all unit size are in bytes, fetchSize limited between 10000 and 100000000 bytes
        if first_pass:
            mem_copy = psutil.virtual_memory()
            mem_copy_used = mem_copy.used
            limit_fetchSize = min(mem_copy.available / 2, 1000000000) # max limit 1 billion

        # Potential issue: There might be duplicates now possible as drop_duplicates over smaller range
        # expected that user db does not have dupliacted
        while (True): # Looping till all rows are completed and processed
            result = cursor.fetchmany(fetchSize)
            result = pd.DataFrame(result)
            if (result.shape[0] == 0):
                break

            # Cleaning Part
            result = result.applymap(clean_token)  # strip tokens and lower case strings
            result = result[~result.iloc[:, 1].isin(INVALID_ENTRY_LIST)]  # clean invalid data
            result = result[~result.iloc[:, 0].isin(INVALID_ENTRY_LIST)]
            result = result.drop_duplicates()  # remove invalid row

            result.iloc[:, 0] = table_name1 + "_" + col_name1 + '_' + result.iloc[:, 0]   # src
            result.iloc[:, 1] = table_name2 + "_" + col_name2 + '_' + result.iloc[:, 1] # dst/target
            result.insert(1, "rel", edge_entity_entity_rel_list[i])  # rel
            result.columns = ["src", "rel", "dst"]
            
            if generate_uuid:
                # convert entity nodes to respective UUIDs
                result['src'] = result['src'].map(entity_mapping)
                if (result['src'].isna().any()):
                    logging.warning(f'Some src column entities did not map in edges_entity_entity')
                    exit(1)
                
                result['dst'] = result['dst'].map(entity_mapping)
                if (result['dst'].isna().any()):
                    logging.warning(f'Some dst column entities did not map in edges_entity_entity')
                    exit(1)

            result.to_csv(output_dir / Path("all_edges.txt"), sep='\t',\
                header=False, index=False, mode='a') # Appending the output to disk
            del result
            rowsCompleted += fetchSize

            if first_pass:
                delta = psutil.virtual_memory().used - mem_copy_used # delta between two virtual_memory(), i.e. mem used for curr fetchSize
                est_fetchSize = limit_fetchSize / (delta + 1) * fetchSize # estimated optimal fetchSize for 
                if est_fetchSize > limit_fetchSize:
                    fetchSize = int(limit_fetchSize)
                elif 10000 < est_fetchSize and est_fetchSize <= limit_fetchSize:
                    fetchSize = int(est_fetchSize)
                else:
                    fetchSize = 10000
                first_pass = False  # executing get_fetchSize means we are in 
        logging.info(f'finishing post_processing enttiy nodes, execution time: {time.time() - start_time2}')

    # edges from entity node to feature values processing
    # Note: feature values will not have table_name and col_name appended
    fetchSize = 10000
    for i in range(len(edge_entity_feature_val_queries_list)):
        start_time2 = time.time()
        first_pass = True

        # Executing the query and timing it
        add_time1 = time.time()
        query = edge_entity_feature_val_queries_list[i]
        cursor_name = "edge_entity_feature_val_cursor" + str(i)  # Name imp because: https://www.psycopg.org/docs/usage.html#server-side-cursors
        cursor = []
        if db_server == 'maria-db':
            cursor = cnx.cursor()
        elif db_server == 'postgre-sql':
            cursor = cnx.cursor(name = cursor_name)
        cursor.execute(query)
        logging.info(f'Cursor.execute time is: {time.time() - add_time1}')
            
        # Getting Basic Details
        table_name_list = re.split(' ', query)  # table name of the query to execute
        table_name1 = table_name_list[1].split('.')[0] # src table
        col_name1 = table_name_list[1].split('.')[1][:-1] # src column, (note last character ',' is removed)
        
        # Processing each batch of cursor on client
        rowsCompleted = 0

        # In an initial sample pass, estimates the optimal maximum possible fetchSize for given query based on memory usage report of virtual_memory() 
        # process data with fetchSize=10000, record the amount of memory used,
        # increase fetchSize if the amount of memory used is less than half of machine's total available memory, 
        # Note: all unit size are in bytes, fetchSize limited between 10000 and 100000000 bytes
        if first_pass:
            mem_copy = psutil.virtual_memory()
            mem_copy_used = mem_copy.used
            limit_fetchSize = min(mem_copy.available / 2, 1000000000) # max limit 1 billion

        # Potential issue: There might be duplicates now possible as drop_duplicates over smaller range
        # expected that user db does not have dupliacted
        while (True): # Looping till all rows are completed and processed
            result = cursor.fetchmany(fetchSize)
            result = pd.DataFrame(result)
            if (result.shape[0] == 0):
                break

            # Cleaning Part
            result = result.applymap(clean_token)  # strip tokens and lower case strings
            result = result[~result.iloc[:, 1].isin(INVALID_ENTRY_LIST)]  # clean invalid data
            result = result[~result.iloc[:, 0].isin(INVALID_ENTRY_LIST)]
            result = result.drop_duplicates()  # remove invalid row

            result.iloc[:, 0] = table_name1 + "_" + col_name1 + '_' + result.iloc[:, 0]   # src
            result.insert(1, "rel", edge_entity_feature_val_rel_list[i])  # rel
            result.columns = ["src", "rel", "dst"]
            
            if generate_uuid:
                # convert entity nodes to respective UUIDs
                result['src'] = result['src'].map(entity_mapping)
                if (result['src'].isna().any()):
                    logging.warning(f'Some src column entities did not map in edges_entity_entity')
                    exit(1)

            result.to_csv(output_dir / Path("all_edges.txt"), sep='\t',\
                header=False, index=False, mode='a') # Appending the output to disk
            del result
            rowsCompleted += fetchSize

            if first_pass:
                delta = psutil.virtual_memory().used - mem_copy_used # delta between two virtual_memory(), mem used for curr fetchSize
                est_fetchSize = limit_fetchSize / (delta + 1) * fetchSize # estimated optimal fetchSize for 
                if est_fetchSize > limit_fetchSize:
                    fetchSize = int(limit_fetchSize)
                elif 10000 < est_fetchSize and est_fetchSize <= limit_fetchSize:
                    fetchSize = int(est_fetchSize)
                else:
                    fetchSize = 10000
                first_pass = False  # executing get_fetchSize means we are in 
        logging.info(f'finishing post_processing feature nodes, execution time: {time.time() - start_time2}')
    return 0

def main():
    total_time = time.time()
    parser = set_args()
    args = parser.parse_args()

    ret_data = config_parser_fn(args.config_path)
    db_server = ret_data[0]
    db_name = ret_data[1]
    db_user = ret_data[2]
    db_password = ret_data[3]
    db_host = ret_data[4]
    generate_uuid = ret_data[5]
    entity_queries_list = ret_data[6]
    edge_entity_entity_queries_list = ret_data[7]
    edge_entity_entity_rel_list = ret_data[8]
    edge_entity_feature_val_queries_list = ret_data[9]
    edge_entity_feature_val_rel_list = ret_data[10]

    output_dir = Path(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=output_dir / Path('output.log'), encoding='utf-8', level=logging.INFO) # set filemode='w' if want to start a fresh log file

    try:
        logging.info('Starting a new run!!!\n')
        print('Starting a new run!!!')
        # returning both cnx & cursor because cnx is main object deleting it leads to lose of cursor
        cnx = connect_to_db(db_server, db_name, db_user, db_password, db_host)
        if generate_uuid:
            entity_queries_list = validation_check_entity_queries(entity_queries_list)
        edge_entity_entity_queries_list = validation_check_edge_entity_entity_queries(edge_entity_entity_queries_list)
        edge_entity_feature_val_queries_list = validation_check_edge_entity_feature_val_queries(edge_entity_feature_val_queries_list)
        
        if generate_uuid:
            entity_mapping = entity_node_to_uuids(output_dir, cnx, entity_queries_list, db_server)
        else:
            entity_mapping = None
        
        src_rel_dst = post_processing(output_dir, cnx, edge_entity_entity_queries_list, edge_entity_entity_rel_list,
            edge_entity_feature_val_queries_list, edge_entity_feature_val_rel_list,
             entity_mapping, generate_uuid, db_server)  # this is the pd dataframe
        # convert_to_int() should be next, but we are relying on the Marius' preprocessing module
        cnx.close()
        print('Edge file written to ' + str(output_dir / Path("all_edges.txt")))
        logging.info(f'Total execution time: {time.time()-total_time}\n')
    except Exception as e:
        print(e)
        logging.error(e)
        logging.info(f'Total execution time: {time.time()-total_time}\n')

if __name__ == "__main__":
    main()

