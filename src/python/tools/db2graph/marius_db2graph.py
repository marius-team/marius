import argparse
import logging
import re
import sys
import time
from pathlib import Path

import mysql.connector
import pandas as pd
import psutil
import psycopg2
from mysql.connector import errorcode
from omegaconf import OmegaConf

INVALID_ENTRY_LIST = ["0", None, "", 0, "not reported", "None", "none"]
FETCH_SIZE = int(1e4)
MAX_FETCH_SIZE = int(1e9)
OUTPUT_FILE_NAME = "edges.txt"


def set_args():
    parser = argparse.ArgumentParser(
        description=(
            "Db2Graph is tool to generate graphs from relational database using SQL queries.                See"
            " documentation docs/db2graph for more details."
        ),
        prog="db2graph",
    )

    parser.add_argument(
        "--config_path",
        metavar="config_path",
        type=str,
        default="",
        help="Path to the config file. See documentation docs/db2graph for more details.",
    )

    parser.add_argument(
        "--output_directory",
        metavar="output_directory",
        type=str,
        default="./",
        help="Directory to put output data and log file. See documentation docs/db2graph for more details.",
    )
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
        - edges_queries_list: list of sql queries to define edges of type entity nodes to entity nodes
            & the names of edges
    """
    input_cfg = None
    input_config_path = Path(config_name).absolute()

    input_cfg = OmegaConf.load(input_config_path)

    # db_server used to distinguish between different databases
    db_server = None
    if "db_server" in input_cfg.keys():
        db_server = input_cfg["db_server"]
    else:
        logging.error("ERROR: db_server is not defined")
        exit(1)

    # db_name is the name of the database to pull the data from
    db_name = None
    if "db_name" in input_cfg.keys():
        db_name = input_cfg["db_name"]
    else:
        logging.error("ERROR: db_name is not defined")
        exit(1)

    # db_user is the user name used to access the database
    db_user = None
    if "db_user" in input_cfg.keys():
        db_user = input_cfg["db_user"]
    else:
        logging.error("ERROR: db_user is not defined")

    # db_password is the password used to access the database
    db_password = None
    if "db_password" in input_cfg.keys():
        db_password = input_cfg["db_password"]
    else:
        logging.error("ERROR: db_password is not defined")

    # db_host is the hostname of the database
    db_host = None
    if "db_host" in input_cfg.keys():
        db_host = input_cfg["db_host"]
    else:
        logging.error("ERROR: db_host is not defined")

    # Getting all edge queries for edge type entity node to entity node
    edges_queries_list = list()
    edge_rel_list = list()
    if "edges_queries" in input_cfg.keys():
        query_filepath = input_cfg["edges_queries"]

        if not Path(query_filepath).exists():
            raise ValueError("{} does not exist".format(str(query_filepath)))

        edge_queries_file = open(query_filepath, "r")
        read_lines = edge_queries_file.readlines()
        for i in range(len(read_lines)):
            read_lines[i] = read_lines[i].strip()
            if read_lines[i] == "":
                logging.error("Error: Empty lines are not allowed in edges_query file. " + "Please remove them")
                exit(1)

            # Removing the last '\n' character
            if read_lines[i][-1] == "\n":
                read_lines[i] = read_lines[i][:-1]

            # Adding the line to rel_list if even else its a query
            if i % 2 == 0:
                edge_rel_list.append(read_lines[i])
            else:
                edges_queries_list.append(read_lines[i])
    else:
        logging.error("ERROR: edges_queries is not defined")
        exit(1)

    return db_server, db_name, db_user, db_password, db_host, edges_queries_list, edge_rel_list


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
    if db_server == "maria-db" or db_server == "my-sql":
        try:
            cnx = mysql.connector.connect(user=db_user, password=db_password, host=db_host, database=db_name)
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                logging.error(f"Incorrect user name or password\n{err}")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                logging.error(f"Non-existing database\n{err}")
            else:
                logging.error(err)

    elif db_server == "postgre-sql":
        try:
            cnx = psycopg2.connect(user=db_user, password=db_password, host=db_host, database=db_name)
        except psycopg2.Error as err:
            logging.error(f"Error\n{err}")

    else:
        logging.error("Other databases are currently not supported.")

    return cnx


def validation_check_edge_entity_entity_queries(edges_queries_list):
    """
    Ensures that the edge_entity_entity_queries are correctly formatted.

    :param edges_queries_list: List of all the queries defining edges from entity nodes to entity nodes
    :return new_query_list: These are updated queries with necessary changes if any
    """
    # Format: SELECT table1_name.col1_name, table2_name.col2_name FROM ____ WHERE ____ (and so on);
    logging.info("\nValidating queries for proper formatting")
    new_query_list = list()
    for q in range(len(edges_queries_list)):
        logging.info(f"Checking query[{q}]")
        qry_split = edges_queries_list[q].strip().split()

        if "AS" in qry_split or "as" in qry_split:
            logging.error("Error: Cannot use AS keyword in query. Please update" + " the query")
            exit(1)

        check_var = qry_split[0].lower()
        if check_var != "select":
            logging.error("Error: Incorrect edge entity node - entity node formatting, " + "not starting with SELECT")
            exit(1)

        check_split = qry_split[1].split(".")
        if len(check_split) != 2:
            logging.error(
                "Error: Incorrect edge entity node - entity node formatting, "
                + "table1_name.col1_name not correctly formatted"
            )
            exit(1)
        if check_split[1][-1] != ",":
            logging.error(
                "Error: Incorrect edge entity node - entity node formatting, "
                + "missing ',' at the end of table1_name.col1_name"
            )
            exit(1)

        check_split = qry_split[2].split(".")
        if len(check_split) != 2:
            logging.error(
                "Error: Incorrect edge entity node - entity node formatting, "
                + "table2_name.col2_name not correctly formatted"
            )
            exit(1)

        check_var = qry_split[3].lower()
        if check_var != "from":
            logging.error(
                "Error: Incorrect edge entity node - entity node formatting, "
                + "extra elements after table2_name.col2_name"
            )
            exit(1)

        new_query_list.append(edges_queries_list[q])

    return new_query_list


def clean_token(token):
    """
    Helper to clean a dataframe, can be used by applying this function to a dataframe

    :param token: elements to clean
    :return token: cleaned token
    """
    token = str(token)
    token = token.strip().strip("\t.'\" ")
    return token.lower()


def get_init_fetch_size():
    """
    In an initial pass, estimates the optimal maximum possible fetch_size
    for given query based on memory usage report of virtual_memory()

    :return limit_fetch_size: the optimal maximum possible fetch_size for database engine
    """
    mem_copy = psutil.virtual_memory()
    mem_copy_used = mem_copy.used
    limit_fetch_size = min(mem_copy.available / 2, MAX_FETCH_SIZE)  # max fetch_size limited to MAX_FETCH_SIZE
    return limit_fetch_size, mem_copy_used


def get_fetch_size(fetch_size, limit_fetch_size, mem_copy_used):
    """
    Calculates the optimal maximum fetch_size based on the current snapshot of virtual_memory()
    Increase fetch_size if the amount of memory used is less than half of machine's total available memory
    The size of fetch_size is limited between 10000 and limit_fetch_size bytes

    :param limit_fetch_size: the optimal maximum possible fetch_size
    :return fetch_size: updated fetch_size passed into database engine
    """
    delta = (
        psutil.virtual_memory().used - mem_copy_used
    )  # delta between two virtual_memory(), i.e. mem used for curr fetch_size
    est_fetch_size = limit_fetch_size / (delta + 1) * fetch_size  # estimated optimal fetch_size
    if est_fetch_size > limit_fetch_size:
        fetch_size = int(limit_fetch_size)
    elif FETCH_SIZE < est_fetch_size and est_fetch_size <= limit_fetch_size:
        fetch_size = int(est_fetch_size)
    else:
        fetch_size = FETCH_SIZE
    return fetch_size


def get_cursor(cnx, db_server, cursor_name):
    """
    Gets the cursor for the database connection

    :param cnx: database connection
    :param db_server: database server
    :param cursor_name: name of the cursor (needed for postgre-sql)
    :return cursor: cursor for database connection
    """
    cursor = []
    if db_server == "maria-db" or db_server == "my-sql":
        cursor = cnx.cursor()
    elif db_server == "postgre-sql":
        cursor = cnx.cursor(name=cursor_name)
    return cursor


def post_processing(output_dir, cnx, edges_queries_list, edge_rel_list, db_server):
    """
    Executes the given queries_list one by one, cleanses the data by removing duplicates,
    then append the entity nodes with tableName_colName which works as Unique Identifier,
    and store the final result in a dataframe/.txt file

    :param output_dir: Directory to put output file
    :param cnx: Cursor object
    :param edges_queries_list: List of all the queries defining edges from entity nodes to entity nodes
    :param edge_rel_list: List of all the relationships defining edges from entity nodes to entity nodes
    :param db_server: database server name
    :return 0: 0 for success, exit code 1 for failure
    """
    if len(edges_queries_list) != len(edge_rel_list):
        logging.error("Number of queries in edges_queries_list must match number of edges in edge_rel_list")
        exit(1)

    open(output_dir / Path(OUTPUT_FILE_NAME), "w").close()  # Clearing the output file
    logging.info("\nProcessing queries to generate edges")

    fetch_size = FETCH_SIZE
    # generating edges entity node to entity nodes
    for i in range(len(edges_queries_list)):
        start_time2 = time.time()
        first_pass = True

        # Executing the query and timing it
        query = edges_queries_list[i]
        cursor_name = "edge_entity_entity_cursor" + str(
            i
        )  # Name imp because: https://www.psycopg.org/docs/usage.html#server-side-cursors
        cursor = get_cursor(cnx, db_server, cursor_name)
        cursor.execute(query)

        # Getting Basic Details
        table_name_list = re.split(" ", query)  # table name of the query to execute
        table_name1 = table_name_list[1].split(".")[0]  # src table
        col_name1 = table_name_list[1].split(".")[1][:-1]  # src column, (note last character ',' is removed)
        table_name2 = table_name_list[2].split(".")[0]  # dst/target table
        col_name2 = table_name_list[2].split(".")[1]  # dst/target column

        # Processing each batch of cursor on client
        rows_completed = 0

        # In an initial sample pass, estimates the optimal maximum possible fetch_size for
        # given query based on memory usage report of virtual_memory()
        # process data with fetch_size=10000, record the amount of memory used,
        # increase fetch_size if the amount of memory used is less than half of machine's total available memory,
        # Note: all unit size are in bytes, fetch_size limited between 10000 and 100000000 bytes
        if first_pass:
            limit_fetch_size, mem_copy_used = get_init_fetch_size()

        # Potential issue: There might be duplicates now possible as drop_duplicates over smaller range
        # expected that user db does not have dupliacted
        while True:  # Looping till all rows are completed and processed
            result = cursor.fetchmany(fetch_size)
            result = pd.DataFrame(result)
            if result.shape[0] == 0:
                break

            # Cleaning Part
            result = result.applymap(clean_token)  # strip tokens and lower case strings
            result = result[~result.iloc[:, 1].isin(INVALID_ENTRY_LIST)]  # clean invalid data
            result = result[~result.iloc[:, 0].isin(INVALID_ENTRY_LIST)]
            result = result.drop_duplicates()  # remove invalid row

            result.iloc[:, 0] = table_name1 + "_" + col_name1 + "_" + result.iloc[:, 0]  # src
            result.iloc[:, 1] = table_name2 + "_" + col_name2 + "_" + result.iloc[:, 1]  # dst/target
            result.insert(1, "rel", edge_rel_list[i])  # rel
            result.columns = ["src", "rel", "dst"]

            # storing the output
            result.to_csv(
                output_dir / Path(OUTPUT_FILE_NAME), sep="\t", header=False, index=False, mode="a"
            )  # Appending the output to disk
            del result
            rows_completed += fetch_size

            # update fetch_size based on current snapshot of the machine's memory usage
            if first_pass:
                fetch_size = get_fetch_size(fetch_size, limit_fetch_size, mem_copy_used)
                first_pass = False
        logging.info(f"Finished processing query[{i}] in {time.time() - start_time2:.3f} seconds")


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
    edges_queries_list = ret_data[5]
    edge_rel_list = ret_data[6]

    output_dir = Path(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=output_dir / Path("marius_db2graph.log"), level=logging.INFO, filemode="w"
    )  # set filemode='w' if want to start a fresh log file
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  # add handler to print to console

    try:
        logging.info(f"\nStarting marius_db2graph conversion tool for config: {args.config_path}")

        cnx = connect_to_db(db_server, db_name, db_user, db_password, db_host)

        # Generating edges
        edges_queries_list = validation_check_edge_entity_entity_queries(edges_queries_list)
        post_processing(output_dir, cnx, edges_queries_list, edge_rel_list, db_server)

        cnx.close()
        logging.info(f"\nTotal execution time: {time.time()-total_time:.3f} seconds")
        logging.info("\nEdge file written to " + str(output_dir / Path(OUTPUT_FILE_NAME)))
    except Exception as e:
        logging.error(e)
        logging.info(f"\nTotal execution time: {time.time()-total_time:.3f} seconds")


if __name__ == "__main__":
    main()
