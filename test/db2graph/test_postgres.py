import os
import psycopg2
import random
import sys
from pathlib import Path
sys.path.append('src/python/tools/db2graph/') # moving to the parent directory
from db2graph import connect_to_db, entity_node_to_uuids, post_processing

class TestConnector():
    database = "postgres"
    user = "postgres"
    password = "postgres"
    host = os.environ.get("POSTGRES_HOST")
    port = os.environ.get("POSTGRES_PORT")

    customer_names = ['sofia', 'lukas', 'rajesh', 'daiyu', 'hina', 'lorenzo',
                        'donghai', 'shuchang', 'johnny']
    country_names = ['spain', 'germany', 'india', 'china', 'japan', 'italy',
                        'china', 'china', 'usa']
    item_names = ['fenugreek', 'soy sauce', 'oregano', 'tomato', 'cumin', 'soy sauce',
                    'eggs', 'onions', 'onions', 'wasabi', 'rice', 'chicken breast',
                    'salmon', 'sourdough bread', 'meatballs', 'root beer', 'croissant',
                    'taco sauce']

    @classmethod
    def set_up(self):
        pass
    
    @classmethod
    def tear_down(self):
        pass

    def fill_db(self):
        """
        Filling the database with data for testing things
        """
        conn = psycopg2.connect(database = self.database,
                                user = self.user,
                                password = self.password,
                                host = self.host,
                                port = self.port)
        cur = conn.cursor()

        # DROP TABLE IF EXISTS
        cur.execute("DROP TABLE IF EXISTS ORDERS;")
        cur.execute("DROP TABLE IF EXISTS CUSTOMERS;")
        conn.commit()

        # Create two tables - First Customers and second Orders
        cur.execute('''CREATE TABLE CUSTOMERS 
                        (ID INT PRIMARY KEY NOT NULL,
                        CUSTOMERNAME TEXT NOT NULL,
                        COUNTRY TEXT NOT NULL,
                        PHONE VARCHAR(10) NOT NULL);''')
        conn.commit()
        cur.execute('''CREATE TABLE ORDERS 
                        (ID INT PRIMARY KEY NOT NULL,
                        CUSTOMERID INT NOT NULL,
                        AMOUNT INT NOT NULL,
                        ITEM TEXT NOT NULL,
                        CONSTRAINT fk_customer
                            FOREIGN KEY(CUSTOMERID) 
                                REFERENCES CUSTOMERS(ID));''')
        conn.commit()

        # Insert some data
        # Inserting Customers
        cur.execute(f"INSERT INTO CUSTOMERS (ID,CUSTOMERNAME,COUNTRY,PHONE) \
            VALUES (1, '{self.customer_names[0]}', '{self.country_names[0]}', '6081237654')")
        cur.execute(f"INSERT INTO CUSTOMERS (ID,CUSTOMERNAME,COUNTRY,PHONE) \
            VALUES (2, '{self.customer_names[1]}', '{self.country_names[1]}', '6721576540')")
        cur.execute(f"INSERT INTO CUSTOMERS (ID,CUSTOMERNAME,COUNTRY,PHONE) \
            VALUES (3, '{self.customer_names[2]}', '{self.country_names[2]}', '5511234567')")
        cur.execute(f"INSERT INTO CUSTOMERS (ID,CUSTOMERNAME,COUNTRY,PHONE) \
            VALUES (4, '{self.customer_names[3]}', '{self.country_names[3]}', '3211248173')")
        cur.execute(f"INSERT INTO CUSTOMERS (ID,CUSTOMERNAME,COUNTRY,PHONE) \
            VALUES (5, '{self.customer_names[4]}', '{self.country_names[4]}', '6667890001')")
        cur.execute(f"INSERT INTO CUSTOMERS (ID,CUSTOMERNAME,COUNTRY,PHONE) \
            VALUES (6, '{self.customer_names[5]}', '{self.country_names[5]}', '6260001111')")
        cur.execute(f"INSERT INTO CUSTOMERS (ID,CUSTOMERNAME,COUNTRY,PHONE) \
            VALUES (7, '{self.customer_names[6]}', '{self.country_names[6]}', '7874561234')")
        cur.execute(f"INSERT INTO CUSTOMERS (ID,CUSTOMERNAME,COUNTRY,PHONE) \
            VALUES (8, '{self.customer_names[7]}', '{self.country_names[7]}', '4041015059')")
        cur.execute(f"INSERT INTO CUSTOMERS (ID,CUSTOMERNAME,COUNTRY,PHONE) \
            VALUES (9, '{self.customer_names[8]}', '{self.country_names[8]}', '5647525398')")
        conn.commit()

        # Inserting Orders
        cur.execute(f"INSERT INTO ORDERS (ID,CUSTOMERID,AMOUNT,ITEM) \
            VALUES (1, 3, 5, '{self.item_names[0]}')")
        cur.execute(f"INSERT INTO ORDERS (ID,CUSTOMERID,AMOUNT,ITEM) \
            VALUES (2, 7, 7, '{self.item_names[1]}')")
        cur.execute(f"INSERT INTO ORDERS (ID,CUSTOMERID,AMOUNT,ITEM) \
            VALUES (3, 6, 2, '{self.item_names[2]}')")
        cur.execute(f"INSERT INTO ORDERS (ID,CUSTOMERID,AMOUNT,ITEM) \
            VALUES (4, 1, 3, '{self.item_names[3]}')")
        cur.execute(f"INSERT INTO ORDERS (ID,CUSTOMERID,AMOUNT,ITEM) \
            VALUES (5, 3, 5, '{self.item_names[4]}')")
        cur.execute(f"INSERT INTO ORDERS (ID,CUSTOMERID,AMOUNT,ITEM) \
            VALUES (6, 5, 7, '{self.item_names[5]}')")
        cur.execute(f"INSERT INTO ORDERS (ID,CUSTOMERID,AMOUNT,ITEM) \
            VALUES (7, 2, 1, '{self.item_names[6]}')")
        cur.execute(f"INSERT INTO ORDERS (ID,CUSTOMERID,AMOUNT,ITEM) \
            VALUES (8, 9, 3, '{self.item_names[7]}')")
        cur.execute(f"INSERT INTO ORDERS (ID,CUSTOMERID,AMOUNT,ITEM) \
            VALUES (9, 4, 3, '{self.item_names[8]}')")
        cur.execute(f"INSERT INTO ORDERS (ID,CUSTOMERID,AMOUNT,ITEM) \
            VALUES (10, 5, 15, '{self.item_names[9]}')")
        cur.execute(f"INSERT INTO ORDERS (ID,CUSTOMERID,AMOUNT,ITEM) \
            VALUES (11, 8, 9, '{self.item_names[10]}')")
        cur.execute(f"INSERT INTO ORDERS (ID,CUSTOMERID,AMOUNT,ITEM) \
            VALUES (12, 4, 12, '{self.item_names[11]}')")
        cur.execute(f"INSERT INTO ORDERS (ID,CUSTOMERID,AMOUNT,ITEM) \
            VALUES (13, 5, 20, '{self.item_names[12]}')")
        cur.execute(f"INSERT INTO ORDERS (ID,CUSTOMERID,AMOUNT,ITEM) \
            VALUES (14, 6, 11, '{self.item_names[13]}')")
        cur.execute(f"INSERT INTO ORDERS (ID,CUSTOMERID,AMOUNT,ITEM) \
            VALUES (15, 2, 8, '{self.item_names[14]}')")
        cur.execute(f"INSERT INTO ORDERS (ID,CUSTOMERID,AMOUNT,ITEM) \
            VALUES (16, 9, 2, '{self.item_names[15]}')")
        cur.execute(f"INSERT INTO ORDERS (ID,CUSTOMERID,AMOUNT,ITEM) \
            VALUES (17, 2, 6, '{self.item_names[16]}')")
        cur.execute(f"INSERT INTO ORDERS (ID,CUSTOMERID,AMOUNT,ITEM) \
            VALUES (18, 1, 4, '{self.item_names[17]}')")
        conn.commit()

        conn.close()
        return

    def test_connect_to_db(self):
        """
        Basic connecter to db test. Just checking if connection established
        and corrected values are fetched
        """
        # Filling database with data for testing
        conn = psycopg2.connect(database = self.database,
                                user = self.user,
                                password = self.password,
                                host = self.host,
                                port = self.port)
        
        # Create table
        cur = conn.cursor()
        cur.execute('''CREATE TABLE COMPANY
            (ID INT PRIMARY KEY     NOT NULL,
            NAME           TEXT    NOT NULL,
            AGE            INT     NOT NULL);''')
        conn.commit()

        # Insert some data
        num_data_to_insert = 5
        self.name = []
        self.age = []
        for i in range(num_data_to_insert):
            self.name.append("name" + str(i))
            self.age.append(random.randint(1, 100))
        
        for i in range(num_data_to_insert):
            cur.execute(f"INSERT INTO COMPANY (ID,NAME,AGE) \
                VALUES ({i}, '{self.name[i]}', {self.age[i]})")
        conn.commit()
        conn.close()

        # Setting the connect function to test
        conn = connect_to_db(db_server = "postgre-sql",
                            db_name = self.database,
                            db_user = self.user,
                            db_password = self.password,
                            db_host = self.host)
        cur = conn.cursor()
        cur.execute("SELECT id, name, age from COMPANY")
        rows = cur.fetchall()
        index = 0
        for row in rows:
            assert(row[0] == index)
            assert(row[1] == self.name[index])
            assert(row[2] == self.age[index])
            index += 1
        conn.close()
    
    def test_entity_node_to_uuids(self):
        """
        Testing entity_node_to_uuids function from db2graph.py
        """
        self.fill_db() # Filling database with data for testing

        # Getting all the inputs for the function
        output_dir = Path("output_dir/")
        output_dir.mkdir(parents=True, exist_ok=True)
        db_server = 'postgre-sql'
        conn = psycopg2.connect(database = self.database,
                                user = self.user,
                                password = self.password,
                                host = self.host,
                                port = self.port)
        entity_queries_list = []
        entity_queries_list.append("SELECT DISTINCT customers.customername from customers ORDER BY customers.customername ASC;")
        entity_queries_list.append("SELECT DISTINCT customers.country from customers ORDER BY customers.country ASC;")
        entity_queries_list.append("SELECT DISTINCT orders.item from orders ORDER BY orders.item ASC;")

        # Testing the function
        entity_mapping = entity_node_to_uuids(output_dir, conn, entity_queries_list, db_server)

        # Asserting the corrections of the output
        custs = list(set(self.customer_names))
        custs.sort()
        custs = ['customers_customername_' + elem for elem in custs]
        counts = list(set(self.country_names))
        counts.sort()
        counts = ['customers_country_' + elem for elem in counts]
        itms = list(set(self.item_names))
        itms.sort()
        itms = ['orders_item_' + elem for elem in itms]
        with open(output_dir / "entity_mapping.txt", "r") as file:
            lines = file.readlines()
            lines = [elem.split('\t')[0] for elem in lines]

            # Checking the elements that we added
            for elem in custs:
                assert(elem in lines)
            for elem in counts:
                assert(elem in lines)
            for elem in itms:
                assert(elem in lines)
        
        return
    
    def test_edges_entity_entity(self):
        """
        Testing edges_entity_entity type of queries which generate edges
        """
        self.fill_db() # Filling database with data for testing
        
        # Getting all the inputs for the function
        output_dir = Path("output_dir_edges_entity_entity/")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        db_server = 'postgre-sql'
        
        conn = psycopg2.connect(database = self.database,
                                user = self.user,
                                password = self.password,
                                host = self.host,
                                port = self.port)
        
        edge_entity_entity_queries_list = []
        edge_entity_entity_queries_list.append("SELECT customers.customername, customers.country FROM customers ORDER BY customers.customername ASC;")
        edge_entity_entity_queries_list.append("SELECT orders.item, customers.country FROM orders, customers WHERE orders.customerid = customers.id ORDER BY orders.item ASC;")
        edge_entity_entity_rel_list = ["lives_in", "ordered_by_people_from_country"]

        edge_entity_feature_val_queries_list = []
        edge_entity_feature_val_rel_list = []

        entity_mapping = None

        generate_uuid = False

        # Testing the function
        post_processing(output_dir,
                        conn,
                        edge_entity_entity_queries_list,
                        edge_entity_entity_rel_list,
                        edge_entity_feature_val_queries_list,
                        edge_entity_feature_val_rel_list,
                        entity_mapping,
                        generate_uuid,
                        db_server)
        
        # Asserting the correctionness of the output
        # Predefined correct output for the input queries
        correct_output = []

        # expected outputs for query 1
        correct_output.append(f"customers_customername_daiyu\tlives_in\tcustomers_country_china\n")
        correct_output.append(f"customers_customername_donghai\tlives_in\tcustomers_country_china\n")
        correct_output.append(f"customers_customername_hina\tlives_in\tcustomers_country_japan\n")
        correct_output.append(f"customers_customername_johnny\tlives_in\tcustomers_country_usa\n")
        correct_output.append(f"customers_customername_lorenzo\tlives_in\tcustomers_country_italy\n")
        correct_output.append(f"customers_customername_lukas\tlives_in\tcustomers_country_germany\n")
        correct_output.append(f"customers_customername_rajesh\tlives_in\tcustomers_country_india\n")
        correct_output.append(f"customers_customername_shuchang\tlives_in\tcustomers_country_china\n")
        correct_output.append(f"customers_customername_sofia\tlives_in\tcustomers_country_spain\n")

        # expected outputs for query 2
        correct_output.append(f"orders_item_chicken breast\tordered_by_people_from_country\tcustomers_country_china\n")
        correct_output.append(f"orders_item_croissant\tordered_by_people_from_country\tcustomers_country_germany\n")
        correct_output.append(f"orders_item_cumin\tordered_by_people_from_country\tcustomers_country_india\n")
        correct_output.append(f"orders_item_eggs\tordered_by_people_from_country\tcustomers_country_germany\n")
        correct_output.append(f"orders_item_fenugreek\tordered_by_people_from_country\tcustomers_country_india\n")
        correct_output.append(f"orders_item_meatballs\tordered_by_people_from_country\tcustomers_country_germany\n")
        correct_output.append(f"orders_item_onions\tordered_by_people_from_country\tcustomers_country_usa\n")
        correct_output.append(f"orders_item_onions\tordered_by_people_from_country\tcustomers_country_china\n")
        correct_output.append(f"orders_item_oregano\tordered_by_people_from_country\tcustomers_country_italy\n")
        correct_output.append(f"orders_item_rice\tordered_by_people_from_country\tcustomers_country_china\n")
        correct_output.append(f"orders_item_root beer\tordered_by_people_from_country\tcustomers_country_usa\n")
        correct_output.append(f"orders_item_salmon\tordered_by_people_from_country\tcustomers_country_japan\n")
        correct_output.append(f"orders_item_sourdough bread\tordered_by_people_from_country\tcustomers_country_italy\n")
        correct_output.append(f"orders_item_soy sauce\tordered_by_people_from_country\tcustomers_country_japan\n")
        correct_output.append(f"orders_item_soy sauce\tordered_by_people_from_country\tcustomers_country_china\n")
        correct_output.append(f"orders_item_taco sauce\tordered_by_people_from_country\tcustomers_country_spain\n")
        correct_output.append(f"orders_item_tomato\tordered_by_people_from_country\tcustomers_country_spain\n")
        correct_output.append(f"orders_item_wasabi\tordered_by_people_from_country\tcustomers_country_japan\n")
        with open(output_dir / "all_edges.txt", "r") as file:
            for line in file:
                assert(line in correct_output)
        
        return
    
    def test_edges_entity_feature_values(self):
        """
        Testing the edges_entity_feature_values function
        """
        self.fill_db() # Filling the database with the test data

        # Getting all the inputs to the function
        output_dir = Path("output_dir_edges_entity_feature_values/")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        db_server = 'postgre-sql'
        
        conn = psycopg2.connect(database = self.database,
                                user = self.user,
                                password = self.password,
                                host = self.host,
                                port = self.port)
        
        edge_entity_entity_queries_list = []
        edge_entity_entity_rel_list = []

        edge_entity_feature_val_queries_list = []
        edge_entity_feature_val_queries_list.append("SELECT orders.item, orders.amount FROM orders ORDER BY orders.item ASC;")
        edge_entity_feature_val_rel_list = ["costs"]

        entity_mapping = None

        generate_uuid = False

        # Testing the function
        post_processing(output_dir,
                        conn,
                        edge_entity_entity_queries_list,
                        edge_entity_entity_rel_list,
                        edge_entity_feature_val_queries_list,
                        edge_entity_feature_val_rel_list,
                        entity_mapping,
                        generate_uuid,
                        db_server)
        
        # Asserting the correctionness of the output
        # Predefined correct output for the input queries
        correct_output = []

        # expected outputs for query
        correct_output.append(f"orders_item_chicken breast\tcosts\t12\n")
        correct_output.append(f"orders_item_croissant\tcosts\t6\n")
        correct_output.append(f"orders_item_cumin\tcosts\t5\n")
        correct_output.append(f"orders_item_eggs\tcosts\t1\n")
        correct_output.append(f"orders_item_fenugreek\tcosts\t5\n")
        correct_output.append(f"orders_item_meatballs\tcosts\t8\n")
        correct_output.append(f"orders_item_onions\tcosts\t3\n")
        correct_output.append(f"orders_item_oregano\tcosts\t2\n")
        correct_output.append(f"orders_item_rice\tcosts\t9\n")
        correct_output.append(f"orders_item_root beer\tcosts\t2\n")
        correct_output.append(f"orders_item_salmon\tcosts\t20\n")
        correct_output.append(f"orders_item_sourdough bread\tcosts\t11\n")
        correct_output.append(f"orders_item_soy sauce\tcosts\t7\n")
        correct_output.append(f"orders_item_taco sauce\tcosts\t4\n")
        correct_output.append(f"orders_item_tomato\tcosts\t3\n")
        correct_output.append(f"orders_item_wasabi\tcosts\t15\n")

        with open(output_dir / "all_edges.txt", "r") as file:
            for line in file:
                assert(line in correct_output)
        
        return