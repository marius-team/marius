#!/bin/sh
systemctl start mysql
mkdir /db2graph_eg
wget -O /db2graph_eg/sakila-db.tar.gz https://downloads.mysql.com/docs/sakila-db.tar.gz
tar -xf /db2graph_eg/sakila-db.tar.gz -C /db2graph_eg/
mysql -u root -p=password < /db2graph_eg/sakila-db/sakila-schema.sql
mysql -u root -p=password < /db2graph_eg/sakila-db/sakila-data.sql
## For creating a new user for accessing the data
mysql -u root -p=password mysql -e "CREATE USER 'sakila_user'@'localhost' IDENTIFIED BY 'sakila_password';"
mysql -u root -p=password mysql -e "GRANT ALL PRIVILEGES ON *.* TO 'sakila_user'@'localhost';"
mysql -u root -p=password mysql -e "FLUSH PRIVILEGES;"
service mysql restart