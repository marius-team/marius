# Dataset importer

## Installation

You need scala and spark and python

For python, 
```
conda create -n pbg --y python=3.6
conda activate pbg
pip install -r requirements.txt
```

And pip install 

## Documentation
First we need to create parquet files with partitioned entities, relations, and triplets.

Structure of the folder:
/dataset_parquet/entities/partition=0/part1.parquet (index, ent of type string; so it is the name of entity) 
/dataset_parquet/relations/part0.parquet (index, rel)
/dataset_parquet/train/part_left=0/part_right=1/part3.parquet (subj, rel, obj of type int; so it is indexes)

parquet_to_hdf5.py then converts parquet to hdf5 that you can provide to PBG.

The function below creates partitions the triplets using the same partitions as provided entities
```
def createPartitionedDatasetWithoutRepartition(dataset: Dataset[Triplet],
                             relationInputPath: String,
                             entitiesInputPath: String,
                             tripletsOutputPath: String)
```

The function below saves partitioned entities, relations, and partitioned triplets 
```
def createPartitionedDataset(dataset: Dataset[Triplet],
                             entitiesPartitionsCount: Int,
                             relationOutputPath: String,
                             entitiesOutputPath: String,
                             tripletsOutputPath: String)
```

## Usage 

### Freebase

#### Download

You need to download the dataset from https://developers.google.com/freebase
It is 396GB dataset if you download it and uncompress.

#### Generate triplets with spark

Run freebase importer in a spark shell. This takes about 30min.
```
:load triplet.scala
:load create_partitioned_dataset.scala
:load freebase_importer/freebase_importer.scala
```

#### Convert to hdf5 files with python

To generate hdf5 files from partitioned parquet files run this. It takes about 1h
```shell script
python parquet_to_hdf5.py \
--edge_dir_in "hdfs://root/user/r.beaumont/freebase_refactored/train" \
--ent_part_count 4 \
--entity_dir "hdfs://root/user/r.beaumont/freebase_refactored/entities" \
--relation_dir "hdfs://root/user/r.beaumont/freebase_refactored/relations" \
--edge_dir_out "/var/opt/data/user_data/r.beaumont/freebase_train" \
--output_path "/var/opt/data/user_data/r.beaumont/freebase_metadata"
```

```shell script
python parquet_to_hdf5.py \
--edge_dir_in "hdfs://root/user/r.beaumont/freebase_refactored/valid" \
--ent_part_count 4 \
--entity_dir None \
--relation_dir None \
--edge_dir_out "/var/opt/data/user_data/r.beaumont/freebase_valid" \
--output_path None
```

```shell script
python parquet_to_hdf5.py \
--edge_dir_in "hdfs://root/user/r.beaumont/freebase_refactored/test" \
--ent_part_count 4 \
--entity_dir None \
--relation_dir None \
--edge_dir_out "/var/opt/data/user_data/r.beaumont/freebase_test" \
--output_path None
```
