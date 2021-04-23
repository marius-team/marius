import spark.implicits._
import org.apache.spark.sql.Dataset

def writePartitionedEntities(dataset: Dataset[Triplet],
                             entitiesPartitionsCount: Int,
                             entitiesOutputPath: String): Unit = {
    dataset.flatMap{case Triplet(subj, rel, obj) => Seq(subj, obj)}.dropDuplicates()
    .repartition(entitiesPartitionsCount).rdd
    .mapPartitionsWithIndex((partition, iterator) => {
        var index = 0
        iterator.map { x => {
        val t = (partition, index, x)
        index = index + 1
        t
        }
        }
      }
    )
    .toDF("partition", "index", "ent")
    .write.partitionBy("partition").mode("overwrite").parquet(entitiesOutputPath)
}

def writeRelations(dataset: Dataset[Triplet],
                   relationOutputPath: String): Unit = {
    dataset.map{case Triplet(subj, rel, obj) => rel}.dropDuplicates().repartition(1).rdd
        .mapPartitionsWithIndex((partition, iterator) => {
            var index = 0
            iterator.map { x => {
            val t = (index, x)
            index = index + 1
            t
            }
            }
          }
        )
        .toDF("index", "relation")
        .write.mode("overwrite").parquet(relationOutputPath)
}

def writePartitionedTriplets(dataset: Dataset[Triplet],
                  relationInputPath: String,
                  entitiesInputPath: String,
                  tripletsOutputPath: String): Unit = {
    val part_ent = spark.read.parquet(entitiesInputPath).toDF("index", "ent", "partition")
    val rel = spark.read.parquet(relationInputPath)

    val part_triplet = dataset
        .join(part_ent, 'subj === 'ent)
        .drop("subj")
        .withColumnRenamed("partition", "part_left")
        .withColumnRenamed("index", "subj")
        .drop("ent")
        .join(part_ent, 'obj === 'ent)
        .drop("obj")
        .withColumnRenamed("index", "obj")
        .withColumnRenamed("partition", "part_right")
        .drop("ent")
        .join(rel, 'rel === 'relation)
        .drop("rel")
        .withColumnRenamed("index", "rel")
        .drop("relation")

    part_triplet.repartition(10, 'part_left, 'part_right).write.partitionBy("part_left", "part_right")
        .mode("overwrite")
        .parquet(tripletsOutputPath)

}



def createPartitionedDatasetWithoutRepartition(dataset: Dataset[Triplet],
                             relationInputPath: String,
                             entitiesInputPath: String,
                             tripletsOutputPath: String): Unit = {

    writePartitionedTriplets(dataset,
      relationInputPath,
      entitiesInputPath,
      tripletsOutputPath)
}

def createPartitionedDataset(unified: Dataset[Triplet],
                             training: Dataset[Triplet],
                             entitiesPartitionsCount: Int,
                             relationOutputPath: String,
                             entitiesOutputPath: String,
                             tripletsOutputPath: String): Unit = {

    writePartitionedEntities(unified, entitiesPartitionsCount, entitiesOutputPath)

    writeRelations(unified, relationOutputPath)

    writePartitionedTriplets(training,
      relationOutputPath,
      entitiesOutputPath,
      tripletsOutputPath)
}

def create_dataset(unified: Dataset[Triplet],
                            train_triplets: Dataset[Triplet],
                            valid_triplets: Dataset[Triplet],
                            test_triplets: Dataset[Triplet],
                            entitiesPartitionsCount: Int,
                            rootOutputPath: String
                            ): Unit = {
    val relationOutputPath = rootOutputPath + "/relations"
    val entitiesOutputPath = rootOutputPath + "/entities"
    val tripletsTrainOutputPath = rootOutputPath + "/train"
    val tripletsValidationOutputPath = rootOutputPath + "/valid"
    val tripletsTestOutputPath = rootOutputPath + "/test"

    val training = train_triplets
    val valid = valid_triplets
    val test = test_triplets

    createPartitionedDataset(unified,
                            training,
                            entitiesPartitionsCount,
                            relationOutputPath,
                            entitiesOutputPath,
                            tripletsTrainOutputPath)

    createPartitionedDatasetWithoutRepartition(valid,
                            relationOutputPath,
                            entitiesOutputPath,
                            tripletsValidationOutputPath)

    createPartitionedDatasetWithoutRepartition(test,
                            relationOutputPath,
                            entitiesOutputPath,
                            tripletsTestOutputPath)
}