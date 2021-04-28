import spark.implicits._


def preprocess_twitter(path: String) : Dataset[Triplet] = {
    val twitter = sc.textFile(path)
    val triplets = twitter.map(line => {
    val parts = line.split(" ")
    Triplet(parts(0), "0", parts(2))
    })

    val filteredTriplets = triplets
        .toDF("subj", "rel", "obj")
        .as[Triplet]

    filteredTriplets
}


val unified = preprocess_twitter("twitter_16/twitter-2010.txt").orderBy(rand())
val splits = unified.randomSplit(Array(0.9, 0.05, .05), seed = 11L)
val train_triplets = splits(0)
val valid_triplets = splits(1)
val test_triplets = splits(2)


create_dataset(unified, train_triplets, valid_triplets, test_triplets, 16, "twitter_16")
