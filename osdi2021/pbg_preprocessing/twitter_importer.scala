import spark.implicits._


def preprocess_twitter(path: String) : Dataset[Triplet] = {
    val twitter = sc.textFile(path)
    val triplets = twitter.map(line => {
    val parts = line.split(" ")
    Triplet(parts(0), "0", parts(1))
    })

    val filteredTriplets = triplets
        .toDF("subj", "rel", "obj")
        .as[Triplet]

    filteredTriplets
}


val unified = preprocess_twitter("twitter_16/twitter-2010.txt").orderBy(rand())
val Array(train_triplets, test)  = unified.randomSplit(Array(0.9, 0.1))
val Array(valid_triplets, test_triplets) = test.randomSplit(Array(0.5, 0.5))

create_dataset(unified, train_triplets, valid_triplets, test_triplets, 16, "twitter_16")
