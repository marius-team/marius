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
val split_one = unified.randomSplit(Array(0.9, 0.1))
val train_triplets = split_one(0)
val split_two = split_one(1).randomSplit(Array(0.5, 0.5))
val valid_triplets = split_two(0)
val test_triplets = split_two(1)


create_dataset(unified, train_triplets, valid_triplets, test_triplets, 16, "twitter_16")
