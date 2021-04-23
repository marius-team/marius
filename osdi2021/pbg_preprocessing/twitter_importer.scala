import spark.implicits._


def preprocess_twitter(path: String) : Dataset[Triplet] = {
    val twitter = sc.textFile(path)
    val triplets = twitter.map(line => {
    val parts = line.split(" ")
    Triplet(parts(0), 0, parts(2))
    })

    val filteredTriplets = triplets
        .toDF("subj", "rel", "obj")
        .as[Triplet]

    filteredTriplets
}

def get_twitter_unified() : Dataset[Triplet] = {
    val twitter = sc.textFile("twitter_16/*edges.txt")
    val triplets = twitter.map(line => {
    val parts = line.split(" ")
    Triplet(parts(0), 0, parts(2))
    })

    val filteredTriplets = triplets
        .toDF("subj", "rel", "obj")
        .as[Triplet]

    filteredTriplets
}

val unified = get_twitter_unified()
val train_triplets = preprocess_twitter("twitter_16/train_edges.txt").orderBy(rand())
val valid_triplets = preprocess_twitter("twitter_16/valid_edges.txt")
val test_triplets = preprocess_twitter("twitter_16/test_edges.txt")


create_dataset(unified, train_triplets, valid_triplets, test_triplets, 16, "twitter_16")

