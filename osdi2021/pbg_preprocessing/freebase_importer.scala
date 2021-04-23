import spark.implicits._


def preprocess_freebase86(path: String) : Dataset[Triplet] = {
    val freebase = sc.textFile(path)
    val triplets = freebase.map(line => {
    val parts = line.split(" ")
    Triplet(parts(0), parts(2), parts(1))
    })

    val filteredTriplets = triplets
        .toDF("subj", "rel", "obj")
        .as[Triplet]

    filteredTriplets
}

def get_freebase86m_unified() : Dataset[Triplet] = {
    val freebase = sc.textFile("freebase_16/*edges.txt")
    val triplets = freebase.map(line => {
    val parts = line.split(" ")
    Triplet(parts(0), parts(2), parts(1))
    })

    val filteredTriplets = triplets
        .toDF("subj", "rel", "obj")
        .as[Triplet]

    filteredTriplets
}

val unified = get_freebase86m_unified()
val train_triplets = preprocess_freebase86("freebase_16/train_edges.txt")
val valid_triplets = preprocess_freebase86("freebase_16/valid_edges.txt")
val test_triplets = preprocess_freebase86("freebase_16/test_edges.txt")


create_dataset(unified, train_triplets, valid_triplets, test_triplets, 16, "freebase_16")
//create_dataset(unified, train_triplets, valid_triplets, test_triplets, 32, "freebase16")
