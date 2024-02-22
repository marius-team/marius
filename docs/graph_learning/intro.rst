Intro to Graph Embeddings
***************************

A brief overview of graph-structured data, graph embeddings, and their applications.

Graph-Structured Data
-----------------------
A graph is a data structure consisting of nodes and edges connecting them. For example, a social media network can be modeled as a graph with users as the nodes and friendships as edges between them. Protein networks can be modeled as a graph, with proteins as the nodes and different edge types specifying the different biological interactions between them. In knowledge graphs, such as Wikidata, nodes represent different real-world concepts and edges the relations between them.

Graph-structured data is different from other common data types such as images or text in that it is non-Euclidian: unlike the grid-like structure of these other data types, graphs have no clear "start" or "end" point and have a complex, arbitrary structure. This allows them to represent complex data in a rich and easily understandable way, but also makes it difficult to apply modern machine learning algorithms on them, which are usually built to handle vectorized data.

Graph Embeddings
-----------------------
Graph embeddings are used to solve our aforementioned problem. The idea is to transform nodes, edges, and other graph features into vector representations, in which each embedding encodes some information about the structure of the graph. For example, if we were to embed the nodes of a graph, we would expect a good embedding output to be one in which the embeddings for nodes which are close together in the graph to also be similar in the embedding vector space.

The purpose of the Marius system is to quickly generate these embeddings for a graph, in which embeddings accurately reflect properties of the graph structure. Using data movement techniques, Marius can generate embeddings for massive graphs with billions of nodes and edges.

Graph Learning Tasks
-----------------------
Graph embeddings make it easy to perform downstream graph analytics tasks. Two of the most common inference tasks are:

**Node Classification:** Often we come across graphs in which some nodes have labels or categories attached to them while others do not. Node classification is the process of predicting these missing labels for unlabeled nodes. An example of using Marius for node classification can be found here.

**Link Prediction:** Link prediction is the process of predicting whether an edge exists between two particular nodes. We can use node and relation embeddings to predict new or missing edges in a graph. For example, in a social media network, link prediction could amount to predicting new friend recommendations. An end-to-end example of using Marius for link prediction can be found here.