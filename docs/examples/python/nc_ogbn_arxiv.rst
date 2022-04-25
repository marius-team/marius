Small Scale Node Classification (OGBN-Arxiv)
---------------------------------------------
OGBN-Arxiv is a built in dataset in Marius for node classification. In this example
we will use the dataset class for OGBN-Arxiv (already defined in Marius) and the 
python APIs to make a node classification example. This example will use GraphSage
and will have two layer encoder. First layer will be the FEATURE layer and the second
will be the GNN layer.

*Example file location: examples/python/ogbn_arxiv_nc.py*

By going through the example we aim you will understand following things:

- How to use Marius' internally defined in dataset to do preprocessing
- How to define a GraphSage based node classification model using python APIs for
  node classification
- How to add reporting metrics for you task
- How to initialize data loading objects for training and evaluation
- And lastly how to do training and evaluation

Note: This is a GPU example and we are setting the device to GPU at the start of the main using the line::

    device = torch.device("cuda")

If you want to run CPU based training please change *cuda* to *cpu*.

1. Create Dataset Class
^^^^^^^^^^^^^^^^^^^^^^^
The dataset in this example is OGBN-Arxiv. For this dataset we already have a preprocessing
dataset class defined in Marius. So we can use that class directly. This will help us
avoid writing a new custom dataset class. To use the dataset class you need to import
it using::

    from marius.tools.preprocess.datasets.ogbn_arxiv import OGBNArxiv

Once the dataset class is imported you can easily do the preprocessing by calling the
download and the preprocess methods::

    dataset_dir = Path("ogbn_arxiv_nc_dataset/")
    dataset = OGBNArxiv(dataset_dir)
    if not (dataset_dir / Path("edges/train_edges.bin")).exists():
        dataset.download()
        dataset.preprocess()

This will return all the necessary data in properly formated .bin files which can
now be used in Marius for training.

2. Create Model
^^^^^^^^^^^^^^^
In this example we are going to create a four layer model using GraphSage. While
defining a model you need to define three things. First is the encoder, second is
a decoder and lastly we need to define the loss function. This will setup the model.
Additionally to get the accuracy metrics we also need to set a reporter. In this
section we will discuss all this things.

**Encoder:**

For node classification we are going to define a four layer encoder. The first layer
will be a FEATURE layer. For the feature layer we do not need to define anything
complicated so we can easily define the feature layer as::

    feature_layer = m.nn.layers.FeatureLayer(dimension=feature_dim, device=device)

The rest three layers are GraphSage layers. There can also be defined simply using
the following code::

    graph_sage_layer1 = m.nn.layers.GraphSageLayer(input_dim=feature_dim,
                                                   output_dim=feature_dim,
                                                   device=device,
                                                   bias=True)

    graph_sage_layer2 = m.nn.layers.GraphSageLayer(input_dim=feature_dim,
                                                   output_dim=feature_dim,
                                                   device=device,
                                                   bias=True)

    graph_sage_layer3 = m.nn.layers.GraphSageLayer(input_dim=feature_dim,
                                                   output_dim=num_classes,
                                                   device=device,
                                                   bias=True)

Notice that the last layer has different activation function then other 2. There are 
other options available for the activation function, please refer to documentation for
more details.

Once we have setup both this layers we can call the encoder method to set up the encoder.
Note that in the code below we are setting the feature layer first then the GraphSage layers::

    encoder = m.encoders.GeneralEncoder(layers=[[feature_layer],
                                                [graph_sage_layer1],
                                                [graph_sage_layer2],
                                                [graph_sage_layer3]])

**Decoder**

Setting up the decoder in this example is simple. All we are doing is setting up
the No Op Node Decoder::

    decoder = m.nn.decoders.node.NoOpNodeDecoder()

**Loss Function**

For the loss function we are using *Cross Entropy* with reduction as *SUM*. We are 
setting it as follows::

    loss = m.nn.CrossEntropyLoss(reduction="sum")

**Reporter**

Lastly we need to set the reporter for getting the results. In this example we are
going to use ``CategoricalMetric`` as our metric::

    reporter = m.report.NodeClassificationReporter()
    reporter.add_metric(m.report.CategoricalAccuracy())

**Defining the Model**

Once all this details are set up properly all we need to do is call the ``Model``
method to initialize the model::

    model = m.nn.Model(encoder, decoder, loss, reporter)

Lastly we are also adding a optimizer to the model and after that we are done with
model creation::

    model.optimizers = [m.nn.AdamOptimizer(model.named_parameters(), lr=.01)]

1. Create Dataloader
^^^^^^^^^^^^^^^^^^^^
The dataloader object is used for setting up the storage layer. We are going to use
``tensor_to_file()`` method for defining the dataloader. This method stores all the data 
in memory.

In this example for training dataloader we first need to setup the storage for
four files which are: ``edges_all``, ``train_nodes``, ``features`` and ``labels``. All
four can be done easily using the following API calls::

    edges_all = m.storage.tensor_from_file(filename=dataset.edge_list_file, shape=[dataset_stats.num_edges, -1], dtype=torch.int32, device=device)
    train_nodes = m.storage.tensor_from_file(filename=dataset.train_nodes_file, shape=[dataset_stats.num_train], dtype=torch.int32, device=device)
    features = m.storage.tensor_from_file(filename=dataset.node_features_file, shape=[dataset_stats.num_nodes, -1], dtype=torch.float32, device=device)
    labels = m.storage.tensor_from_file(filename=dataset.node_labels_file, shape=[dataset_stats.num_nodes], dtype=torch.int32, device=device)

In the examples above we are passing the file which we got from the preprocessor with proper shape.
The details for shape can be fetched from the yaml file retruned from the preprocessor.

In this example we are setting up a 3-hop neighbour sampler and we define this next::

    nbr_sampler_3_hop = m.data.samplers.LayeredNeighborSampler(num_neighbors=[-1, -1, -1])

After defining the 3-hop sampler we can define the dataloader class as follows::

    train_dataloader = m.data.DataLoader(nodes=train_nodes,
                                         edges=edges_all,
                                         node_features=features,
                                         node_labels=labels,
                                         batch_size=1000,
                                         nbr_sampler=nbr_sampler_3_hop,
                                         learning_task="nc",
                                         train=True)

The things that we need to pass into the dataloader definition is all the file objects 
that we defined, the batch size and the neighbour sampler that we want to use.

Similar to the ``train_dataloader``, we also define the ``eval_dataloader``. please
refer to the example for more details. ``eval_dataloader`` definition is similar to the 
train.

4. Train Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Now we have defined both the model and the dataloaders so we can start with the training
task. To train an epoch all we need to do is call the following function::

    def train_epoch(model, dataloader):
        dataloader.initializeBatches()
        while dataloader.hasNextBatch():
            batch = dataloader.getBatch()
            model.train_batch(batch)

This function does the following:

- Initialize the batches for training.
- Load the next batch (if it is there)
- Train the model

5. Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Similar to the training we can do the evaluation using the following function::

    def eval_epoch(model, dataloader, device):
        dataloader.initializeBatches()
        
        while dataloader.hasNextBatch():
            batch = dataloader.getBatch()
            model.evaluate_batch(batch)
        
        model.reporter.report()

Here all we are doing is as follows:

- Initialize the batches for evaluation
- Load the next batch (if it is there)
- Evalutate the batch
- Call the report function on the model to get the metrics

6. Save Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Work in progress - More details later
