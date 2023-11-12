Custom Dataset Link Prediction
---------------------------------------------
This example will demonstrate how to use Marius Python API to do a Link 
Prediction task on a small scale graph. In this example we will use ogbn-arxiv
graph. Ogbn-arxiv graph is s dataset that is not added in Marius by default. So 
we have to write a custom dataset class before we can do the model training. In 
this example we will explain both the process of defining a custom dataset class
and how to make a model for link prediction using DistMult. Also it would be 
benefical to go through the lp_fb15k_237 example too because both custom and fb15k_237
will have similar model.

*Example file location: examples/python/custom_lp.py*

By going through the example we aim you will understand following things:

- How to use make your own custom dataset class to preprocess data
- How to define a model using the Python APIs and configure it as needed
- How to add different reporting metrics for the accuracy
- How to initialize dataloader objects for training and evaluation
- And lastly how to do training and evaluation

Note: This is a GPU example and we are setting the device to GPU at the start of the
main using the line::

    device = torch.device("cuda")

If you want to run CPU based training please change *cuda* to *cpu*.

1. Create Dataset Class
^^^^^^^^^^^^^^^^^^^^^^^
The dataset orbn-arxiv is a custom dataset so for that we will need to make a new
dataset class for preprocessing. This new dataset which in the example is called
``MYDATASET`` is a child class of the parent class ``LinkPredictionDataset``.
Making a new dataset class requires writing two methods:

- ``download()``: This method downloads the dataset from the source location and
  extracts all the necessary files for preprocessing. In this example we are only
  using the ``raw/edges.csv``. So in the download method we extract it properly.
  We are doing it using the following method::

        self.input_train_edges_file = self.output_directory / Path("edge.csv")
        download = False
        if not self.input_train_edges_file.exists():
            download = True
        if download:
            archive_path = download_url(self.dataset_url, self.output_directory, overwrite)
            extract_file(archive_path, remove_input=False)
            extract_file(self.output_directory / Path("arxiv/raw/edge.csv.gz"))
            (self.output_directory / Path("arxiv/raw/edge.csv")).rename(self.input_train_edges_file)

  All that we are doing here is to download the file, extract the edge.csv file
  and rename it to ensure that we can easily reference it in the preprocess function.
  Note that marius has built in ``download_url`` and ``extract_file`` function if 
  you want to use it.

- ``preprocess()``: The main job of this method is to call the ``convertor()`` function.
  Marius supports two types of convertor. First is a torch based convertor and 
  the other is a spark based convertor. In this example we are only using 
  ``TorchEdgeListConverter``. For more details about both the convertor you can 
  find the class defination at location ``src/python/tools/preprocess/convertors``.
  To use the convertor class we need to define an object of convertor class and 
  after that we can call ``convertor.convert()`` to generate the preprocessed files::

        converter = TorchEdgeListConverter
        splits = [0.8, 0.1, 0.1] # 80%-train, 10%-validation, 10%-test
        converter = converter(
            output_dir=self.output_directory,
            train_edges=self.input_train_edges_file,
            src_column = 0, # col 0 is src and col 1 dst node in input csv
            dst_column = 1,
            delim=",", # CSV delimitor is ","
            splits = splits, # Splitting the data in train, valid and test
            remap_ids=remap_ids # Remapping the raw entity ids into random integers
        )
        return converter.convert()

  As shown above in the code, first we are defining a convertor object. There are
  many options in convertor object and you can find more details in the class 
  definition. In this example we are passing following things:

  - ``output_dir=self.output_directory``: For file output location
  - ``train_edges=self.input_train_edges_file``: Input edges file to preprocess
  - ``columns = [0,1]``: Specifics which columns in edge.csv are source and destination
  - ``delim=","``: What delimitor is used in the csv
  - ``splits = splits``: In this example we only have single edge.csv file so what fractions to split data in train, valid and test
  - ``remap_ids=remap_ids``: Remapping the raw entity ids to a random number

  Lastly once the ``converter.convert()`` is called the input ``edge.csv`` is then 
  converted into ``edges.bin`` file. The file will be located at ``self.output_directory / Path("edge.csv")``.
  And Marius uses this file as input.

Once you have defined the class all you need to do is instansiate the base directory
where you will store all the dataset and preprocessed files. And call the download
and preprocess on the objects. As shown in the code.::

    dataset_dir = Path("ogbn_arxiv_dataset/")
    dataset = MYDATASET(dataset_dir)
    if not (dataset_dir / Path("edges/train_edges.bin")).exists():
        dataset.download()
        dataset.preprocess()

Lastly, note that dataset preprocessing will return a ``dataset.yaml`` file which
is needed for further tasks, so in the example we are reading it after ``preprocess()``.

Once you are done with preprocessing the dataset rest of the steps will be similar
to the lp_fb15k_237 example.

2. Create Model
^^^^^^^^^^^^^^^
Next step is to define a model for the task. In this example we are going to make
a model with *DistMult*. The model is defined in the function ``init_model``. 
There are three steps to defining a model:

1. Defining an encoder: In this example we are defining a single layer encoder.
The layer is an embedding layer::

   embedding_layer = m.nn.layers.EmbeddingLayer(dimension=embedding_dim, 
                                                device=device)
 
To define a model all you need to do is call the ``GeneralEncoder(..)`` method with all
the layers as shown below::

    encoder = m.encoders.GeneralEncoder(layers=[[embedding_layer]])

In this example we are only having a single layer in the encoder but you can have
more than one layer also. (See the node classification example for refer on how to
pass more than one layer to ``GeneralEncoder(..)`` method)

2. Defining a decoder: In this example we are using *DistMult* as our decoder so
we are calling the following method::

    decoder = m.nn.decoders.edge.DistMult(num_relations=num_relations,
                                          embedding_dim=embedding_dim,
                                          use_inverse_relations=True,
                                          device=device,
                                          dtype=dtype,
                                          mode="train")


3. Defining a loss function: We are using *SoftmaxCrossEntropy* in this example. And defining
it is just doing a function call::

    loss = m.nn.SoftmaxCrossEntropy(reduction="sum")

There are many other options available for encoder, decoder and loss functions.
Please refer to the API documentation for more details.

In addition to doing the above three tasks, which defines the model, we also need
to provide details regarding which metrics we want to be reported. This is done through
following code::

    reporter = m.report.LinkPredictionReporter()
    reporter.add_metric(m.report.MeanReciprocalRank())
    reporter.add_metric(m.report.MeanRank())
    reporter.add_metric(m.report.Hitsk(1))
    reporter.add_metric(m.report.Hitsk(10))

Notice that you can add multiple metrics.

Once we have defined the encoder, decoder, loss function and the reporter, we can
create a model object using the following method::

    m.nn.Model(encoder, decoder, loss, reporter)

And now this model can be passed to during training and evaluation.

Lastly if you want to add an optimizer to the function you can do it as follows::

    model.optimizers = [m.nn.AdamOptimizer(model.named_parameters(), lr=.1)]

3. Create Dataloader
^^^^^^^^^^^^^^^^^^^^
After defining the model we need to define two dataloader objects, one for training
and the other for evaluation. Dataloader objects are used to handle all the data
movement required for training. Marius supported different types of storage backends
like complete InMemory, Partition Buffers, Flat_File, etc. Please refer to documentation
and the original paper for more details.

In this example we are using an InMemory storage backend where all the data will reside
in memory. This can be defined using the method ``tensor_to_file()``. Do define 
a dataloader object we need to do 3 things:

- First is a simple method call to define which objects need to be read::

    train_edges = m.storage.tensor_from_file(filename=dataset.train_edges_file, shape=[dataset_stats.num_train, -1], dtype=torch.int32, device=device)
    
- Second for this example we want to use a negative edge sampler so we define it
  as follows::
    
    train_neg_sampler = m.data.samplers.CorruptNodeNegativeSampler(num_chunks=10, num_negatives=500, degree_fraction=0.0, filtered=False)

- And last is to make the data loader object itself which will be used during training
  to fetch the data and process batches::

    train_dataloader = m.data.DataLoader(edges=train_edges,
                                         node_embeddings=embeddings,
                                         batch_size=1000,
                                         neg_sampler=train_neg_sampler,
                                         learning_task="lp",
                                         train=True)

Once done with this we have defined the dataloader for training task. Similarly in the
example we also define a dataloader for evaluation.

4. Train Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Now we have everything available to start the training. For training we run multiple
epochs of training and evaluation in this example.

For training all we need is the following function::
    
    def train_epoch(model, dataloader):
        dataloader.initializeBatches()

        while dataloader.hasNextBatch():
            batch = dataloader.getBatch()
            model.train_batch(batch)
            dataloader.updateEmbeddings(batch)

All we are doing in this function is as follows:

- Initializing the batches before the start of the epoch
- If there is a next batch available we fetch the next batch
- We train the model on the fetched batch
- And we update the embeddings

5. Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Similar to training the evaluation is also pretty simple can be concluded easily
using the following function::

    def eval_epoch(model, dataloader):
        dataloader.initializeBatches()

        while dataloader.hasNextBatch():
            batch = dataloader.getBatch()
            model.evaluate_batch(batch)
        
        model.reporter.report()

The function does the following:

- Initialize the batches before the start of every epoch
- Load if there is a next batch of data available
- Evaluate the batch
- Once all batches are done report the metrics we defined earlier in reporter

6. Save Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Work in progress - More details will be added soon