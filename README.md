# Quickstart

Before starting the training, you have to download the [S3DIS dataset](http://buildingparser.stanford.edu/dataset.html) and extract it to an arbitrary directory. 
After that, create an environment variable *S3DIS_DIR* which points to the root directory of the dataset (e.g. *S3DIS_DIR = your/path/to/unzipped/files/AlignedVersion*). Create a folder *S3DIS_DIR/data* and copy all AREAs into it. For instance, *S3DIS_DIR/data/Area1* should be valid path.

The next step is to prepare the dataset for the training. Apply the script 

*python s3dis_prepare.py*. 

A directory *./S3DIS_Scenes* will be created where the point clouds and partitions are stored in a .npz fileformat. 
Then, you create the feature vectors for the superpoints by executing the script

*python create_feature_ds.py --n_cpus 4 --dataset ./S3DIS_Scenes*

Now you can apply the training on a single or multiple PCs. In order to start the training on a single PC, type the following command: 

*python train_tf_server_graph.py --ip 127.0.0.1*

Open another terminal and type: 

*python train_tf_client_graph.py --ip 127.0.0.1*

The trained models will be stored in the *./models* directory and the tensorboard logs in the *./logs* directory.

# TODOs

* Connect single edges (i.e. nodes with no edges) to the remaining graph
* Store links as dataset?
* Sample yes and no decisions equally
* Compare Edge Convolution with Node Convolution? 