# Estimating Superpoint Similarities with a Graph Neural Network

This repository contains the code of the graph neural network experiments of my dissertation. The code was tested on Windows und Ubuntu. 

## Quickstart

Before starting the training, you have to download the [S3DIS dataset](http://buildingparser.stanford.edu/dataset.html) and extract it to an arbitrary directory. 
After that, create an environment variable *S3DIS_DIR* which points to the root directory of the dataset (e.g. *S3DIS_DIR = your/path/to/unzipped/files/AlignedVersion*). Create a folder *S3DIS_DIR/data* and copy all AREAs into it. For instance, *S3DIS_DIR/data/Area1* should be valid path.

The next step is to prepare the dataset for the training. Apply the script:

```
python s3dis_prepare.py
```

A directory *./S3DIS_Scenes* will be created where the point clouds and partitions are stored in a .npz fileformat. 
Then, you create the feature vectors for the superpoints by executing the script

```
python create_feature_ds.py --n_cpus 4 --dataset ./S3DIS_Scenes
```

Now you can apply the training on a single or multiple PCs. In order to start the training on a single PC, type the following command: 

```
python train_tf_server_graph.py --ip 127.0.0.1
```

Open another terminal and type: 

```
python train_tf_client_graph.py --ip 127.0.0.1
```

The trained models will be stored in the *./models* directory and the tensorboard logs in the *./logs* directory.

## Create Feature Data Set

```
python create_feature_ds.py -h
```

This will create a path *./s3dis/graphs* if using the full graphs with the S3DIS data set. If using a batch size >0, the subgraphs will be calculated and stored in folder *./s3dis/subgraphs*. Set kfold >= 2 to compute the folds for the cross validation, stored in a directory *./s3dis/folds*.  

The following eigenfeatures will be computed: linearity, planarity, scattering, verticality, eigensum, ansitropy, eigenentropy, omnivariance. These features will be used by the graph neural network.

Define an environment variable *SCANNET_DIR* to use the ScanNet data set. The path *SCANNET_DIR/scans* should be a valid path on your system. After that, you can call:

```
python scannet_prepare.py
```

## Training and Testing

The training and testing procedures can be started by calling the server and the clients (see Quickstart section). If the k fold cross validation should be applied, use the option *--kfold True*. Specify the data set as follows for the usage of the ScanNet data set.

```
python train_tf_server_graph.py --ip 127.0.0.1 --dataset scannet
```

```
python train_tf_client_graph.py --ip 127.0.0.1 --dataset_dir ./scannet/graphs
```

More options of the scripts can be printed with the *-h* option. 

## Graph Neural Network

The graph neural network is specified in the *./policies* folder. The network in [ff_graphnet.py](./policies/ff_graphnet.py) applies the correlation operation to estimate the superpoint similarities. The network in [ff_graphnet.py](./policies/graphnet_head.py) applies the sigmoid function. A certain network can be specified by modifying the [args file](./args_file/super_args_ffgn.json). To do so, change the *policy_path* and *policy_type*. The former one should be changed to *policies.script_name* where script_name can be ff_graphnet or graphnet_head. The latter one is the class name of the neural network, which can be found in the python script of the network. For simplicity, the class name of both exemplary networks is GraphNetLight. 

## Citation

Coming soon...

## Acknowledgements

This project is sponsored by: German Federal Ministry of Education and Research (BMBF) under the project number 13FH022IX6. Project name: Interactive body-near production technology 4.0 (German: Interaktive körpernahe Produktionstechnik 4.0 (iKPT4.0))

![bmbf](figures/bmbflogo.jpg)
