# ACIQ: ANALYTICAL CLIPPING FOR INTEGER QUANTIZATION OF NEURAL NETWORKS
This is complete example for applying Laplace and Gaussian clipping on activations of CNN.

## Dependencies
- python3.x
- [pytorch](<http://www.pytorch.org>)
- [torchvision](<https://github.com/pytorch/vision>) to load the datasets, perform image transforms
- [pandas](<http://pandas.pydata.org/>) for logging to csv
- [bokeh](<http://bokeh.pydata.org>) for training visualization

## Data
- To run this code you need validation set from ILSVRC2012 data
- Configure your dataset path by providing --data "PATH_TO_ILSVRC" or copy ILSVRC dir to ~/datasets/ILSVRC2012.
- To get the ILSVRC2012 data, you should register on their site for access: <http://www.image-net.org/>

## Building cuda kernels for GEMMLOWP
To improve performance GEMMLOWP quantization was implemented in cuda and requires to compile kernels.

- Create virtual environment for python3 and activate:
```
virtualenv --system-site-packages -p python3 venv3
. ./venv3/bin/activate
```
- build kernels
```
cd kernels
./build_all.sh
```

## Prepare setup for Inference
Low precision inference requires to find scale of low precision tensors ahead of time. In order to calculate scale we need to collect statistics of activations for specific topology and dataset.
### Collect statistics
```
python inference-sim -a resnet18 -b 512 --qtype int8 -sm collect
```
Statistics will be saved under ~/asiq_data/statistics folder.
### Run inference experiment
Following command line will evaluate resnet18 with 4bit activations and Laplace clipping
```
python inference-sim -a resnet18 -b 512 --qtype int4 -sm use -th laplace
```

For not clipped version just omit -th or set "-th no"
```
python inference-sim -a resnet18 -b 512 --qtype int4 -sm use
```
