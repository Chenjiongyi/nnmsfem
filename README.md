## MsFEM-PI-DeepONet

This code repository contains two parts: the code for PI-DeepONet training and testing in the nn folder, and the code for the numerical experiments section of the paper in the experiments folder.

### Environment configuration

The program requires two environments: a PyTorch environment for training and a C++ environment for running MsFEM. We have uploaded the environments to Docker Hub, where users can search for chenjoy/pytorch:nnmsfem and chenjoy/dealii:nnmsfem to retrieve them. Alternatively, you can directly run the following code to pull the images:

```bash
docker pull chenjoy/pytorch:nnmsfem
docker pull chenjoy/dealii:nnmsfem
```

After pulling the image, it is recommended that users create the PyTorch container on a machine with a GPU when creating the container:

```bash
docker run -it --name pytorch --gpus all --ipc=host chenjoy/pytorch:nnmsfem /bin/bash
docker run -it --name dealii chenjoy/dealii:nnmsfem /bin/bash
```

### Running the network training and testing code

Download the code in the nn folder into the created PyTorch container. Next, you need to download the dataset and model from Zenodo. Download the nn.zip file from this link: https://doi.org/10.5281/zenodo.19547217. This file contains all the datasets and models needed for the neural network. Simply place it in the corresponding path to run the code. The final file structure is as follows:

```txt
nn/
├── data/				# Dataset
├── results/			# The model that has been trained
├── utils/
├── get_model.py		# Export ONNX
├── test.py				# Test model
├── train_valid.py		# Train model
```

### Running the numerical experiment code

After exporting the trained model as an ONNX model, it can be deployed to the MsFEM. After downloading the code from the experiments folder to the dealii container, download the necessary mesh files and configuration files from this link: https://doi.org/10.5281/zenodo.19547217. Place them in the corresponding paths to run the code. The final file structure is as follows (using nnmsfem for model1 as an example):

```txt
experiments/
├── model1/					# Case 1
    ├── fem/ 				# FEM-fine
    ├── msfem/ 				# MsFEM
    ├── nnmsfem/ 			# MsFEM-PI-DeepONet
    	├── workspace/ 		# Grid file and configuration file
        ├── CMakeLists.txt
        ├── nnmsfem.cc
```

Taking nnmsfem from model1 as an example again, first ensure that it is in the nnmsfem folder, and then run the specific code as follows:

```bash
cmake .
make
cd workspace
../nnmsfem fwd.prm
```

