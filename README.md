## MsFEM-PI-DeepONet

这个代码仓库包含了两部分的内容，nn文件夹下的关于神经网络训练和测试的代码以及experiments文件夹下的关于论文数值实验部分的代码。

### 环境配置

运行程序所需要两个环境，分别是训练神经网络的pytorch环境和运行多尺度有限元的C++环境。我们已经将环境分别上传到了Docker Hub上，用户可以在上面搜索chenjoy/pytorch:nnmsfem和chenjoy/dealii:nnmsfem进行获取。或者直接运行以下代码拉取镜像：

```bash
docker pull chenjoy/pytorch:nnmsfem
docker pull chenjoy/dealii:nnmsfem
```

拉去镜像后创建容器，建议用户将pytorch的容器创建到带有GPU的机器上：

```bash
docker run -it --name pytorch --gpus all --ipc=host chenjoy/pytorch:nnmsfem /bin/bash
docker run -it --name dealii chenjoy/dealii:nnmsfem /bin/bash
```

### 运行网络训练和测试部分代码

将nn文件夹下的代码下载到创建的pytorch容器中，接着需要去Zenodo上下载数据集和模型，下载这个链接 https://doi.org/10.5281/zenodo.19547217 上的nn.zip就是神经网络需要的所有数据集和模型。将其放在对应的路径上即可运行代码，最终的文件结构如下：

```txt
nn/
├── data/ #数据集
├── results/ #完成训练的模型
├── utils/
├── get_model.py #导出ONNX
├── test.py #测试模型
├── train_valid.py #训练模型
```

### 运行数值实验部分代码

在把完成训练的模型导出为ONNX模型后，可以将其部署到多尺度有限元方法中。同样去这个链接 https://doi.org/10.5281/zenodo.19547217 上下载所需的网格文件和配置文件。同样将其放在对应的路径上即可运行代码，最终的文件结果如下（以model1的nnmsfem为例）：

```txt
experiments/
├── model1/ #算例1
    ├── fem/ #FEM-fine
    ├── msfem/ #MsFEM
    ├── nnmsfem/ #MsFEM-PI-DeepONet
    	├── workspace/ #存放网格文件和配置文件
        ├── CMakeLists.txt
        ├── nnmsfem.cc
```

同样以model1的nnmsfem为例运行这个程序，具体代码如下：

```bash
cmake .
make
cd workspace
../nnmsfem fwd.prm
```

