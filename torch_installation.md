# How to install Pytorch==0.4.1 with cuda support.

Firstly, you can initialize a new conda environment. Note that you'd better create a environment based on python 3.6.

```
conda create -n hrnet python=3.6
```

Then you can manually install cudatoolkit 9.0.

```
conda install cudatoolkit=9.0 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/linux-64/
```

Then install Pytorch-0.4.1.

```
pip install http://download.pytorch.org/whl/cu90/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
```

Then install other modules like: Cython, opencv-python and so on.