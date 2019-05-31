# FCOS

this is a fork version of fcos which did those improvements based on original implementation:



- trained fcos on kitti etc;
- evaluated fcos speed on real world system.
- tried different backbones;
- further enhancement on anchor-free method such as reduce pixel calculation to speed up.



Here are some result on fcos:

<p align="center">
    <img src="https://s2.ax1x.com/2019/05/31/Vl6Yxe.png" />
    <img src="https://s2.ax1x.com/2019/05/31/Vl6NKH.png" />
    <img src="https://s2.ax1x.com/2019/05/31/Vl6BIP.png" />
</p>





## Install

First install maskrcnn-benchmark:

```
python3 setup.py build develop
```

then install some dependencies for run:

```
pip3 install yacs
pip3 install alfred-py
```



## Demo

Quick demo to visualize the result:

```
wget https://cloudstor.aarnet.edu.au/plus/s/dDeDPBLEAt19Xrl/download -O FCOS_R_50_FPN_1x.pth
python3 demo/fcos_demo.py
```

