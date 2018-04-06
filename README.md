# MATRIX CAPSULES WITH EM ROUTING: A Simple Implement

This is a simple implement of the model in the paper [MATRIX CAPSULES WITH EM ROUTING](https://openreview.net/pdf?id=HJWLfGWRb). We refer to the following codes:

* https://jhui.github.io/2017/11/14/Matrix-Capsules-with-EM-routing-Capsule-Network/
* https://github.com/shzygmyx/Matrix-Capsules-pytorch
* https://github.com/ducminhkhoi/EM_Capsules

## Requirements

* Python 3.5
* PyTorch 0.3 
* MNIST
* Support Both CPU and GPU 

## Usage

Just use the following command:

```
python3 trainandtest.py -data where_the_mnist_data_be_placed -iter em_routing_iteration -batch batch_size
```

## TODO

* Coordinate Addition
* Transform Share
