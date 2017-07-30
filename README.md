# PyTorch Transfer Learning of VGG19 on Cifar-10 Dataset
Transfer Learning of VGG19 trained using ImageNet an retrained for Cifar-10 Dataset using PyTorch.

## Requirement
**Library** | **Version**
--- | ---
**Python** | **^2.7**
**Keras** | **^2.0.2** 
**Numpy** | **^1.12.0** 
**Pickle** |  *  

## Usage
### Download code:
```sh
git clone https://github.com/eugenelet/Keras-Transfer-Learning-of-VGG19-on-Cifar-10-Dataset

cd Keras-Transfer-Learning-of-VGG19-on-Cifar-10-Dataset
```

### Train cnn:
Batch size: 128

Prediction made on per epoch basis. 

161 epochs takes about 3h on GTX 1080.

#### Retrain Model:
```sh
python main.py
```


## Tensorboard
```sh
tensorboard --logdir=./tensorboard
```

## License
[Apache License 2.0](https://github.com/eugenelet/PyTorch-Transfer-Learning-of-VGG19-for-Cifar-10-Dataset/blob/master/LICENSE)

## Implementation Details
[My Blog](https://embedai.wordpress.com/2017/07/30/transfer-learning-of-vgg19-on-cifar-10-dataset-using-pytorch/)
