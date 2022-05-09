# Deep Learning Models towards Fashion-MNIST
This repo contains deep learning implementations in pytorc for `Fashion-MNIST dataset`. Fashion-MNIST dataset contains 70,000 fashion imgaes with 10 class labels in total, which is devided into 60,000 and 10,000 examples for a training and test set, respectively. As a benchmarking purpose, MNIST dataset containing hand-written degits has been widely used. Fashion-MNIST was created as traditional MNIST dataset is too simple for benchmarking, For more details of Fashion-MNIST, please visit [the original repo](https://github.com/zalandoresearch/fashion-mnist).

## Models 
The implmented models are as follows.
- Logistic Regression
- Multilayer Perceptron (MLP)
	- Three layers neural networks with leaky relu activation
	- The default of hidden units are [256, 128, 64] (you can specify different configurations in cli).
- Convolutional Neural Networks (CNN)
	- Three convolution blcoks and 2 fully connected layers
	- Each convolution block contains Conv2d followed by batch normalization, leaky reu activation, and max pooling layer.
	- Dropout layer is added to fully connected layers. 
	- Each CNN block dimension is
		- CNN layer 1: input 28 * 28 * 1, output 28 * 28 * 16
		- Max pooling 1: om[it 28 * 28 * 16, output 14 * 14 * 16
		- CNN layer 2: input 14 * 14 * 16, outpu 14 * 14 * 32
		- Max pooling 2: input 14 * 14 * 32, output 7 * 7 * 32
		- CNN layer 3: input 7 * 7 * 32, output 7 * 7 * 64

## Results
| Model | Accuracy |
|-------|----------|
|Logistic Regression|84.39%|
|MLP|88.42%|
|CNN|91.59%|

## Repo Structure
- dataset: contains Fashion-MNIST dataset
- `models.py`: model implementation with pytorch
- `trainer.py`: a training script for the given model
- `test.py`: a testing script fot the trained model

## Replication
### Requirement
To replicate the model development, first you need to install pytorch. 
### Training
Run the following command for training.
- Logistic Regression
```
python trainer.py --model 'logistic regression' --epochs 100 --batch_size 32 --optimizer adam --learning_rate 0.001 --weight_decay 0.0001 --model_save True
```

- MLP
```
python trainer.py --model mlp --epochs 300 --batch_size 64 --optimizer adam --learning_rate 0.0001 --weight_decay 0.00001 --model_save True
```

- CNN
```
python trainer.py --model 'cnn' --epochs 300 --batch_size 64 --optimizer adam --learning_rate 0.0001 --weight_decay 0.00001 --model_save True
```
You can refer to the `trainer.py` to see more options of cli flags.

### Evaluation

- Logistic Regression
```
python test.py --model_path outputs/mnist_logistic\ regression.pt --model_name mlp --batch_size 64
```

- MLP
```
python test.py --model_path outputs/mnist_mlp.pt --model_name mlp --batch_size 64

```

- CNN
```
python test.py --model_path outputs/mnist_cnn.pt --model_name cnn --batch_size 64
```
