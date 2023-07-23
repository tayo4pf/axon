# axon
Neural network from scratch with numpy, and matplotlib for visualization

# Usage
Tested on python 3.10 - the package may not function as intended on other python versions
Pull from github, then create a python 3.10.9 venv and pip install requirements.txt
```bash
$ mkdir <dir>
$ cd <dir>
$ git clone https://github.com/tayo4pf/axon.git
$ python3.10 -m venv <env>
$ pip install -r requirements.txt
```

# API
The API offers 4 modules for use - activation, loss, Network, and optimizer


Activation functions are defined using the module name
```python
import axon

linear = axon.activation.Identity
relu = axon.activation.Relu
leakyrelu = axon.activation.LeakyRelu
sigmoid = axon.activation.Sigmoid
softmax = axon.activation.Softmax
```


Loss functions are similarly defined using the module name
```python
mse = axon.loss.MSE
logistic = axon.loss.logistic
```


Optimizers can be defined using the enum from the Optimizer module
```python
sgd = axon.Optimizer.SGD
sgd_with_momentum = axon.Optimizer.SGD_WM
nag = axon.Optimizer.NAG
ada_grad = axon.Optimizer.AdaGrad
ada_delta = axon.Optimizer.AdaDelta
adam = axon.Optimizer.AdaM
```


Networks can be instantiated with their shape, the activations for each layer, and the loss function for the network
```python
nn = axon.Network([4, 6, 1], [leakyrelu, softmax], logistic)
```
They can then be trained on a dataset with 

`Network.train(data, labels, optimizer, learning_rate, momentum = 0.9, epsilon = 0.01, gamma = 0.8, fresh = True)`

Performance on a test set can then be computed and input values can be predicted with the model
```python
losses = nn.train(data, labels, adam, 0.03)
perf = nn.test(data, labels)
predictions = nn.predict(inputs)
```
Model weights and biases can be visualized
```python
nn.visualize()
```
Models can be saved and loaded from csvs
```python
filename = 'model'
nn.write_to(filename)
copy = axon.Network.read(filename)
```
