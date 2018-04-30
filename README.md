# **Diabetes Neural Network**</h1> 
__**By: Svanik Dani**__


This Neural Network has been trained to predict the probability of getting diabetes.
<br />

Data used to train this model: https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes<br />


__**Architecture:<br />**__

Input Layer:  10 nodes - relu activation<br />
Hidden Layer 1:  50 nodes - relu activation<br />
Hidden Layer 2:  10 nodes - relu activation<br />
Hidden Layer 3:  5 nodes - relu activation<br />
Output Layer: 1 node - sigmoid activation<br />

This Neural Network  is feedforward and fully connected. Every node in a given layer will be connected to all nodes in the next layer.<br />

__**Activation Function:<br />**__

The relu activation function has been used for all nodes except those in the output layer, which is yielding either 1 or 0.<br />
The output layer uses sigmoid activation to yield a value between 0 to 1, which would be the predicted chance of diagnosis.<br />


__**Loss or Cost Function:<br />**__

Binary cross entropy loss function has been used because the output contains either 1 for positive diagnosis or 0 for negative diagnosis. Cross-entropy is used to evaluate the difference in two probabilities. Usually the "true" distribution (the one that neural network is trying to match) is expressed in terms of 1 or 0 for true or false in classification problems. <br />

__**Optimizer:<br />**__

Adam has been used as the optimizer because it is an efficient way to apply gradient descent. Gradient descent is an optimization algorithm used to find the values of the parameters of a function that minimizes a loss function, in this case the binary cross entropy discussed above. <br />


__**Gradient Descent Procedure:<br />**__

Gradient Descent begins with initial values for the parameter of the function. This can be 0 or a random value(usually a small number) between -1 and 1. Let us use coefficients as the parameter for this example.<br />
<p align="center">coefficient = 0.0</p><br />
The loss of the coefficient is evaluated by plugging it into the function and calculating the loss. In this case we used binary-cross-entropy which calculates the difference between predicted and actual. Predicted comes from the neural networks output layer and the actually comes from the dataset.<br />
<p align="center">cost = f(coefficient) or cost = evaluate(f(coefficient))</p><br />
The derivative of the loss is then calculated. Derivative refers to the slope of the function at a given point. The reason the slope is important is to find the direction (sign +/-) to move the coefficient values in order to get a lower loss on the next iteration. Iterations are also known as epochs.<br />
<p align="center">delta = derivative(cost)</p><br />
In our case, the derivative and the sign is negative, hence the coefficient values can be updated accordingly to increase the accuracy. A learning rate parameter (alpha) must be specified that controls how much the coefficients can change on each iteration. The learning rate is important so that we can determine accuracy, progressively with each iteration without changing weights drastically. For this project the learning reat is 0.001.<br />
<p align="center">coefficient = coefficient – (alpha * delta)</p><br />
This process is repeated until the loss of coefficients (cost) is 0.0 or close enough to zero to be good enough.<br />


__**Summary of the Neural Net:<br />**__

The neural network is a fully connected, feedforward, and has three hidden layers. The loss function used in this case was binary cross-entropy and gradient descent was used to adjust weight of each nodes connection to optimize the accuracy of the neural network.<br />


## Flow of the program:<br />


__Files:<br />__

There are two files one containing the neural network and one for predictions. The predictions file is run and loads the dataset and allows you to enter new data for the neural network to form a prediction. 
File(diabetes_diagnosis_nn.py code breakdown).<br />


__Introduction:<br />__

The Neural Network is programmed in python mainly using the keras library. It also uses numpy, sklearn, and h5py. The keras library uses tensorflow in the backend, google's own machine learning library.<br />


__diabetes_diagnosis_nn.py code breakdown:<br />__

The following lines import all libraries and dependencies used.<br />
```python
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy
import h5py
```

Random seed for reproducibility is defined.<br />
```python
numpy.random.seed(2)
```

Loading pima indians diabetes dataset, past 5 years of medical history.<br />
```python
dataset = numpy.loadtxt("prima-indians-diabetes.csv", delimiter=",")
```

The data set is split into input (X) and output (Y). X are inputs and Y is the output, then x and y are split into training and testing groups using sklearns built-in train_test_split function. The testing portion will be 30% of the total dataset and the neural network will use the other 70% to train itself. <br /> 
```python
X = dataset[:, 0:8]
Y = dataset[:, 8]
x_train, x_validation, y_train, y_validation = train_test_split(X, Y, test_size=0.30 ,random_state=5)
```

The model (neural network) is created in a sequential manner using Keras functional API.. Dense(fully connected) layers are added one by one specifying activation function.<br />
```python
model = Sequential()
model.add(Dense(10, input_dim=8, activation='relu'))  # input layer requires input_dim param
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
```

Sigmoid is used instead of relu because relu yields 1 or 0, and the output needs to be a probability and sigmoid will yield a float between 0 and 1, which will be the predicted possibility of a positive diabetes diagnosis. <br />
```python
model.add(Dense(1, activation='sigmoid'))
```

Compile the neural network using binary cross-entropy to calculate loss(loss function) and adam gradient descent (optimizer) to optimize the metric accuracy.<br />
```python
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
```

Call the function to fit the neural network to the data (training the network) on x_train and y_train the input and output of the training portion of data. The validation of accuracy will be performed using x_test and y_test the testing portion this is done so that the model can be validated on data combinations it is has never seen before. Epoch is an iteration through the training process each epoch consists of two parts. The first is the forward pass, the testing data is passed through the model and then for each prediction binary cross entropy is performed to calculate error. The second part is optimizing the model using adam by gradient descent to adjust weights based off the results of binary cross entropy. Finally, the testing data is run through the model to get an accuracy score. It is defined to repeat this 1000 times to fine tune the weights of the model.<br />
```python
model.fit(x_train, y_train, epochs=1000, batch_size=10,validation_data=(x_validation, y_validation))
```

The model is then evaluated on all the test data to check its final accuracy. We found that this neural network has an 85% accuracy.<br />
```python
scores = model.evaluate(X, Y)<br />
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
```

The model is then saved so it can later be reloaded into another file for predictions. This was done so the model can be trained just once and then used for predictions later, instead of training before every prediction.<br />
```python
model.save('diabetes_risk_nn.h5')
```

Thank you so much for reading about this neural network and how it was constructed. Hope it inspires you to create something yourself!<br/>


**~Svanik Dani**






