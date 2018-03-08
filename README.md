# **Diabetes Neural Network**</h1> 
__**By: Svanik Dani**__


This is a Neural Network that is trained once and then used to predict the chance of a diabetes diagnosis.<br />

More on the data used to train this model: https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes<br />

The dataset itsef: http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data<br />


__**Architecture:<br />**__

Input Layer - 10 nodes - relu activation<br />
Hidden Layer 1 - 50 nodes - relu activation<br />
Hidden Layer 2 - 10 nodes - relu activation<br />
Hidden Layer 3 - 5 nodes - relu activation<br />
Output Layer - 1 node - sigmoid activation<br />

The Neural Net is fedforward and fully connected meaning any node will be connected to all other nodes in the next layer.<br />

__**Activation Function:<br />**__

The activation function used for all node except those was the relu activation function yielding either a 1 or 0<br />
The output layer uses sigmoid to yield a value between 0-1 giving a predicted chance of diagnosis<br />


__**Loss or Cost Function:<br />**__

The loss function used was binary cross entropy, the reasoning behind this is the dataset contains either 1 or 0 for diagnosis of diabete or no diagnosis. Cross-entropy is used to evaluate the difference amongst probability distributions. Cross-entropy is commonly used to quantify the difference between two probability distributions. Usually the "true" distribution (the one that your machine learning algorithm is trying to match) is expressed in terms of 1 or 0 for true or false in classification problems. This is why binary cross entropy was used in the neural network.<br />


__**Optimizer:<br />**__

The optimizer used was Adam, this is an efficient way to apply gradient descent. Gradient descent is an optimization algorithm used to find the values of parameters of a function that minimizes a cost function, in this case the binary cross entropy discussed above. <br />


__**Gradient Descent Procedure:<br />**__

Gradient Descent begins with initial values for the parameter for the function. These could be 0 or and random value(usually a small number). Let us use coefficients as the parameter for this example.<br />

<p align="center">coefficient = 0.0</p><br />

The cost of the coefficients is evaluated by plugging them into the function and calculating the cost. In this case we used binary cross entropy which calculates the difference between predicted and actual. Predicted comes from the neural networks output layer and the actually comes from the dataset.<br />

<p align="center">cost = f(coefficient) or cost = evaluate(f(coefficient))</p><br />

The derivative of the cost is then calculated. Derivative refers to the slope of the function at a given point. The reason the slope is import is so that we know the direction (sign +/-) to move the coefficient values in order to get a lower cost on the next iteration. Iterations are also known as epochs.<br />

<p align="center">delta = derivative(cost)</p><br />

Now that we know the derivative and the sign is negative, the coefficient values can be updated to increase the accuracy. A learning rate parameter (alpha) must be specified that controls how much the coefficients can change on each update. The learning rate is important so that we can pinpoint accuracy and not change weights too drastically. Changing weights by up having to move back in this case positively.<br />

<p align="center">coefficient = coefficient – (alpha * delta)</p><br />

This process is repeated until the cost of the coefficients (cost) is 0.0 or close enough to zero to be good enough.<br />


__**Summary of the Neural Net:<br />**__

The neural network is a fully connected and feedforward neural network with 3 hidden layers. The loss function used in this case was binary cross entropy and gradient descent was used to adjust weight of each nodes connection to optimize the accuracy of the neural network.<br />


## Flow of the program:<br />


__Files:<br />__

There are 2 files one containing the neural network and one for predictions. The predictions file is run and loads the dataset and allows you to enter new data for the neural network to form a prediction. The important file is the neural network file.<br />


__Intro:<br />__

The Neural Network is programmed in python mainly using the keras library. It also uses numpy, sklearn, and h5py. The keras libraya use tensorflow in the backend, google's machine learning library.<br />


__diabetes_diagnosis_nn.py code breakdown:<br />__

The following files import all libraries and dependencies used<br />
```python
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy
import h5py
```

Random seed for reproducibility<br />
```python
numpy.random.seed(2)
```

This loads pima indians diabetes dataset, past 5 years of medical history<br />
```python
dataset = numpy.loadtxt("prima-indians-diabetes.csv", delimiter=",")
```

The data set is split into input (X) and output (Y) variables. X are inputs and Y is the output, then x and y are split into training and testing groups using sklearns built-in train_test_split function. The testing portion will be 30% of the total dataset and the neural network will use the other 70% to train itself. <br /> 
```python
X = dataset[:, 0:8]
Y = dataset[:, 8]
x_train, x_validation, y_train, y_validation = train_test_split(X, Y, test_size=0.30 ,random_state=5)
```

Creating the model(neural network) in a sequential manner using keras. Dense(fully connected) layers are added one by one specifying activation function.<br />
```python
model = Sequential()
model.add(Dense(10, input_dim=8, activation='relu'))  # input layer requires input_dim param
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
```

Sigmoid is used instead of relu because relu yields 1 or 0, and the output needs to be a percentage and sigmoid will yield a float between 0 and 1, which will be the predicted percentage of diagnosis. <br />
```python
model.add(Dense(1, activation='sigmoid'))
```

Compile the neural network using binary cross entropy to calculate loss and adam gradient descent (optimizer) for optimize the metric accuracy.<br />
```python
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
```

Call the function to fit the neural network to the data (training the network) on x_train and y_train the input and output of the training portion of data. The validation of accuracy will be performed using x_test and y_test the testing portion this is done so that the model can be validated on data combinations it is has never seen before. Epoch is an iteration through the training process each epoch consists of two parts. The first is the froward run, the testing data is run through the model and then for each prediction binary cross entropy is performed to calculate error. The second part is optimizing the model using adam by gradient descent to adjust weights based off the results of binary cross entropy. Finally, the testing data is run through to get an accuracy score. It is defined to repeat this 1000 times to fine tune the weights of the model.<br />
```python
model.fit(x_train, y_train, epochs=1000, batch_size=10,validation_data=(x_validation, y_validation))
```

The model is then evaluated on all the data to check its final accuracy. We found that this neural network had an 85% accuracy.<br />
```python
scores = model.evaluate(X, Y)<br />
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
```

The model is then saved so it can later be reloaded into another file for predictions. This was done so the model can we trained just once and then used for predictions later, instead of training before every prediction.<br />
```python
model.save('diabetes_risk_nn.h5')
```

Thank you so much for reading about this neural network and how it was consturcted. Hope it inspires you to create something better!<br/>
**~Svanik Dani**



