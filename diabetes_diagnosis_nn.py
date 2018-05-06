from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy
import h5py

# random seed for reproducibility
numpy.random.seed(2)

# loading load prima indians diabetes dataset, past 5 years of medical history
dataset = numpy.loadtxt("prima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables, splitting csv data
X = dataset[:, 0:8]
Y = dataset[:, 8]
x_train, x_validation, y_train, y_validation = train_test_split(
    X, Y, test_size=0.30, random_state=5)
# create model, add dense layers one by one specifying activation function
model = Sequential()
# input layer requires input_dim param
model.add(Dense(10, input_dim=8, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
# sigmoid instead of relu for final probability between 0 and 1
model.add(Dense(1, activation='sigmoid'))

# compile the model, adam gradient descent (optimized)
model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=['accuracy'])

# call the function to fit to the data (training the network)
model.fit(x_train, y_train, epochs=1000, batch_size=10,
          validation_data=(x_validation, y_validation))

# evaluate the model

scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


model.save('diabetes_risk_nn.h5')
