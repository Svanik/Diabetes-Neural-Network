from keras.models import load_model
import numpy as np

model = load_model('diabetes_risk_nn.h5')


def pickthistodo():
    """Pick what needs to be done next."""
    global x
    dothis = input('''Would you like to enter more data:(y/n )''')
    if dothis == 'y' or dothis == 'y':
        x = 1
    elif dothis == 'n' or dothis == 'N':
        x = -1
    else:
        print("Your input is not valid please input 1, 2, or 3")
        pickthistodo()


x = 1
while x == 1:
    print("Please Enter the Folowing Metrics one at a time")
    a = input("Enter Metric 1: ")
    b = input("Enter Metric 2: ")
    c = input("Enter Metric 3: ")
    d = input("Enter Metric 4: ")
    e = input("Enter Metric 5: ")
    f = input("Enter Metric 6: ")
    g = input("Enter Metric 7: ")
    h = input("Enter Metric 8: ")

    makeprediction = np.array([a, b, c, d, e, f, g, h])
    makeprediction = makeprediction.reshape(1, -1)
    finalprediction = model.predict(makeprediction)
    print(finalprediction)
    pickthistodo()
