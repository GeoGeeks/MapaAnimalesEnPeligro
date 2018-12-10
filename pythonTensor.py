#Imports
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np



#Lectura de path para sacar un csv desde la carpeta donde está el script
def read_data():

    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    path = os.path.join(__location__, 'Rhino.csv')
    data = pd.read_csv(path,engine='python')
    year_data = data["ï»¿year"].values
    pop_data = data["pop"].values
    return year_data, pop_data

#Se divide toda la data en 2 sets, uno de entrenamiento y otro de test, la división está al 10% por la falta de datos en muchos animales
def split_test_train(year, pop):
    year_train, year_test, pop_train, pop_test = train_test_split(year, pop, test_size=0.10)
    return year_train, year_test, pop_train, pop_test


# Función formula de normalización para los array
def normalize(array):
    return (array - array.mean()) / array.std()


def get_model_tensors():
    #Placeholders para X y Y que son año y población
    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    # Parametros para formula de  regresión lineal
    theta1 = tf.Variable(np.random.randn(), name="weight")
    theta0 = tf.Variable(np.random.randn(), name="bias")
    # formula de regresión lineal (Hypothesis = theta0 + theta1 * X)
    x_theta1 = tf.multiply(X, theta1)
    model = tf.add(x_theta1, theta0)
    return X, Y, theta1, theta0, model

    #Funcion optimizadora de costo para gradient descent, la idea es disminuir el costo en cada iteración
def get_cost_optimizer_tensor(Y, model, size, learning_rate):
    cost_function = tf.reduce_sum(tf.pow(model - Y, 2)) / (2 * size)
    gradient_descent = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = gradient_descent.minimize(cost_function)
    return optimizer, cost_function


year, pop = read_data()
X_train, X_test, Y_train, Y_test = split_test_train(year, pop)
originA = X_train
originB = Y_train
X_train = normalize(X_train)
Y_train = normalize(Y_train)
X_test = normalize(X_test)
Y_test = normalize(Y_test)


plt.scatter(X_train, Y_train, label='Test')
plt.draw()
plt.show()


#plt.scatter(size, price)
#plt.show()


# Parametros, rate de aprendizaje y las iteraciones de entreamiento para conseguir resultado bajando costo
learning_rate = 0.1
training_iteration = 200

X, Y, theta1, theta0, model = get_model_tensors()
optimizer, cost_function = get_cost_optimizer_tensor(Y, model, len(X_train), learning_rate)
init = tf.global_variables_initializer()


#Inicio sesión en tensorflow
with tf.Session() as sess:
    sess.run(init)

    display_step = 2
    for iteration in range(training_iteration):
        # optimizador de gradient descent
        sess.run(optimizer, feed_dict={X: X_train, Y: Y_train})

        #Logs por iteración
        if iteration % display_step == 0:
            training_cost = sess.run(cost_function, feed_dict={X: X_train, Y: Y_train})

    tuning_cost = sess.run(cost_function, feed_dict={X: X_train, Y: Y_train})

    #validación del modelo
    testing_cost = sess.run(cost_function, feed_dict={X: X_test, Y: Y_test})

    print
    "Testing data cost:", testing_cost


    def drawLine2P(x, y, xlims):
        xrange = np.arange(xlims[0], xlims[1], 0.1)
        A = np.vstack([x, np.ones(len(x))]).T
        k, b = np.linalg.lstsq(A, y)[0]
        plt.plot(xrange, k * xrange + b, 'k')


    def denormalize(array, second):
        return ((array * second.std()*-1) - second.mean())

    plt.figure()
    plt.plot(X_train, Y_train, 'ro', label='Muestras normalizadas')
    plt.plot(X_test, Y_test, 'go', label='Muestras normalizadas de test')
    lims = [0,2]
    drawLine2P(X_train, sess.run(theta1) * X_train + sess.run(theta0), lims)
    plt.plot(X_train, sess.run(theta1) * X_train + sess.run(theta0), label='linea')
    li = plt.plot(X_train, sess.run(theta1) * X_train + sess.run(theta0), label='linea')
    normalData = li[0].get_xydata()
    counter = 0
    newX = []
    newY = []
    for s in normalData:
        for j in s:
            if(counter == 0):
                newX.append(j)
                counter = 1
            else:
                newY.append(j)
                counter = 0

            print(j, sep=' ')

    print(X_train)
    #print(denormalize(X_train, originA))

    data = np.array(normalData)

    print (normalData)
    print (newX)
    print(denormalize(np.array(newX), originA))
    print(denormalize(np.array(newY), originB))
    plt.legend()
    plt.xlim([-1.5, 2])
    plt.ylim([-1.5, 2])
    plt.yticks(np.arange(-1.5, 2, 0.25))
    plt.xticks(np.arange(-1.5, 2, 0.25))
    plt.show()



    plt.figure()
    plt.plot(year, pop, 'ro', label='Muestras normalizadas')
  #  plt.plot(Xtest2, Ytest2, 'go', label='Muestras normalizadas de test')
    lims = [1970,2022]
    #drawLine2P(denormalize(np.array(newX), originA), denormalize(np.array(newY), originB), lims)
    plt.plot(-1*denormalize(np.array(newX),originA),  -1*denormalize(np.array(newY), originB), label='linea')
    plt.legend()
    plt.show()


