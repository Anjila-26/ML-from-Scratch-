from perceptron import Perceptron
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.colors import ListedColormap

def plot_decision(X, y, classifier, resolution = 0.02):

    markers = ['o','s','v','^','<']
    colors = ['red','blue','green','cyan','lightgreen']
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                        np.arange(y_min, y_max, resolution))
    
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap = cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    for i, c in enumerate(np.unique(y)):
        plt.scatter(
            x = X[y == c, 0],
            y = X[y == c, 1],
            alpha= 0.8,
            c = colors[i],
            marker= markers[i],
            label = f'class {c}',
            edgecolors= 'black'
        )

def main():
    st.title("Perceptron From Scratch! with Iris Dataset")
    
    st.sidebar.header('Model Parameters')
    n_iter = st.sidebar.slider('Number of Epochs', min_value=1, max_value=20, value=10)
    learning_rate = st.sidebar.slider('Learning Rate', min_value=0.01, max_value=1.0, value=0.01)

    df = pd.read_csv('iris.csv')
    y = df.iloc[0:100, 4].values
    y = np.where(y == y[0], 0, 1)
    X = df.iloc[0:100, [0,2]].values

    st.set_option('deprecation.showPyplotGlobalUse', False)

    perceptron = Perceptron(learning_rate=learning_rate, epochs=n_iter)
    perceptron.fit(X, y)

    col1 , col2 = st.columns(2)
    # Plot error over epochs
    with col1 : 
        fig, ax = plt.subplots(figsize = (8,6))
        plt.plot(range(1, len(perceptron.errors_) + 1), perceptron.errors_, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Number of Errors')
        plt.title('Error vs. Epochs')
        st.pyplot()

    with col2 :
        fig, ax = plt.subplots(figsize = (8,6))
        plot_decision(X, y, classifier= perceptron)
        plt.xlabel('Sepal Length[cm]')
        plt.ylabel('Petal Length[cm]')
        plt.legend(loc= 'upper left')
        st.pyplot()

if __name__ == "__main__":
    main()