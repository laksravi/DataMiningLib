from matplotlib import pyplot as plt


def plot():
    X = [1,2,3,4,5,6,7,8,9]
    Y = [55.83, 94.1, 93.3, 94.3, 90.3, 83.3, 80.6, 85.3, 89.1]
    plt.plot(X, Y, 'rx')
    plt.plot(X, Y)
    plt.xlabel("Hidden Layers")
    plt.ylabel("Accuracy plot")
    plt.show()

plot()