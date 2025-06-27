import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def plot_results(losses, solutions):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    fig, ax = plt.subplots()
    x_plot = np.linspace(-1, 1, 100)
    line, = ax.plot(x_plot, solutions[0])
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.set_title("Burgers Equation Solution")

    def update(frame):
        line.set_ydata(solutions[frame])
        return line,

    ani = FuncAnimation(fig, update, frames=len(solutions), interval=200)
    plt.subplot(1, 2, 2)
    plt.show()
