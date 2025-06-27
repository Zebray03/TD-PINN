import matplotlib.pyplot as plt


class ResultVisualizer:
    @staticmethod
    def plot_loss_history(loss_history, config):
        plt.figure(figsize=(10, 6))
        plt.semilogy(loss_history['total'], label='Total Loss', alpha=0.8)
        plt.semilogy(loss_history['icbc'], label='IC+BC Loss', alpha=0.8)
        plt.semilogy(loss_history['pde'], label='PDE Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title(f'Training Loss (LR={config.LEARNING_RATE})')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.show()

    @staticmethod
    def plot_3d_solution(solution_data):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(solution_data['X'], solution_data['T'], solution_data['U'],
                        cmap='viridis', rstride=1, cstride=1)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_zlabel("u(x,t)")
        ax.set_title("Burgers Equation Solution")
        plt.show()
