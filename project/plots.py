import definitions as df
import data_manipulation as dm
import matplotlib.pyplot as plt


def plot_syn(random_trajectories=[]):
    if(len(random_trajectories) == 0):
        random_trajectories = dm.create_synthetic(3)
    print(random_trajectories)
    for trajectory in random_trajectories:
        x_values = [point[0] for point in trajectory]
        y_values = [point[1] for point in trajectory] 
        plt.plot(x_values, y_values)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectories')
    plt.grid(True)
    plt.show()
def plot_syn_norm(data):
    trajectories, replacements = data
    count = 0
    for trajectory, replacement_point in zip(trajectories, replacements):
        print(count)
        x_values = []
        y_values = []
        x_replacements = []
        y_replacements = []
        print(trajectory)
        for point in trajectory:
            if point is None:
                x_replacements.append(replacement_point[0])
                y_replacements.append(replacement_point[1])
            else:
                x_values.append(point[0])
                y_values.append(point[1])

        plt.plot(x_values, y_values, marker='o', linestyle='-', label='Original Points')
        plt.scatter(x_replacements, y_replacements, color='red', marker='x', label='Replaced Points')

        count = count + 1

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectories with Hidden Points Replaced')
    plt.grid(True)
    plt.show()

def plot(data):
    df.plot_points(data, "fig1.png")