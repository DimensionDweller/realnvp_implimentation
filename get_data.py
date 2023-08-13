import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import torch
import matplotlib.pyplot as plt

def plot_normalized_moons(n_samples=30000, noise=.05):
    """
    Plots a scatter plot of normalized moon data.

    Parameters:
    n_samples (int): The number of samples to generate. Default is 30000.
    noise (float): The standard deviation of the Gaussian noise added to the data. Default is 0.05.

    Returns:
    None
    """
    data = datasets.make_moons(n_samples=n_samples, noise=noise)[0].astype(np.float32)

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # converting to torch tensor
    normalized_data = torch.from_numpy(data)

    # plotting it on a scatter
    plt.scatter(normalized_data[:, 0], normalized_data[:, 1], s=10, c='b')
    plt.show()
