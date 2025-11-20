
import numpy as np
import matplotlib.pyplot as plt

def plot_data(data, fig=None, ax=None, stim_scaler=None, resp_scaler=None):
    """ assume the data is a tensor of shape (seq_len), already decoded"""
    #data above 1 is the input, below 1 is the output
    input_data = data[np.where(data>1)]
    output_data = data[np.where(data<1)]
    if stim_scaler is not None:
        input_data = stim_scaler.inverse_transform(input_data)
    if resp_scaler is not None:
        output_data = resp_scaler.inverse_transform(output_data)
    if fig is None and ax is None:
        fig, ax = plt.subplots(2,1)
    ax[0].plot(input_data)
    ax[0].set_title("Input")
    ax[1].plot(output_data)
    ax[1].set_title("Output")
    #ax[1].set_ylim(0,1)
    return fig, ax
