'''
Original Code Source
W. Zhu, S. M. Mousavi and G. C. Beroza, "Seismic Signal Denoising and Decomposition Using Deep Neural Networks," in 
IEEE Transactions on Geoscience and Remote Sensing, vol. 57, no. 11, pp. 9476-9488, Nov. 2019, doi: 10.1109/TGRS.2019.
2926772. keywords: {Noise reduction;Neural networks;Noise measurement;Transforms;Time-domain analysis;Earthquakes;Deep 
learning;Convolutional neural networks;decomposition;deep learning;seismic denoising},
'''
import obspy
from obspy.clients.fdsn import Client
import pandas as pd
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import requests
from obspy import read
from obspy import Stream, Trace, UTCDateTime
import streamlit as st

def denoiser(net, cha, stt, stream, starttime):
    st.header("Running Denoiser")
    st.write("Removing excess noise from the waveform to isolate seismic signals and better predict primary and secondary waves")
    # skipping function if channel+date has already been run
    if os.path.exists(f"input_data/{cha+stt}/denoiser_image.png"):
        return
    
    # calling api
    sys.path.insert(0, os.path.abspath("../"))
    DEEPDENOISER_API_URL = "https://ai4eps-deepdenoiser.hf.space"
    
    # extract 3-component data
    stream = stream.sort()
    data = []
    for trace in stream:
        data.append(trace.data)
    data = np.array(data).T
    assert(data.shape[-1] == 3)
    data_id = stream[0].get_id()[:-1]
    timestamp = stream[0].stats.starttime.datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    # Add some noise
    noisy_data = data + np.random.randn(*data.shape)*np.max(data)/20 

    # run the main denoiser function
    req = {"id": [data_id],
        "timestamp": [timestamp],
        "vec": [noisy_data.tolist()]}

    # call the api
    resp = requests.post(f'{DEEPDENOISER_API_URL}/predict', json=req)
    # confirm a positive result (200)
    st.write(resp)
    # save out denoised data to variable
    denoised_data = np.array(resp.json()["vec"])


    # for outputting denoised data to .mseed
    output_data = denoised_data
    output_data = output_data.squeeze()  # shape becomes (720000, 3)
    sampling_rate = 100.0  # Hz
    traces = []
    channel_names = ["HNE", "HNN", "HNZ"]  
    for i in range(3):
        trace = Trace(data=output_data[:, i])
        trace.stats.station = cha
        trace.stats.network = net
        trace.stats.channel = channel_names[i]
        trace.stats.starttime = starttime
        trace.stats.sampling_rate = sampling_rate
        traces.append(trace)
    # create stream for export
    stream = Stream(traces)
    stream = stream.sort()
    # save to .mseed in channel+date input folder
    stream.write(f"input_data\{cha+stt}\denoised.miniseed", format="MSEED") 

    # creating figure of denoised data
    plt.figure(figsize=(10,4))
    plt.subplot(331)
    plt.plot(data[:,0], 'k', linewidth=0.5, label="E")
    plt.legend()
    plt.title("Raw signal")
    plt.subplot(332)
    plt.plot(noisy_data[:,0], 'k', linewidth=0.5, label="E")
    plt.title("Nosiy signal")
    plt.subplot(333)
    plt.plot(denoised_data[0, :,0], 'k', linewidth=0.5, label="E")
    plt.title("Denoised signal")
    plt.subplot(334)
    plt.plot(data[:,1], 'k', linewidth=0.5, label="N")
    plt.legend()
    plt.subplot(335)
    plt.plot(noisy_data[:,1], 'k', linewidth=0.5, label="N")
    plt.subplot(336)
    plt.plot(denoised_data[0,:,1], 'k', linewidth=0.5, label="N")
    plt.subplot(337)
    plt.plot(data[:,2], 'k', linewidth=0.5, label="Z")
    plt.legend()
    plt.subplot(338)
    plt.plot(noisy_data[:,2], 'k', linewidth=0.5, label="Z")
    plt.subplot(339)
    plt.plot(denoised_data[0,:,2], 'k', linewidth=0.5, label="Z")
    plt.tight_layout()
    # saving figure out to input folder
    plt.savefig(f"input_data\{cha+stt}\denoiser_image.png")
