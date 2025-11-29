'''
Original Code Source
Zhu, Weiqiang, and Gregory C Beroza. “PhaseNet: A Deep-Neural-Network-Based Seismic Arrival Time Picking Method.” 
Geophysical Journal International, 2019. https://doi.org/10.1093/gji/ggy423.
'''
import obspy
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import requests
import os, sys

# calling the api
PHASENET_API_URL = "http://127.0.0.1:7860"

def phaseNetPredict(cha, stt, visual, bgmm_stream=None):
    # visual determines if the call is coming from the initial app call or bgmm additional channel call
    if visual:
        st.header("Running PhaseNet")
        st.write("Picking primary and secondary waves and their probability of existance")
    # skipping function if channel+date has already been run
    if os.path.exists(f"input_data/{cha+stt}/phasenet_imageOne.png"):
        return

    # declaring stream
    if visual:
        # pull stream from denoiser
        stream = obspy.read(f"input_data\{cha+stt}\denoised.miniseed")
    elif not visual:
        # pull stream from bgmm
        stream = bgmm_stream

    # extract 3-component data
    stream = stream.sort()
    assert(len(stream) == 3)
    data = []
    for trace in stream:
        data.append(trace.data)
    data = np.array(data).T
    assert(data.shape[-1] == 3)
    data_id = stream[0].get_id()[:-1]
    timestamp = stream[0].stats.starttime.datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    # run the main phasenet function
    req = {"id": [data_id],
        "timestamp": [timestamp],
        "vec": [data.tolist()]}

    # call the api
    resp = requests.post(f'{PHASENET_API_URL}/predict', json=req)
    # confirm a positive result (200)
    if visual:
        st.write(resp)
    
    # reformat picks
    picks = pd.DataFrame(resp.json())
    # check picks were chosen
    if picks.empty:
        if visual:
            st.write("PhaseNet has determined that there are no primary or secondary waves present (and therefore discernible earthquakes)")
            # end the app if nothing was determined to be present
            st.stop()
        # if not visual, bgmm will cycle to the next channel
    if visual:
        st.write(picks)
        picks["phase_time"] = pd.to_datetime(picks["phase_time"])

    # write out chosen phasenet picks
    if visual:
        picks.to_csv(f'input_data\{cha+stt}\picks.csv', index=False)
        # create visual from picks
        fig, ax = plt.subplots(len(stream), 1, figsize=(10, 4))
        for i, tr in enumerate(stream):
            # Plot stream traces (black line)
            label = "stream" if i == 0 else None   # only label once
            ax[i].plot(tr.times(), tr.data, label=label, c="k")

            # Plot picks
            for _, pick in picks.iterrows():
                c = "blue" if pick["phase_type"] == "P" else "red"
                # Only label once (on first subplot)
                if i == 0:
                    label = "p" if pick["phase_type"] == "P" else "s"
                else:
                    label = None
                ax[i].axvline(
                    (pick["phase_time"] - tr.stats.starttime.datetime).total_seconds(),
                    c=c,
                    label=label,
                    alpha=pick["phase_score"]
                )
        # Collect handles/labels from the first axis only
        handles, labels = ax[0].get_legend_handles_labels()
        # Deduplicate by dictionary (preserves order)
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc="upper right")
        plt.tight_layout()
        # saving figure out to input folder
        plt.savefig(f"input_data\{cha+stt}\phasenet_imageOne.png")

    # if call is from bgmm, only the picks are needed (reformatting happens in bgmm)
    if not visual:
        return picks

def phaseNetPredictProb(cha, stt):
    # skipping function if channel+date has already been run
    if os.path.exists(f"input_data/{cha+stt}/phasenet_imageTwo.png"):
        return
    
    # pull stream from denoiser (bgmm doesn't use this function so boolean visual isn't needed)
    stream = obspy.read(f"input_data\{cha+stt}\denoised.miniseed")

    # extract 3-component data
    stream = stream.sort()
    assert(len(stream) == 3)
    data = []
    for trace in stream:
        data.append(trace.data)
    data = np.array(data).T
    assert(data.shape[-1] == 3)
    data_id = stream[0].get_id()[:-1]
    timestamp = stream[0].stats.starttime.datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    # run the main phasenet probability function
    req = {"id": [data_id],
       "timestamp": [timestamp],
       "vec": [data.tolist()]}
    
    # call the api
    resp = requests.post(f'{PHASENET_API_URL}/predict_prob', json=req)
    # confirm a positive result (200)
    st.write(resp)

    # reformat the picks
    picks, preds = resp.json() 
    preds = np.array(preds)

    # create the visual
    plt.figure(figsize=(10,4))
    plt.plot(preds[0, :, 0, 1], label="P", alpha=0.5, color="blue")
    plt.plot(preds[0, :, 0, 2], label="S", alpha=0.5, color="red")
    # saving figure out to input folder
    plt.savefig(f"input_data\{cha+stt}\phasenet_imageTwo.png")