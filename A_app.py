import streamlit as st
import pandas as pd
import datetime
import numpy as np
import obspy
from obspy import UTCDateTime
from obspy.clients.fdsn import Client as ClNet
import os
from PIL import Image

st.title("Earthquake Finder")
# calls for data
# network
net = st.text_input("Station Network", "CI")
# channel
cha = st.text_input("Station Channel", "PASC")
# date of observation
stt = st.date_input("Date of Observation", 
                    value=None, 
                    min_value=datetime.date(2000,1,1),
                    max_value=datetime.date.today())

# shortening run time
timing = st.checkbox("only running an hour (for testing purposes)", value=False)
# bgmm shortening run time
toggle = st.checkbox("Running Bayesian Gaussian Mixture Model (note: will significantly increase wait time)", value=True)

# prevents running client error
if stt is None:
    st.stop()
# changing date input into first half of UTCDateTime input
stt = stt.strftime("%Y-%m-%d")

# calling in the California stations dataset
stations = pd.read_csv("input_data\stations.csv")
# checking user input is valid
id_check = net+cha
if id_check not in stations["check_id"].values:
    st.warning('Network and Station not in catalog')
    st.stop()

# set time variables
starttime = UTCDateTime(stt+"T06:00:00.008300")
if timing:
    endtime = UTCDateTime(stt+"T06:59:59.998300")
else:
    endtime = UTCDateTime(stt+"T20:59:59.998300")

# connect to client using stations table
center = stations[stations["check_id"]==id_check]["center"].values[0]
# defaults to South California
if center=="BOTH" or center=="SCEDC":
    clNet = ClNet("SCEDC")
else:
    clNet = ClNet("NCEDC")

st.write("You are pulling waveform data from ", net, "network and", cha, "channel which can be pulled from the", clNet.base_url, "catalog")

# get stream from client
stream = clNet.get_waveforms(net, cha, "00", "HN?", starttime, endtime)
# check stream has north/south, east/west, up/down
if len(stream) != 3:
    st.warning("The pulled stream does not have three recorded waveforms")
    st.stop()


# creating input folder path
input_path = f"input_data\{cha+stt}"
if not os.path.exists(input_path):
    os.makedirs(input_path)
# storing raw waveform
stream.write(f"{input_path}\stream.miniseed", format="MSEED")

# visualize the raw stream
stream_image = stream.plot(show=False)
# display in Streamlit
st.pyplot(stream_image)

# visualize the stream statistics (confirm user input)
st.write(stream)

# _______________________________________________________________
# Denoiser
from B_denoiser import denoiser
# call the function
denoiser(net, cha, stt, stream, starttime)
# call the resulting image
denoiser_image = Image.open(input_path+"\denoiser_image.png")
# visualize the image
st.image(denoiser_image)

# _______________________________________________________________
# PhaseNet
from C_phaseNet import phaseNetPredict
# call the function (1: p&s on waveform)
predict_stream = phaseNetPredict(cha, stt, visual=True)
# call the resulting image
predict_image = Image.open(input_path+"\phasenet_imageOne.png")
# visualize the image
st.image(predict_image)

from C_phaseNet import phaseNetPredictProb
# call the function (2: p&s probability)
predictProb_stream = phaseNetPredictProb(cha, stt)
# call the resulting image
predictProb_image = Image.open(input_path+"\phasenet_imageTwo.png")
# visualize the image
st.image(predictProb_image)

# ____________________________________________________________________________
#output_path = f"output_data\{cha+stt}"
# if os.path.exists(output_path):
#     bgmm_freqTime_image = Image.open(output_path+"\bgmm_freqTime.png")
#     st.image(bgmm_freqTime_image)
#     bgmm_lat_image = Image.open(output_path+"\bgmm_lat.png")
#     st.image(bgmm_lat_image)
#     bgmm_freqMag_image = Image.open(output_path+"\bgmm_freqMag.png")
#     st.image(bgmm_freqMag_image)
#     bgmm_mag_image = Image.open(output_path+"\bgmm_mag.png")
#     st.image(bgmm_mag_image)
#     st.stop()
# ____________________________________________________________________________
# BGMM

if toggle:
    # holder function for running first (real) part and second (backup) part
    from D_bgmm import inputs
    # run the first part
    inputs(net, cha, stt, starttime, endtime, clNet)
    # call pre-made example
    output_path = f"output_data/backup"
    bgmm_freqTime_image = Image.open(output_path+"/bgmm_freqTime.png")
    st.image(bgmm_freqTime_image)
    bgmm_lat_image = Image.open(output_path+"/bgmm_lat.png")
    st.image(bgmm_lat_image)
    bgmm_freqMag_image = Image.open(output_path+"/bgmm_freqMag.png")
    st.image(bgmm_freqMag_image)
    bgmm_mag_image = Image.open(output_path+"/bgmm_mag.png")
    st.image(bgmm_mag_image)
    gamma_events = pd.read_csv("output_data/backup/gamma_events.csv")
    st.map(data=gamma_events, latitude="latitude", longitude="longitude",size="magnitude")

    # BGMM REAL
    #output_path = f"output_data/{cha+stt}
    # first function (runs algorithm and visualizes frequency over time)
    #from D_bgmm import bgmm_freqTime
    # call the function
    #bgmm_freqTime(net, cha, stt, starttime, endtime, clNet)
    # call the resulting image
    #bgmm_freqTime_image = Image.open(output_path+"/bgmm_freqTime.png")
    # visualize the image
    #st.image(bgmm_freqTime_image)
    
    # second function (latitude, longitude, and depth with station points)
    #from D_bgmm import bgmm_lat
    # call the function
    #bgmm_lat(cha, stt)
    # call the resulting image
    #bgmm_lat_image = Image.open(output_path+"\bgmm_lat.png")
    # visualize the image
    #st.image(bgmm_lat_image)

    # third function (magnitude frequency)
    #from D_bgmm import bgmm_freqMag
    # call the function
    #bgmm_freqMag(cha, stt)
    # call the resulting image
    #bgmm_freqMag_image = Image.open(output_path+"/bgmm_freqMag.png")
    # visualize the image
    #st.image(bgmm_freqMag_image)

    # fourth function (magnitude over time)
    #from D_bgmm import bgmm_mag
    # call the function
    #bgmm_mag(cha, stt, starttime, endtime)
    # call the resulting image
    #bgmm_mag_image = Image.open(output_path+"/bgmm_mag.png")
    # visualize the image
    #st.image(bgmm_mag_image)

    # fifth function (Streamlit map)
    # checking which gamma_events to use (successful run or example)
    #if os.path.exists(f"output_data\{cha+stt}\gamma_events.csv"):
        #gamma_events = pd.read_csv(f"output_data\{cha+stt}\gamma_events.csv")
    #else:
        #gamma_events = pd.read_csv("output_data\backup\gamma_events.csv")
    # calling the Streamlit map function using the chosen gamma_events
    #st.map(data=gamma_events, latitude="latitude", longitude="longitude",size="magnitude")