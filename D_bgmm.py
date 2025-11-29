'''
Original Code Source
Zhu, Weiqiang, Ian W McBrearty, S. Mostafa Mousavi, William L Ellsworth, and Gregory C Beroza. “Earthquake 
Phase Association Using a Bayesian Gaussian Mixture Model.” Journal of Geophysical Research. Solid Earth 127, 
no. 5 (2022). https://doi.org/10.1029/2021JB023249.
'''
import pandas as pd
from gamma.utils import association, estimate_eps
import numpy as np
import os
from pyproj import Proj
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
from obspy import UTCDateTime
from obspy.clients.fdsn import Client as ClNet
from C_phaseNet import phaseNetPredict
import streamlit as st

# backup function
def inputs(net, cha, stt, starttime, endtime, clNet):
    st.header("Running Bayesian Gaussian Mixture Model (incomplete)")
    st.write("Determining epicenter and magnitude by adding additional channels' PhaseNet primary and secondary waves within the network")
    st.write("NOTICE: The current model doesn't produce output, so this model will run up to gathering primary and secondary waves from all surrounding channels and running example images of the output given a pre-run dataset with different inputs from California data")
    st.subheader("Part 1:")
    st.write("Gathering primary and secondary waves from other channels within network ", net)

    # calling stations
    stations = pd.read_csv("input_data/stations.csv")
    stations_backup = pd.read_csv("input_data/stations_bgmm.csv")
    # this is where the test data needs to go
    picks = pd.read_csv(f"input_data/{cha+stt}/picks.csv", parse_dates=["phase_time"])
    picks_backup =  pd.read_csv("input_data/picks_bgmm.csv", parse_dates=["phase_time"])
    # insert an id column that contains example CI.PASC.00.HNN
    picks.insert(0, "id", net+"."+cha+".00.HNN")
    # for checking if there are more picks we can use
    og_size = len(picks)

    # gathering data from other channels within the network
    cha_temp_used = [cha]
    for index, row in stations.iterrows():
        if row["check_id"].startswith(net):
            cha_temp = row["check_id"][len(net):]
            #st.write(cha_temp)
            if cha_temp not in cha_temp_used:
                try:
                    stream_temp = clNet.get_waveforms(net, cha_temp, "00", "HN?", starttime, endtime)
                except Exception as e:
                    cha_temp_used.append(cha_temp)
                    continue
                if len(stream_temp) != 3:
                    cha_temp_used.append(cha_temp)
                    continue
                local_pick = phaseNetPredict(cha_temp, stt, visual=False, bgmm_stream=stream_temp)
                local_pick.insert(0, "id", net+"."+cha_temp+".00.HNN")
                picks = pd.concat([picks, pd.DataFrame(local_pick)], ignore_index=False)
                cha_temp_used.append(cha_temp)
    
    # if nothing new was added...
    if og_size == len(picks):
        st.write("Additional Stations within the same network didn't provide any additional information")
        st.write("In this case, an example BGMM will be run instead")
        # ... run the example code
        run_path = False
    # if new locations were added so bgmm can triangulate...
    else:
        st.write("Complete list of channels in the network included in the analysis:")
        st.write(cha_temp_used)
        # alter the resulting picks data
        picks.drop(columns=["station_id"], inplace=True)
        picks.rename(columns={"id": "station_id"}, inplace=True)
        # ... run the function on the date's data
        run_path = True
    
    if run_path: # actual date's data
        # read picks
        picks.rename(columns={"station_id": "id", "phase_time": "timestamp", "phase_type": "type", "phase_score": "prob"}, inplace=True)
        picks["timestamp"] = pd.to_datetime(picks["timestamp"])
        st.write("Pick format:")
        st.write(picks)

        # read stations
        stations.rename(columns={"station_id": "id"}, inplace=True)
        # remove the 'check_id' column
        stations.drop(columns=["check_id"], inplace=True)
        st.write("Station format (first 10 rows):")
        st.write(stations.iloc[:10])
    elif not run_path: # example data
        # read picks
        picks = picks_backup
        picks.rename(columns={"station_id": "id", "phase_time": "timestamp", "phase_type": "type", "phase_score": "prob"}, inplace=True)
        st.write("Pick format:")
        st.write(picks)

        ## read stations
        stations = stations_backup
        stations.rename(columns={"index": "id"}, inplace=True)
        st.write("Station format (first 10 rows):")
        st.write(stations.iloc[:10])
    st.subheader("Part 2:")
    st.write("The following images are from the sample data due to the incomplete process model")

# ______________________________________________________________________________________________________________
# the following code is not run in the current model

def bgmm_freqTime(net, cha, stt, starttime, endtime, clNet):
    st.write("Running Bayesian Gaussian Mixture Model")
    stt = str(stt)

#-11
    st.write("-11")
    # st.write(clNet)
    # st.write(starttime)
# create folder path
# delete if statement later
    if not os.path.exists(f"output_data/{cha+stt}"):
        os.makedirs(f"output_data/{cha+stt}")

#-12
    st.write("-12")
# picks

    # calling stations
    stations = pd.read_csv("input_data/stations_copy.csv")
    # how stations would be called ^
    stations_backup = pd.read_csv("input_data/stations_bgmm.csv")
    # how sample is called ^

    # this is where the test data needs to go
    picks = pd.read_csv(f"input_data/{cha+stt}/picks.csv", parse_dates=["phase_time"])
    # how the picks would be called ^
    picks_backup =  pd.read_csv("input_data/picks_bgmm.csv", parse_dates=["phase_time"])
    # how picks is called ^

    # insert an id column that contains example CI.PASC.00.HNN
    picks.insert(0, "id", net+"."+cha+".00.HNN")
    # for checking if there are more picks we can use
    og_size = len(picks)

    #-13
    st.write("-13")
    # get other waveforms in network
    # generating picks from other stations in the same network
    from obspy.clients.fdsn import Client as ClNet
    cha_temp_used = [cha]
    # st.write("Cha called")
    # st.write(cha_temp_used)
    for index, row in stations.iterrows():
        if row["check_id"].startswith(net):
            cha_temp = row["check_id"][len(net):]
            # st.write("Cha temp called")
            # st.write(cha_temp)
            if cha_temp not in cha_temp_used:
                stt = str(stt)
                #print(cha_temp)
                # st.write("cha temp passed if")
                # st.write(cha_temp)
                try:
                    stream_temp = clNet.get_waveforms(net, cha_temp, "00", "HN?", starttime, endtime)
                except Exception as e:
                    #st.write("exception raised on", cha_temp)
                    # st.write(e)
                    cha_temp_used.append(cha_temp)
                    continue
                if len(stream_temp) != 3:
                    cha_temp_used.append(cha_temp)
                    continue
                local_pick = phaseNetPredict(cha_temp, stt, visual=False, bgmm_stream=stream_temp)
                local_pick.insert(0, "id", net+"."+cha_temp+".00.HNN")
                #picks.append(local_pick)
                picks = pd.concat([picks, pd.DataFrame(local_pick)], ignore_index=False)
                #print(local_pick)
                cha_temp_used.append(cha_temp)

    #-14
    st.write("-14")
    # CHECK if og_size == picks.length(): end
    # if the additional stations didn't add any new information, then bgmm can't run due to lack of triangulation
    if og_size == len(picks):
        #st.write
        st.write("Additional Stations within the same network didn't provide any additional information")
        st.write("In this case, an example BGMM will be run instead")
        run_path = False
    else:
        ## read stations
        st.write("Complete list of channels in the network included in the analysis:")
        st.write(cha_temp_used)
        picks.drop(columns=["station_id"], inplace=True)
        picks.rename(columns={"id": "station_id"}, inplace=True)
        # run_path should = True here
        run_path = False

# -16 (og)
    st.write("-16")
    if run_path:
        ## read picks
        picks.rename(columns={"station_id": "id", "phase_time": "timestamp", "phase_type": "type", "phase_score": "prob"}, inplace=True)
        picks["timestamp"] = pd.to_datetime(picks["timestamp"])
        st.write("Pick format -16 og")
        #st.write(picks.iloc[:10])
        st.write(picks)

        ## read stations
        stations.rename(columns={"station_id": "id"}, inplace=True)
        # remove the 'check_id' column
        stations.drop(columns=["check_id"], inplace=True)
        #stations.drop(columns=["dt"], inplace=True)
        #print("Station format:", stations.iloc[:10])
        st.write("Station format -16 og:")
        st.write(stations.iloc[:10])
    elif not run_path:
        # -16 (backup)
        # 4
        ## read picks
        picks = picks_backup
        picks.rename(columns={"station_id": "id", "phase_time": "timestamp", "phase_type": "type", "phase_score": "prob"}, inplace=True)
        st.write("Pick format -16 backup")
        st.write(picks.iloc[:10])

        ## read stations
        #stations = pd.read_csv(station_csv)
        stations = stations_backup
        stations.rename(columns={"index": "id"}, inplace=True)
        #print("Station format:", stations.iloc[:10])
        st.write("Station format -16 backup:")
        st.write(stations.iloc[:10])

    # -16 (both)
    ## Automatic region
    x0 = stations["longitude"].median()
    y0 = stations["latitude"].median()
    xmin = stations["longitude"].min()
    xmax = stations["longitude"].max()
    ymin = stations["latitude"].min()
    ymax = stations["latitude"].max()
    config = {}
    config["center"] = (x0, y0)
    config["xlim_degree"] = (2 * xmin - x0, 2 * xmax - x0)
    config["ylim_degree"] = (2 * ymin - y0, 2 * ymax - y0)

    ## projection to km
    proj = Proj(f"+proj=aeqd +lon_0={config['center'][0]} +lat_0={config['center'][1]} +units=km")
    stations[["x(km)", "y(km)"]] = stations.apply(lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1)
    stations["z(km)"] = stations["elevation_m"].apply(lambda x: -x/1e3)

    ### setting GMMA configs
    config["use_dbscan"] = True

    # standard picks.csv doesn't have a amplitude and does have a score
    config["use_amplitude"] = False

    config["method"] = "BGMM"  

    if config["method"] == "BGMM": ## BayesianGaussianMixture
        config["oversample_factor"] = 5

    # earthquake location
    config["vel"] = {"p": 6.0, "s": 6.0 / 1.75}
    config["dims"] = ['x(km)', 'y(km)', 'z(km)']
    config["x(km)"] = proj(longitude=config["xlim_degree"], latitude=[config["center"][1]] * 2)[0]
    config["y(km)"] = proj(longitude=[config["center"][0]] * 2, latitude=config["ylim_degree"])[1]

    # SCEDC depth range = (0,99)
    # NCEDC depth range = (0,87)
    config["z(km)"] = (0, 100) 

    config["bfgs_bounds"] = (
        (config["x(km)"][0] - 1, config["x(km)"][1] + 1),  # x
        (config["y(km)"][0] - 1, config["y(km)"][1] + 1),  # y
        (0, config["z(km)"][1] + 1),  # z
        (None, None),  # t
    )

    # DBSCAN: 
    ##!!Truncate the picks into segments: change the dbscan_eps to balance speed and event splitting. A larger eps prevent spliting events but can take longer time in the preprocessing step.
    config["dbscan_eps"] = estimate_eps(stations, config["vel"]["p"]) 
    config["dbscan_min_samples"] = 3

    # set number of cpus
    config["ncpu"] = 32

    ##!!Post filtering (indepent of gmm): change these parameters to filter out associted picks with large errors
    config["min_picks_per_eq"] = 5
    config["min_p_picks_per_eq"] = 0
    config["min_s_picks_per_eq"] = 0
    config["max_sigma11"] = 3.0 # second
    config["max_sigma22"] = 1.0 # log10(m/s)
    config["max_sigma12"] = 1.0 # covariance

    # for k, v in config.items():
    #     st.write(f"{k}: {v}")

    # -17
    st.write("-17")
    #stt = str(stt)
    event_idx0 = 0 ## current earthquake index
    assignments = []
    st.write("before association")
    events, assignments = association(picks, stations, config, event_idx0, config["method"])
    event_idx0 += len(events)
    st.write("after association")

    ## create catalog
    events = pd.DataFrame(events)
    #print(events)
    st.write("Events logged by BGMM")
    st.write(events)
    # CHECK if events is empty: exit all bgmm
    events[["longitude","latitude"]] = events.apply(lambda x: pd.Series(proj(longitude=x["x(km)"], latitude=x["y(km)"], inverse=True)), axis=1)
    events["depth_km"] = events["z(km)"]
    if run_path:
        events.to_csv(f"output_data/{cha+stt}/gamma_events.csv", index=False, 
                        float_format="%.3f",
                        date_format='%Y-%m-%dT%H:%M:%S.%f')
    elif not run_path:
        events.to_csv(f"output_data/backup/gamma_events.csv", index=False, 
                        float_format="%.3f",
                        date_format='%Y-%m-%dT%H:%M:%S.%f')

    ## add assignment to picks
    assignments = pd.DataFrame(assignments, columns=["pick_index", "event_index", "gamma_score"])
    picks = picks.join(assignments.set_index("pick_index")).fillna(-1).astype({'event_index': int})
    picks.rename(columns={"id": "station_id", "timestamp": "phase_time", "type": "phase_type", "prob": "phase_score"}, inplace=True)

# # -18
    st.write("-18")
# # to_csv
    if run_path:
        # -18 (og)
        # to_csv
        picks.to_csv(f"output_data/{cha+stt}/gamma_picks.csv", index=False, 
                        date_format='%Y-%m-%dT%H:%M:%S.%f')
        # config export
        with open(f"output_data/{cha+stt}/config.pkl", "wb") as f:
            pickle.dump(config, f)
    elif not run_path:
        # -18 (backup)
        # to_csv
        picks.to_csv("output_data/backup/gamma_picks.csv", index=False, 
                        date_format='%Y-%m-%dT%H:%M:%S.%f')

        # config export
        with open("output_data/backup/config.pkl", "wb") as f:
            pickle.dump(config, f)

    # 7
    # -19 
    st.write("-19")

    # label for legends of visuals
    result_label="GaMMA"
    # works for og and backup bc of earlier result_path call split
    #gamma_events = pd.read_csv(result_path("/gamma_events.csv"), parse_dates=["time"])
    if run_path:
        gamma_events = pd.read_csv(f"output_data/{cha+stt}/gamma_events.csv", parse_dates=["time"])
    elif not run_path:
        gamma_events = pd.read_csv(f"output_data/backup/gamma_events.csv", parse_dates=["time"])

    graphstart = gamma_events["time"].min()
    graphend = gamma_events["time"].max()

    st.write("plotting image freqTime")
    plt.figure()
    plt.hist(gamma_events["time"], range=(graphstart, graphend), bins=24, edgecolor="k", alpha=1.0, linewidth=0.5, label=f"{result_label}: {len(gamma_events['time'])}")
    plt.ylabel("Frequency")
    plt.xlabel("Date")
    plt.gca().autoscale(enable=True, axis='x', tight=True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d:%H'))
    plt.gcf().autofmt_xdate()
    #plt.legend()
    plt.savefig(f"output_data\{cha+stt}\bgmm_freqTime.png")

# END bgmm_freqTime



# # bgmm_lat(cha, st)
def bgmm_lat(cha, stt):
    # get neccessary files
    # gamma and config
    st.write("testing: bgmm_lat run")
    if os.path.exists(f"output_data\{cha+stt}\gamma_events.csv"):
        gamma_events = pd.read_csv(f"output_data\{cha+stt}\gamma_events.csv")
        with open(f"output_data\{cha+stt}\config.pkl", "rb") as f:
            config = pickle.load(f)
    else:
        gamma_events = pd.read_csv("output_data\backup\gamma_events.csv")
        with open(f"output_data\backup\config.pkl", "rb") as f:
            config = pickle.load(f)
    # stations
    stations = pd.read_csv("input_data\stations_copy.csv")
    # result label
    result_label="BGMM"

    fig = plt.figure(figsize=plt.rcParams["figure.figsize"]*np.array([1.5,1]))
    box = dict(boxstyle='round', facecolor='white', alpha=1)
    text_loc = [0.05, 0.92]
    grd = fig.add_gridspec(ncols=2, nrows=2, width_ratios=[1.5, 1], height_ratios=[1,1])
    fig.add_subplot(grd[:, 0])
    plt.plot(gamma_events["longitude"], gamma_events["latitude"], '.',markersize=2, alpha=1.0)
    plt.axis("scaled")
    plt.xlim(np.array(config["xlim_degree"]))
    plt.ylim(np.array(config["ylim_degree"]))
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.gca().set_prop_cycle(None)
    plt.plot([], [], '.', markersize=10, label=f"{result_label}", rasterized=True)
    plt.plot(stations["longitude"], stations["latitude"], 'k^', markersize=5, alpha=0.7, label="Stations")
    plt.legend(loc="lower right")
    plt.text(text_loc[0], text_loc[1], '(i)', horizontalalignment='left', verticalalignment="top", 
             transform=plt.gca().transAxes, fontsize="large", fontweight="normal", bbox=box)

    fig.add_subplot(grd[0, 1])
    plt.plot(gamma_events["longitude"], gamma_events["depth_km"], '.', markersize=2, alpha=1.0, rasterized=True)
    plt.xlim(np.array(config["xlim_degree"])+np.array([0.2,-0.27]))
    plt.ylim(config["z(km)"])
    plt.gca().invert_yaxis()
    plt.xlabel("Longitude")
    plt.ylabel("Depth (km)")
    plt.gca().set_prop_cycle(None)
    plt.plot([], [], '.', markersize=10, label=f"{result_label}")
    plt.legend(loc="lower right")
    plt.text(text_loc[0], text_loc[1], '(ii)', horizontalalignment='left', verticalalignment="top", 
             transform=plt.gca().transAxes, fontsize="large", fontweight="normal", bbox=box)

    fig.add_subplot(grd[1, 1])
    plt.plot(gamma_events["latitude"], gamma_events["depth_km"], '.', markersize=2, alpha=1.0, rasterized=True)
    plt.xlim(np.array(config["ylim_degree"])+np.array([0.2,-0.27]))
    plt.ylim(config["z(km)"])
    plt.gca().invert_yaxis()
    plt.xlabel("Latitude")
    plt.ylabel("Depth (km)")
    plt.gca().set_prop_cycle(None)
    plt.plot([], [], '.', markersize=10, label=f"{result_label}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.text(text_loc[0], text_loc[1], '(iii)', horizontalalignment='left', verticalalignment="top", 
             transform=plt.gca().transAxes, fontsize="large", fontweight="normal", bbox=box)
    # plt.savefig(figure_dir("earthquake_location.png"), bbox_inches="tight", dpi=300)
    # plt.savefig(figure_dir("earthquake_location.pdf"), bbox_inches="tight", dpi=300)
    # return
    #plt.show();
    plt.savefig(f"output_data\{cha+stt}\bgmm_lat.png")
    


# # bgmm_freqMag(cha, st)
def bgmm_freqMag(cha, stt):
    # get neccessary files
    # gamma_events
    st.write("testing: bgmm_freqMag run")
    if os.path.exists(f"output_data\{cha+stt}\gamma_events.csv"):
        gamma_events = pd.read_csv(f"output_data\{cha+stt}\gamma_events.csv")
    else:
        gamma_events = pd.read_csv("output_data\backup\gamma_events.csv")
    # label
    result_label = "BGMM"
    # visual
    range = (-1, gamma_events["magnitude"].max())

    if (gamma_events["magnitude"] != 999).any():
        plt.figure()
        plt.hist(gamma_events["magnitude"], range=range, bins=25, alpha=1.0,  edgecolor="k", linewidth=0.5, label=f"{result_label}: {len(gamma_events['magnitude'])}")
        plt.legend()
        plt.xlim([-1,gamma_events["magnitude"].max()])
        plt.xlabel("Magnitude")
        plt.ylabel("Frequency")
        plt.gca().set_yscale('log')
        plt.savefig(f"output_data\{cha+stt}\bgmm_freqMag.png")

# # def bgmm_mag(cha, st, starttime, endtime)
def bgmm_mag(cha, stt, starttime, endtime):
    # get neccessary files
    # gamma events
    st.write("testing: bgmm_freqMag run")
    if os.path.exists(f"output_data\{cha+stt}\gamma_events.csv"):
        gamma_events = pd.read_csv(f"output_data\{cha+stt}\gamma_events.csv")
    else:
        gamma_events = pd.read_csv("output_data\backup\gamma_events.csv")
    # label
    result_label = "BGMM"
    if (gamma_events["magnitude"] != 999).any():
        plt.figure()
        plt.plot(gamma_events["time"], gamma_events["magnitude"], '.', markersize=5, alpha=1.0, rasterized=True)
        # graphstart=str(starttime)
        # graphend=str(endtime)
        plt.xlim([starttime, endtime])
        ylim = plt.ylim()
        plt.ylabel("Magnitude")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d:%H'))
        plt.gcf().autofmt_xdate()
        plt.gca().set_prop_cycle(None)
        plt.plot([],[], '.', markersize=15, alpha=1.0, label=f"{result_label}: {len(gamma_events['magnitude'])}")
        plt.legend()
        plt.ylim(ylim)
        plt.grid()
        plt.savefig(f"output_data\{cha+stt}\bgmm_mag.png")
