import main
import io
import numpy as np
import pandas as pd
import pymap3d as pm
import time as tm
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.ticker as ticker
from itertools import chain
from urllib.request import urlopen, Request
from PIL import Image
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import pickle

class Results_for_experiment:
    """
    Store the results of all modes in one experiment, which includes:
    - modes
    - results of each mode: 
        - error: {mode1: {
                    A: [],
                    ...,
                    K: []}
                    
                  mode2: }
        - time: an integer 
        - position: {mode1: {
                    A: np.array([enu_data]),}
    """
    def __init__(self, experiment):

        if experiment==0 or experiment > 3:
            raise ValueError("Experiment number must be between 1 and 3.")
    
        self.experiment = experiment
        self.data = list(map(chr, range(ord('A'), ord('K')+1)))
        #self.data =["L", "M"]

        # dictionary of all modes, each mode contains a dictionary, whose indices are A-K
        self.error = dict()
        self.run_time = 0
        self.pos = dict()
        
        self.modes = ["ls-single", "ls-linear", "ls-combo", "ls-sac", "mle", "ls-sac/mle"]


    def categorize_res(self, results, m, data):

        time = []
        if m not in self.error:
            self.error[m] = {}
        if m not in self.pos:
            self.pos[m] = {}

        for result, d in zip(results, data):
            time[m] += result["time"]
            self.error[m][d] = result["error"]
            self.pos[m][d] = result["position"]

        self.run_time = np.mean(np.array(time))
       

    def processing(self):
        start  = tm.time()

        for m in self.modes:
            print(f"start mode: {m}")
            results = [main.worker(d, self.experiment, m) for d in self.data]
            self.categorize_res(results, m, self.data)

        end = tm.time()
        return f'Processing for experiment {self.experiment} done in {end-start:.4f} mins.'

def plot_culmulative_errors(res_obj, figsize=6):
    """Recreate the original images in the paper, based on https://github.com/JonasBchrt/snapshot-gnss-algorithms"""
    plt.figure(figsize=(figsize, figsize))
    for mode in res_obj.modes:
        # Flatten the dictionary to make it run a bit faster
        all_error_in_mode = list(chain.from_iterable([res_obj.error[mode][key] for key in res_obj.data]))  

        # Sort and normalize the error data for cumulative distribution
        plt.plot(sorted(all_error_in_mode), 
                np.linspace(0, 1, len(all_error_in_mode)), 
                label=mode) 
        
    plt.xlim(0, 200)
    plt.ylim(0, 1)
    plt.grid()
    plt.yticks(np.linspace(0, 1, 11))
    plt.xlabel("Cumulative horizontal error [m]",fontsize=15)
    plt.legend(fontsize=15)
    plt.title(f"Experiment {res_obj.experiment}",fontsize=20)
    plt.show()
    
def get_enu_errors(res_obj, mode, scenario_type):
    """Using mean of ENU to get errors, 
    tight is True if we only get the horizontal error of under range of 200m"""
        # Choose the correct types of scenarios
    if scenario_type == 'static':
        data = res_obj.data[:4]
    elif scenario_type == 'dynamic':
        data = res_obj.data[4:]
    else:
        data = res_obj.data

    all_enu_data = np.concatenate([res_obj.pos[mode][key] for key in data], axis=0)
    all_enu_data = np.where(np.isinf(all_enu_data), np.nan, all_enu_data)
    east_errs, north_errs, up_errs = all_enu_data.T  # transposes the array to split by columns
    all_h_errors = np.array(list(chain.from_iterable([res_obj.error[mode][key] for key in data])))

    return east_errs, north_errs, up_errs, all_h_errors

def generate_statistics(res_obj, modes, scenario_type, tight=True):
    """Generate a DataFrame containing statistics for ENU, 2D, 3D, runtime, and horizontal error under 200m.
        If tight is True, then the errors max and min are limited in 200m range."""

    def find_min_max_std_mean_median(an_array):
        return np.nanmin(an_array), np.nanmax(an_array), np.nanstd(an_array), np.nanmean(an_array), np.nanmedian(an_array)
    
    stats = []   
    for mode in modes:
        east_errs, north_errs, _, all_h_errors = get_enu_errors(res_obj, mode, scenario_type)
        if tight:
            # Filter for only values in [-200, 200]
            east_errs = east_errs[(east_errs > -200) & (east_errs < 200)]
            north_errs = north_errs[(north_errs > -200) & (north_errs < 200)]

        e_min, e_max, e_std, e_mean, e_median = find_min_max_std_mean_median(east_errs)
        n_min, n_max, n_std, n_mean, n_median = find_min_max_std_mean_median(north_errs)
        #u_min, u_max, u_std, u_mean, u_median = find_min_max_std_mean_median(up_errs)

        error_less_200 = ((all_h_errors < 200).sum(axis=0) / len(all_h_errors))*100
        rmse_2d = np.nanmean(all_h_errors[all_h_errors < 200])

        #rmse_3d = np.sqrt(np.nanmean(east_errs**2 + north_errs**2 + up_errs**2))

        stats.append([mode, e_min, n_min, e_max, n_max, e_std, n_std, e_mean, n_mean, 
                      e_median, n_median, rmse_2d, error_less_200, res_obj.run_time])

    columns = ["Mode", "Min E (m)", "Min N (m)", "Max E (m)", "Max N (m)", "SD E", "SD N", "Mean E (m)", "Mean N (m)", 
               "Median E (m)", "Median N (m)", "RMSE 2D", "% Error < 200m", "Mean Runtime (s)"]
    
    # Set format for data in the table
    df = pd.DataFrame(stats, columns=columns)
    format_map = {"% Error < 200m": "{:.0f}",  "Mean Runtime (s)": "{:.2f}",}

    for metric in columns[1:-2]:
            format_map[metric] = "{:.2f}"
        # Formatting outputs for readibility
        # df.style.format(format_map) for newer panda ver
    for key, value in format_map.items():
        df[key] = df[key].apply(value.format)
    return df

def statistics(df, whattoshow="all"):
    """Filter and display statistics from the DataFrame based on whattoshow parameter."""
    
    if whattoshow == "all":
        df_stats = df
    else:
        columns_to_show = ["Mode"] + [col for col in df_stats.columns if whattoshow.lower() in col.lower()]
        df_stats = df[columns_to_show]
    return df_stats

# The next three functions are inspired from https://github.com/agrenier-gnss/MimirAnalyzer/tree/main
def plot_ENU_scenario(res_obj, modes, scenario_type="all", axlim = [-200, 200], figsize=8):
    """Plot ENU errors of all modes in the experiment for static/dynamic scenarios"""
    
    fig, axs = plt.subplots(3, figsize=(figsize,figsize), sharex=True)
    plt.suptitle(f'East/North/Up errors for experiment {res_obj.experiment} in {scenario_type} scenarios', fontsize=20)
    for mode in modes: 
        east_errs, north_errs, up_errs, _ = get_enu_errors(res_obj, mode, scenario_type)
        axs[0].plot(east_errs, label=mode)
        axs[1].plot(north_errs, label=mode)
        axs[2].plot(up_errs, label=mode)

    # Apply limits and labels to all subplots
    labels = ["East [m]","North [m]", "Up [m]"]
    for ax, label in zip(axs, labels):
        ax.set_ylim(axlim[0], axlim[1])
        ax.set_xlim(0, east_errs.size)
        ax.set_ylabel(label,fontsize=15)

    # Custom function to format time in x_aix
    def timeTicks(x, pos):                                                                                                                                                                                                                                                        
        total_seconds = x * 0.012 # Duration of each snapshot 
        return f"{total_seconds:.2f}"  # Formats the string to have no decimal places


    formatter = ticker.FuncFormatter(timeTicks)
    axs[2].xaxis.set_major_formatter(formatter)
    axs[2].legend(loc='upper center', fontsize=15,bbox_to_anchor=(0.5, -0.3), fancybox=True, shadow=True, ncol=6)
    
    plt.xlabel('Duration [s]',fontsize=15)  
    plt.show()

def plot_EN(res_obj, modes, scenario_type="all", axlim = [-200, 200], figsize=8):
    """Plot EN errors of all modes in the experiment for static/dynamic scenarios"""
    from matplotlib.patches import Circle
    # Retrieve a list of colors for plotting points from the default Matplotlib color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # a list of color for drawing points
    fig, axs = plt.subplots(1, figsize=(figsize,figsize))
    plt.suptitle(f'East/North errors for experiment {res_obj.experiment} in {scenario_type} scenarios', fontsize=20)

    for i, mode in enumerate(modes):
        east_errs, north_errs, _, _ = get_enu_errors(res_obj,mode, scenario_type)
        axs.scatter(east_errs, north_errs, label=mode, color=colors[i % len(colors)], s=6, zorder=3)
    
    if axlim != [-200, 200]:
        # Adding a circle to show inliers within 200 meters
        circle = Circle((0, 0), 200, color='blue', fill=False, linestyle='--', linewidth=5, alpha=0.5, zorder=2)
        axs.add_patch(circle)
    
    # Fomratting the axes
    axs.grid(zorder=0) # zorder=0: grid below data points
    axs.axis('square')
    axs.set_xlim(axlim[0], axlim[1])
    axs.set_ylim(axlim[0], axlim[1])
    axs.set_xlabel('East [m]',fontsize=15)
    axs.set_ylabel('North [m]',fontsize=15)
    fig.tight_layout()
    axs.yaxis.set_major_formatter('{x:.0f}')
    axs.xaxis.set_major_formatter('{x:.0f}')
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),fontsize=15, fancybox=True, shadow=True, ncol=6)
    plt.show()

def plot_EN_boxplot(res_obj, modes, axlim = [-200,200], scenario_type="all", figsize=(12, 6)):
    """Plot box plots for East and North errors of all modes in the experiment for static/dynamic scenarios, each mode with East and North errors side by side."""
    
    east_errors, north_errors, labels = [], [], []
    # Collect data
    for mode in modes:
        east_errs, north_errs, _, _ = get_enu_errors(res_obj, mode, scenario_type)
        # Filter for only values in [-200, 200]
        east_errs = east_errs[(east_errs > -200) & (east_errs < 200)]
        north_errs = north_errs[(north_errs > -200) & (north_errs < 200)]
        
        east_errors.append(east_errs)
        north_errors.append(north_errs)
        labels.extend([f"{mode} East", f"{mode} North"])

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    plt.suptitle(f'Boxplot of East/North errors for {res_obj.experiment} in {scenario_type} scenarios', fontsize=20)

    # Create positions for the boxes and their sticks
    positions, tick_pos = [], []
    for i in range(len(modes)):
        positions.extend([i*3, i*3 + 1])
        tick_pos.append(i*3 + 0.5)

    # Box plot for East and North errors, interleaved
    data = [val for pair in zip(east_errors, north_errors) for val in pair]
    box = ax.boxplot(data, patch_artist=True, positions=positions, widths=0.6)

     # Color each box (two colors only, one for East and one for North)
    east_color = 'skyblue'  # Color for East errors
    north_color = 'salmon'  # Color for North errors
    color_cycle = [east_color, north_color] * len(modes)

    for patch, color in zip(box['boxes'], color_cycle):
        patch.set_facecolor(color)

    # Setting labels and grids
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(modes,fontsize=15)
    ax.set_ylabel('Error [m]',fontsize=15)
    ax.set_ylim(axlim) 
    ax.grid(True, linestyle='--', alpha=0.6)

    # Create custom legends for East and North
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=east_color, label='East Errors'),
                       Patch(facecolor=north_color, label='North Errors')]
    ax.legend(handles=legend_elements, loc='upper right',fontsize=15)

    plt.tight_layout()
    plt.show()

def get_geopos(res_obj, modes, scenario, reference):
    """
    Returns the positions as longtitudes and latitudes.
    """
    positions = {}
    for mode in modes:
        enu_pos = res_obj.pos[mode][scenario]
        positions[mode] = {"lons": np.zeros(len(enu_pos), dtype=float), "lats": np.zeros(len(enu_pos), dtype=float)}
        for i in range(len(enu_pos)):
            # get positions for each scenario: 
            lat, lon, _ = pm.enu2geodetic(enu_pos[i][0], enu_pos[i][1], enu_pos[i][2],
                                                reference[0], reference[1], reference[2])
            positions[mode]["lons"][i] = lon
            positions[mode]["lats"][i] = lat
            
    return positions

def plotMap(res_obj, modes, scenario, extent, scale, plot_type="data", figsize=8):

    plt.rcParams.update({'font.size': 11})
    #######################################
    # Get the image from web server
    # function from: https://makersportal.com/blog/2020/4/24/geographic-visualizations-in-python-with-cartopy
    #######################################
    def image_spoof(self, tile): # this function pretends not to be a Python script
        url = self._image_url(tile) # get the url of the street map API
        req = Request(url) # start request
        req.add_header('User-agent','Anaconda 3') # add user agent to request
        fh = urlopen(req) 
        im_data = io.BytesIO(fh.read()) # get image
        fh.close() # close url
        img = Image.open(im_data) # open image with PIL
        img = img.convert(self.desired_tile_form) # set image format
        return img, self.tileextent(tile), 'lower' # reformat for cartopy

    cimgt.OSM.get_image = image_spoof # reformat web request for street map spoofing
    osm_img = cimgt.OSM() # spoofed, downloaded street map

    center = main.init_positions[scenario]
    fig = plt.figure(figsize=(figsize, figsize)) # open matplotlib figure
    ax1 = plt.axes(projection=osm_img.crs) # project using coordinate reference system (CRS) of street map
    bounding = [center[1] - extent[0], center[1] + extent[0], center[0] - extent[1], center[0] + extent[1]]
    ax1.set_extent(bounding) # set extents
    ax1.add_image(osm_img, int(scale)) # add OSM with zoom specification

    #######################################
    # Formatting the Cartopy plot
    #######################################
    ax1.set_xticks(np.linspace(bounding[0],bounding[1],7),crs=ccrs.PlateCarree()) # set longitude indicators
    ax1.set_yticks(np.linspace(bounding[2],bounding[3],7)[1:],crs=ccrs.PlateCarree()) # set latitude indicators
    lon_formatter = LongitudeFormatter(number_format='0.3f',degree_symbol='',dateline_direction_label=True) 
    lat_formatter = LatitudeFormatter(number_format='0.3f',degree_symbol='') 
    ax1.xaxis.set_major_formatter(lon_formatter) 
    ax1.yaxis.set_major_formatter(lat_formatter) 

    #######################################
    # Plotting 
    #######################################
    if plot_type == 'data':
        # To draw polylines with geodetic coordinates, change linewidth to bigger values
        positions = get_geopos(res_obj, modes, scenario, center)
        for label, loc in positions.items():
            ax1.plot(loc['lons'], loc['lats'], linewidth=0, marker='o', markersize=4, transform=ccrs.Geodetic(), label=label)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fontsize=15, fancybox=True, shadow=True, ncol=6)

    if plot_type=="reference":
        ax1.plot(center[1], center[0], marker='x', color='black', markersize=10, markeredgewidth=2, linestyle='', transform=ccrs.Geodetic())
        ax1.text(center[1], center[0], 'Reference', transform=ccrs.Geodetic(), ha='right', va='bottom')

    plt.grid(False)
    plt.show()

if __name__ == '__main__':      
    
    all_experiments = {} # all the results are now stored here
    for i in range(3):
        obj = Results_for_experiment(i+1)
        obj.processing()
        all_experiments[i+1] = obj

    # Pickle the dictionary and save it to a file
    with open('all_experiments.pkl', 'wb') as file:
        pickle.dump(all_experiments, file)