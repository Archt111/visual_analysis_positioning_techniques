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
    plt.xlabel("Cumulative horizontal error [m]")
    plt.legend()
    plt.title(f"Experiment {res_obj.experiment}")
    plt.show()
    
def get_enu_errors(res_obj, mode, data):
    """Using mean of ENU to get errors"""
    all_enu_data = np.concatenate([res_obj.pos[mode][key] for key in data], axis=0)
    all_enu_data = np.where(np.isinf(all_enu_data), np.nan, all_enu_data)
    east_errs, north_errs, up_errs = all_enu_data.T  # transposes the array to split by columns

    return east_errs, north_errs, up_errs

def generate_statistics(res_obj):
    """Generate a DataFrame containing statistics for ENU, 2D, 3D, runtime, and horizontal error under 200m."""
    
    def find_min_max_std_mean_median(an_array):
        return np.nanmin(an_array), np.nanmax(an_array), np.nanstd(an_array), np.nanmean(an_array), np.nanmedian(an_array)
    
    stats = []
    
    for mode in res_obj.modes:
        east_errs, north_errs, up_errs = get_enu_errors(res_obj, mode, res_obj.data)
        e_min, e_max, e_std, e_mean, e_median = find_min_max_std_mean_median(east_errs)
        n_min, n_max, n_std, n_mean, n_median = find_min_max_std_mean_median(north_errs)
        u_min, u_max, u_std, u_mean, u_median = find_min_max_std_mean_median(up_errs)

        rmse_2d = np.sqrt(np.nanmean(east_errs**2 + north_errs**2))
        rmse_3d = np.sqrt(np.nanmean(east_errs**2 + north_errs**2 + up_errs**2))

        all_error_in_mode = np.array(list(chain.from_iterable([res_obj.error[mode][key] for key in res_obj.data])))
        error_less_200 = ((all_error_in_mode < 200).sum(axis=0) / len(all_error_in_mode))*100

        stats.append([mode, e_min, n_min, u_min, e_max, n_max, u_max, e_std, n_std, u_std, e_mean, n_mean, u_mean, 
                      e_median, n_median, u_median, rmse_2d, rmse_3d, error_less_200, res_obj.run_time])

    columns = ["Mode", "Min E", "Min N", "Min U", "Max E", "Max N", "Max U", "SD E", "SD N", "SD U", "Mean E", "Mean N", "Mean U", 
               "Median E", "Median N", "Median U", "RMSE 2D", "RMSE 3D", "% Error < 200m", "Mean Runtime (s)"]
    
    df = pd.DataFrame(stats, columns=columns)
    format_map = {"% Error < 200m": "{:.0f}%",  "Mean Runtime (s)": "{:.2f}",}

    for metric in columns[1:-2]:
            format_map[metric] = "{:.2e}"
        # Formatting outputs for readibility
        # df.style.format(format_map) for newer panda ver
    for key, value in format_map.items():
        df[key] = df[key].apply(value.format)
    return df

# # Adjust global pandas display settings (optional)
# pd.set_option('display.max_columns', None)  # Show all columns
# pd.set_option('display.precision', 2)  # Set precision for decimal places
# pd.set_option('display.float_format', '{:.2e}'.format)  # Scientific notation
def statistics(df, whattoshow="all"):
    """Filter and display statistics from the DataFrame based on whattoshow parameter."""
    
    if whattoshow == "all":
        df_stats = df
    else:
        columns_to_show = ["Mode"] + [col for col in df_stats.columns if whattoshow.lower() in col.lower()]
        df_stats = df[columns_to_show]

    # def formatting(val):
    #     if isinstance(val, str):  
    #         return val
    #     if "Error < 200m" in val:  
    #         return f"{val:.0f}%"
    #     return f"{val:.2e}"
    # df_stats.style.format(formatting)
    return df_stats


# The next three functions are inspired from https://github.com/agrenier-gnss/MimirAnalyzer/tree/main
def plot_ENU_scenario(res_obj, many, figsize=8):
    """Plot ENU errors of all modes in the experiment for static/dynamic scenarios"""
    
    fig, axs = plt.subplots(3, figsize=(figsize,figsize), sharex=True)
    plt.suptitle(f'East / North / Up errors for experiment {res_obj.experiment}', fontsize=20)
    for mode in res_obj.modes: 
        east_errs, north_errs, up_errs = get_enu_errors(res_obj,mode, many)
        axs[0].plot(east_errs, label=mode)
        axs[1].plot(north_errs, label=mode)
        axs[2].plot(up_errs, label=mode)

    # Apply limits and labels to all subplots
    labels = ["East [m]","North [m]", "Up [m]"]
    for ax, label in zip(axs, labels):
        ax.set_ylim(-200, 200)
        ax.set_xlim(0, east_errs.size)
        ax.set_ylabel(label)

    # Custom function to format time in x_aix
    def timeTicks(x, pos):                                                                                                                                                                                                                                                        
        total_seconds = x * 0.012 + 10 # Duration of each snapshot 
        return f"{total_seconds:.0f}"  # Formats the string to have no decimal places


    formatter = ticker.FuncFormatter(timeTicks)
    axs[2].xaxis.set_major_formatter(formatter)
    axs[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=6)
    
    plt.xlabel('Duration [s]')  
    plt.show()

def plot_EN(res_obj, many,figsize=8):
    """Plot EN errors of all modes in the experiment for static/dynamic scenarios"""
    from matplotlib.patches import Circle
    # Retrieve a list of colors for plotting points from the default Matplotlib color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # a list of color for drawing points
    fig, axs = plt.subplots(1, figsize=(figsize,figsize))
    fig.suptitle(f'East/North errorsfor experiment {res_obj.experiment}')

    lim = 1e3
    for i, mode in enumerate(res_obj.modes):
        east_errs, north_errs, _ = get_enu_errors(res_obj,mode, many)
        axs.scatter(east_errs, north_errs, label=mode, color=colors[i % len(colors)], s=6, zorder=3)
    
    # Adding a circle to show inliers within 200 meters
    circle = Circle((0, 0), 200, color='blue', fill=False, linestyle='--', linewidth=1, alpha=0.5, zorder=2)
    axs.add_patch(circle)

    axs.grid(zorder=0) # zorder=0: grid below data points
    axs.axis('square')
    axs.set_xlim(-lim, lim)
    axs.set_ylim(-lim, lim)
    axs.set_xlabel('East [m]')
    axs.set_ylabel('North [m]')
    fig.tight_layout()
    axs.yaxis.set_major_formatter('{x:.0f}')
    axs.xaxis.set_major_formatter('{x:.0f}')
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=6)
def get_geopos(res_obj, scenario, reference):
    """
    Returns the positions as longtitudes and latitudes.
    """
    positions = {}
    for mode in res_obj.modes:
        enu_pos = res_obj.pos[mode][scenario]
        positions[mode] = {"lons": np.zeros(len(enu_pos), dtype=float), "lats": np.zeros(len(enu_pos), dtype=float)}
        for i in range(len(enu_pos)):
            # get positions for each scenario: 
            lat, lon, _ = pm.enu2geodetic(enu_pos[i][0], enu_pos[i][1], enu_pos[i][2],
                                                reference[0], reference[1], reference[2])
            positions[mode]["lons"][i] = lon
            positions[mode]["lats"][i] = lat
            
    return positions
def plotMap(res_obj, scenario, extent, scale, marker='', figsize=8):

    plt.rcParams.update({'font.size': 10})
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
    lon_formatter = LongitudeFormatter(number_format='0.4f',degree_symbol='',dateline_direction_label=True) 
    lat_formatter = LatitudeFormatter(number_format='0.4f',degree_symbol='') 
    ax1.xaxis.set_major_formatter(lon_formatter) 
    ax1.yaxis.set_major_formatter(lat_formatter) 

    # Draw polylines with geodetic coordinates
    positions = get_geopos(res_obj,scenario, center)
    for label, loc in positions.items():
        ax1.plot(loc['lons'], loc['lats'],
                linewidth=2, marker=marker, markersize=1, transform=ccrs.Geodetic(), label=label)

    plt.legend()
    plt.grid(False)

if __name__ == '__main__':      
    
    all_experiments = {} # all the results are now stored here
    for i in range(3):
        obj = Results_for_experiment(i+1)
        obj.processing()
        all_experiments[i+1] = obj

    # Pickle the dictionary and save it to a file
    with open('all_experiments.pkl', 'wb') as file:
        pickle.dump(all_experiments, file)