import datetime
import os
from glob import glob
import re

datetimeformat = "%Y%m%d_%H%M%S"

def weeksecondstoutc(gpsweek, gpsseconds, leapseconds, datetimeformat):
    epoch = datetime.datetime.strptime("19800106_000000", datetimeformat)
    elapsed = datetime.timedelta(days=(gpsweek * 7), seconds=(gpsseconds + leapseconds))
    return datetime.datetime.strftime(epoch + elapsed, datetimeformat)

def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(0)) if match else -1

paths = ['Novatel_20211130_resampled_5MHz_1bit_IQ_gain25',
         'Novatel_20211130_resampled_5MHz_8bit_IQ_gain25',
         'Novatel_20211130_resampled_10MHz_1bit_IQ_gain25', 
         'Novatel_20211130_resampled_10MHz_8bit_IQ_gain25']

for path in paths:
    files = glob(os.path.join("TAU_Dataset", "20211130_Novatel", path, "*.bin"))
    files.sort(key=extract_number)
    print(os.path.join("TAU_Dataset", "20211130_Novatel", path))
    
    utc_seconds = 203959.819
    for filename in files:
        old_path = os.path.join(filename)
        new_name = f"{weeksecondstoutc(2186, utc_seconds, 18, datetimeformat)}.bin" 
        new_path = os.path.join(new_name)

        if not os.path.exists(new_path):
            os.rename(old_path, new_path)
        else:
            print(f"File '{new_name}' already exists. Skipping renaming.")
        
        utc_seconds += 1 + 0.02






