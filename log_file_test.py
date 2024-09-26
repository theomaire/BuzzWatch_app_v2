import os
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def get_datetime_from_file_name(file_name):
    try:
        # General regex pattern for extracting date and time and video number
        match = re.match(r'.*_(\d{6})_[^_]*_(\d{6})_v(\d+)\.log', file_name)
        if not match:
            raise ValueError("No matching pattern found")

        date_part = match.group(1)  # YYMMDD
        time_part = match.group(2)  # HHMMSS
        video_number = int(match.group(3))  # Video number as an integer

        # Parse the date and time components
        YY = int(date_part[:2])
        MM = int(date_part[2:4])
        DD = int(date_part[4:6])
        HH = int(time_part[:2])
        MI = int(time_part[2:4])
        SS = int(time_part[4:6])

        # Base time
        base_time = datetime(2000 + YY, MM, DD, HH, MI, SS)
        
        # Compute the actual start time by adding the video number offset
        start_time = base_time + timedelta(minutes=20 * (video_number - 1))
        return start_time
    except Exception as e:
        print(f"Error parsing date and time from file name '{file_name}': {e}")
        return None
# Path to the directory containing the log files
log_directory = '/Users/tmaire/Documents/Transfer_data/test_analysis_buzzwatch_V3/ANALYZED/test_experiment/Cage04_PHN/batch_1/log_analysis'  # Update this with the path to your log files directory

track_pattern = re.compile(r'Total number of tracks: (\d+)')

# Lists to store datetime and respective number of tracks
datetimes = []
track_counts = []

# Loop through all files in the directory
for filename in os.listdir(log_directory):
    if filename.endswith('.log'):
        datetime_from_file = get_datetime_from_file_name(filename)
        if datetime_from_file:
            with open(os.path.join(log_directory, filename), 'r') as file:
                file_content = file.read()
                match = track_pattern.findall(file_content)
                if match:
                    # Convert matched string to integer and add to the list
                    track_counts.append(int(match[-1])/40)
                    datetimes.append(datetime_from_file)

# Ensure both lists are sorted by datetime
sorted_data = sorted(zip(datetimes, track_counts))
datetimes, track_counts = zip(*sorted_data)

# Plotting the number of tracks as a function of time
plt.figure(figsize=(10, 5))
plt.plot(datetimes, track_counts, marker='o', linestyle='-')
plt.title('Number of Tracks Over Time')
plt.xlabel('Time')
plt.ylabel('Number of Tracks')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
