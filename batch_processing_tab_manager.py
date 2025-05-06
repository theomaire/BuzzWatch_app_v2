import tkinter as tk
from tkinter import ttk
import os
from common_imports import *
from multiprocessing import Pool, cpu_count
import pandas as pd
from tqdm import tqdm

# Define the function at module level
def process_video_files(video_file, folder_analysis):
    """
    Process a single video file to extract necessary data.
    
    Args:
        video_file (str): The name of the video file to process.
        folder_analysis (str): The directory where analysis data is stored.

    Returns:
        tuple or None: Returns the processed data as a tuple or None if an error occurs.
    """
    final_data_path = os.path.join(folder_analysis, "final_tracking_data", f"forward_mosq_tracks_{video_file}")
    if not os.path.exists(final_data_path):
        print(f"File not found: {final_data_path}")
        return None

    try:
        with open(final_data_path, 'rb') as f:
            video_data = pickle.load(f)
    except (OSError, pickle.PickleError) as e:
        # Log the error and return None on exception to prevent crashing
        print(f"Error loading {final_data_path}: {e}")
        return None

    try:
        population_data = video_data.population_variables.resample('1s', label='right').mean()
        individual_data = video_data.individual_variables
        flight_metrics_data = video_data.flight_metrics_around_resting if hasattr(video_data, 'flight_metrics_around_resting') else pd.DataFrame()
        resting_data = video_data.resting_variables

        start_time = video_data.time_stamp[0]
        avg_fraction_flying = video_data.population_variables['numb_mosquitos_flying'].mean()
        total_nb_tracks = len(video_data.objects.keys())

        return (population_data, individual_data, resting_data, flight_metrics_data,
                [start_time, avg_fraction_flying, total_nb_tracks])
    except Exception as e:
        # Log the error and return None on processing error
        print(f"Error processing data from {final_data_path}: {e}")
        return None


class BatchProcessingTabManager:
    def __init__(self, root, ui_manager, state_manager):
        self.root = root
        self.ui_manager = ui_manager
        self.state_manager = state_manager
        self.experiment_manager = ui_manager.experiment_manager
        self.log = ui_manager.log

    def init_batch_processing_tab(self, tab):
        self.tab = tab
        self.configure_layout()
        self.create_controls()

    def configure_layout(self):
        self.tab.grid_rowconfigure(0, weight=1)
        self.tab.grid_columnconfigure(0, weight=1)

    def create_controls(self):
        controls_frame = tk.Frame(self.tab)
        controls_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self._create_button(controls_frame, "Generate Batch Processing Script", self.generate_batch_script)
        self._create_button(controls_frame, "Concatenate and Save Data", self.concatenate_and_save_plots)
        self._create_button(controls_frame, "Load Data", self.load_data)
        self._create_button(controls_frame, "Plot Rolling Average", self.plot_rolling_average)
        self._create_button(controls_frame, "Plot summary data", self.plot_summary_data)
        self._create_button(controls_frame, "Plot Histograms", self.plot_histograms)
        self._create_button(controls_frame, "Plot Scatter and Bar", self.plot_scatter_and_bar)
        self.script_status_label = tk.Label(controls_frame, text='', fg='blue')
        self.script_status_label.pack(side=tk.TOP, pady=5)

    def _create_button(self, parent, text, command):
        button = tk.Button(parent, text=text, command=command)
        button.pack(side=tk.TOP, fill=tk.X, expand=True, pady=5)

    # Implement methods for generating scripts, concatenating data, loading data,
    # plotting rolling averages, histograms, and scatter/bar plots.




    def generate_batch_script(self):
        folder_videos = self.experiment_manager.folder_videos
        folder_analysis = self.experiment_manager.folder_analysis
        settings_file = self.experiment_manager.settings_file
        last_experiment = self.experiment_manager.config.get('last_experiment', '')

        script_content = f'''
    import os
    import sys
    from multiprocessing import Pool, current_process
    import pickle
    import pandas as pd
    import time
    from experiment_manager import ExperimentManager

    # Define a dummy logger that does nothing
    class DummyLogger:
        def write(self, message):
            pass

        def flush(self):
            pass

    # Function to process each video
    def process_video(video_name):
        # Initialize ExperimentManager with a dummy log function
        def dummy_log(message):
            pass

        # Load experiment settings
        experiment_manager = ExperimentManager(dummy_log)
        experiment_manager.load_settings("{settings_file}")
        experiment_manager.folder_videos = "{folder_videos}"
        experiment_manager.folder_analysis = "{folder_analysis}"
        experiment_manager.load_experiment("{last_experiment}")

        # Redirect stdout and stderr to the dummy logger to suppress output
        sys.stdout = DummyLogger()
        sys.stderr = DummyLogger()

        # Set video name here
        experiment_manager.run_single_video_analysis(video_name)
        video_name_bare = os.path.basename(video_name)
        print(f"Processed: {video_name_bare}")

    if __name__ == "__main__":
        video_files = [f.replace(".mp4", "") for f in os.listdir("{folder_videos}") if f.endswith(".mp4")]

        num_videos = len(video_files)
        start_time = time.time()

        with Pool(processes=4) as pool:  # Adjust the number of processes as needed
            for i, _ in enumerate(pool.imap_unordered(process_video, video_files), 1):
                elapsed_time = time.time() - start_time
                time_per_video = elapsed_time / i
                est_total_time = time_per_video * num_videos
                est_remaining_time = est_total_time - elapsed_time
                percent_complete = (i / num_videos) * 100
                print(f"Processed {i}/{num_videos} videos ({percent_complete:.2f}%). Estimated remaining time: {est_remaining_time / 60:.2f} minutes.")

        print("Batch processing complete.")
    '''

        script_path = os.path.join(folder_analysis, "batch_processing_script.py")
        with open(script_path, 'w') as script_file:
            script_file.write(script_content)

        self.log(f"Batch processing script generated at {script_path}")
        self.script_status_label.config(text=f"Script generated at: {script_path}")


    def concatenate_and_save_plots(self):
        """
        Concatenate and save plots data using multiprocessing for efficiency.
        """
        folder_analysis = self.experiment_manager.folder_analysis
        video_files = [f.replace(".png", "") for f in os.listdir(os.path.join(folder_analysis, "images_mortality")) if f.endswith(".png") and f.startswith("Cage")]

        
        if not video_files:
            self.log("No data found for concatenation.")
            return

        video_files.sort()  # Sort the video files for consistent processing order
        #video_files = video_files[0:100]
        # Use a Pool to utilize all available CPU cores
        with Pool(processes=8) as pool:
            results = pool.starmap(process_video_files, [(video_name, folder_analysis) for video_name in video_files])

        # Filter out results that are None (for files that couldn't be processed)
        results = [result for result in results if result is not None]

        if not results:
            self.log("No data found for concatenation.")
            return

        # Separate the results into distinct lists
        all_population_data, all_individual_data, all_resting_data, all_flight_metrics_data, summary_data = zip(*results)

        # Concatenate data, ensuring proper handling of lists
        full_population_data = pd.concat(all_population_data)
        full_individual_data = pd.concat(all_individual_data)
        full_resting_data = pd.concat(all_resting_data)
        try:
            full_flight_metrics_data = pd.concat(filter(lambda x: not x.empty, all_flight_metrics_data), ignore_index=True)
        except Exception:
            full_flight_metrics_data = None
        summary_df = pd.DataFrame(summary_data, columns=['start_time', 'avg_fraction_flying', 'total_nb_tracks'])

        # Save concatenated data into a single .pkl file
        final_output_path = os.path.join(folder_analysis, "analyzed_data.pkl")
        try:
            with open(final_output_path, 'wb') as f:
                pickle.dump({
                    'population_data': full_population_data,
                    'individual_data': full_individual_data,
                    'resting_data': full_resting_data,
                    'summary_data': summary_df,
                    'flight_metrics_data': full_flight_metrics_data
                }, f)
        except Exception as e:
            self.log(f"Failed to save data: {e}")
            return

        # Notify completion
        self.log("Concatenation complete. Data saved to:")
        self.log(final_output_path)



    def load_data(self):
        folder_analysis = self.experiment_manager.folder_analysis
        analyzed_data_path = os.path.join(folder_analysis, "analyzed_data.pkl")

        if os.path.exists(analyzed_data_path):
            with open(analyzed_data_path, 'rb') as f:
                self.analyzed_data = pickle.load(f)

                self.log("Data successfully loaded.")
        else:
            self.log("Data files not found. Please merge the data first.")

    def plot_rolling_average(self):
        if self.analyzed_data is None:
            self.log("Data not loaded. Please load the data first.")
            return
        
        full_population_data = self.analyzed_data['population_data']

        # Resampling and computing the rolling average for population data
        resampled_population_data = full_population_data.resample('1T').mean()   # Resample per minute
        rolling_population_data = resampled_population_data.rolling(window=20).mean()  # Rolling average, window of 20 minutes

        # Plotting resampled and rolling average data
        fig, axs = plt.subplots(2, 1, figsize=(10, 6))
        
        rolling_population_data['numb_mosquitos_flying'].plot(ax=axs[0], legend=True, title='Number of Mosquitos Flying (Rolling Average)')
        rolling_population_data['numb_mosquitos_sugar'].plot(ax=axs[1], legend=True, title='Number of Mosquitos at Sugar Feeder (Rolling Average)')
        rolling_population_data['numb_mosquitos_hs'].plot(ax=axs[1], legend=True, title='Control')
        rolling_population_data['numb_mosquitos_left_ctrl'].plot(ax=axs[1], legend=True, title='Square 3')
        rolling_population_data['numb_mosquitos_right_ctrl'].plot(ax=axs[1], legend=True, title='Square 4')

        for ax in axs:
            ax.set_xlabel("Time")
            ax.set_ylabel("Count")

        plt.tight_layout()
        plt.show()


    def plot_summary_data(self):
        if self.analyzed_data is None:
            self.log("Data not loaded. Please load the data first.")
            return
        
        df = self.analyzed_data['summary_data']
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.scatter(df["avg_fraction_flying"],df["total_nb_tracks"]/40)

        plt.show()


    def plot_histograms(self):
        if self.analyzed_data is None:
            self.log("Data not loaded. Please load the data first.")
            return
        
        full_individual_data = self.analyzed_data['individual_data']

        # Plotting individual statistics histograms
        fig, axs = plt.subplots(2, 1, figsize=(10, 6))

        full_individual_data['flight_duration'].plot(kind='hist', bins=20, ax=axs[0], title='Histogram of Flight Duration', color='purple')
        full_individual_data['average_speed'].plot(kind='hist', bins=20, ax=axs[1], title='Histogram of Flight Speed', color='orange')
        
        for ax in axs:
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()

    def plot_scatter_and_bar(self):
        if self.analyzed_data is None:
            self.log("Data not loaded. Please load the data first.")
            return
        
        full_flight_metrics_data = self.analyzed_data.get('flight_metrics_data', pd.DataFrame())

        # Plot takeoff duration as a function of resting time from concatenated flight metrics data
        if not full_flight_metrics_data.empty:
            fig, axs = plt.subplots(1, 2, figsize=(20, 6))
            zone_colors = {
                'sugar': 'blue',
                'hs': 'green',
                'left_ctrl': 'red',
                'right_ctrl': 'orange'
            }

            # Scatter plot for takeoff duration as a function of resting time
            for zone, color in zone_colors.items():
                zone_data = full_flight_metrics_data[full_flight_metrics_data['zone'] == zone]
                axs[0].scatter(zone_data['resting_time'], zone_data['takeoff'].apply(lambda x: x[0]), color=color, label=zone)

            axs[0].set_title("Takeoff Duration as a function of Resting Time")
            axs[0].set_xlabel("Resting Time (s)")
            axs[0].set_ylabel("Takeoff Duration (s)")
            axs[0].legend(title="Zone")

            # Bar graph for average duration of flight after takeoff for each zone
            avg_takeoff_durations = []
            for zone in zone_colors.keys():
                zone_data = full_flight_metrics_data[(full_flight_metrics_data['zone'] == zone) & (full_flight_metrics_data['resting_time'] > 10)]
                avg_takeoff_duration = zone_data['takeoff'].apply(lambda x: x[0]).mean()
                avg_takeoff_durations.append(avg_takeoff_duration)

            axs[1].bar(zone_colors.keys(), avg_takeoff_durations, color=zone_colors.values())
            axs[1].set_title("Average Duration of Flight After Takeoff (Resting Time > 10s)")
            axs[1].set_xlabel("Zone")
            axs[1].set_ylabel("Average Takeoff Duration (s)")

            plt.tight_layout()
            plt.show()
        else:
            self.log("No flight metrics data to plot.")

