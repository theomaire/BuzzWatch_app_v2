import pickle
import os
import shutil
import yaml
from tkinter import filedialog
import sys
import time
import json
from PIL import Image, ImageTk
import cv2
from threading import Thread

from buzzwatch_data_analysis.experiment_analysis import buzzwatch_experiment_analysis
from buzzwatch_data_analysis.misc_functions import create_folder
from buzzwatch_data_analysis.single_video_analysis import single_video_analysis
from logger import MultiLogger


class ExperimentManager:
    def __init__(self,log_func):
        self.log = log_func
        self.experiment = None
        self.folder_analysis = None
        self.folder_videos = None
        self.experiment_alias = None
        self.settings = None
        self.config_path = os.path.join(os.path.dirname(__file__), "config.json")
        self.config = self.load_config()
        self.settings_file = None
        self.cap = None
        self.total_frames = 0
        self.is_playing = False
        self.tracking_image_label = None
        self.video_label = None
        self.scrollbar = None
        self.mosquito_tracks = None

    def load_settings(self, settings_file):
        with open(settings_file, 'r') as file:
            self.settings = yaml.safe_load(file)
        self.settings_file = settings_file
        self.log(f"Settings loaded from {settings_file}")

    def load_experiment(self, exp_file=None,update_ui_func=None):
        if exp_file is None:
            exp_file = filedialog.askopenfilename(title="Select Experiment File", filetypes=[("Pickle files", "*.pkl")])
        if not exp_file:
            self.log("No experiment file selected.")
            return

        with open(exp_file, 'rb') as f:
            self.experiment = pickle.load(f)
            self.folder_analysis = getattr(self.experiment, 'folder_analysis', None)
            self.folder_videos = getattr(self.experiment, 'folder_videos', None)
            self.experiment_alias = self.experiment.experiment_alias
            self.settings_file = self.experiment.settings_file
            with open(self.settings_file, 'r') as file:
                self.settings = yaml.safe_load(file)
            self.experiment.settings = self.settings
            self.experiment.log = self.log
            #self.settings = self.experiment.settings
            
        
        self.config['last_experiment'] = exp_file
        self.save_config(self.config)
        try:
            self.video_files = sorted([os.path.join(self.folder_videos, f) for f in os.listdir(self.folder_videos) if f.endswith('.mp4')])
            if not self.video_files:
                self.log("No .mp4 files found in the selected directory.")
                return
        except Exception:
            self.log("video_folder found.")

        if update_ui_func:
            update_ui_func()

    def load_experiment_from_json(self, json_path, update_ui_func=None):
        """Load experiment details from a JSON file."""
        if not os.path.exists(json_path):
            self.log("Experiment JSON file not found.")
            return

        with open(json_path, 'r') as json_file:
            experiment_details = json.load(json_file)

        # Load all parts of the experiment details
        self.root_folder_videos = experiment_details.get('root_folder_videos')
        self.root_folder_analysis = experiment_details.get('root_folder_analysis')
        self.experiment_name = experiment_details.get('experiment_name')
        self.cage_name = experiment_details.get('cage_name')
        self.batch_name = experiment_details.get('batch_name')
        self.settings_file = experiment_details.get('settings_file')  # Reference the copied file path

        # Reconstruct full paths using loaded data
        self.folder_videos = os.path.join(self.root_folder_videos, self.experiment_name, self.cage_name, self.batch_name)
        self.folder_analysis = os.path.join(self.root_folder_analysis, self.experiment_name, self.cage_name, self.batch_name)

        # Load settings from the copied file
        self.load_settings(self.settings_file)

        self.experiment_alias = f"exp_{self.experiment_name}_{self.cage_name}_{self.cage_name}"

        # Create or update the experiment object
        self.experiment = buzzwatch_experiment_analysis(
            self.folder_analysis,
            self.folder_videos,
            self.experiment_alias,
            self.settings,
            self.settings_file,
            log_func=self.log,
            debug_mode=False
        )

        self.log(f"Experiment details loaded from {json_path}")

        # Call the UI update function if provided to refresh relevant UI components
        if update_ui_func:
            update_ui_func()


    def initialize_new_experiment(self, root_folder_videos=None, root_folder_analysis=None,
                                  experiment_name=None, cage_name=None, batch_name=None,settings_file=None,
                                  update_ui_func=None):
        if not root_folder_videos or not root_folder_analysis:
            self.log("Root folders for videos and analysis must be provided.")
            return
        
        if not experiment_name or not cage_name or not batch_name:
            self.log("Experiment name, cage name, and batch name must be provided.")
            return
        
        if not settings_file or not os.path.isfile(settings_file):
            self.log("A valid settings file must be provided.")
            return

        folder_videos_path = os.path.join(root_folder_videos, experiment_name, cage_name, batch_name)
        folder_analysis_path = os.path.join(root_folder_analysis, experiment_name, cage_name, batch_name)

        create_folder(folder_analysis_path)


        settings_file_name = os.path.basename(settings_file)
        settings_file_destination = os.path.join(folder_analysis_path, settings_file_name)
        shutil.copyfile(settings_file, settings_file_destination)

        with open(settings_file_destination, 'r') as file:
            self.settings = yaml.safe_load(file)

        experiment_alias = f"{cage_name}_{batch_name}"

        self.folder_videos = folder_videos_path
        self.folder_analysis = folder_analysis_path
        self.experiment_alias = experiment_alias

        self.experiment = buzzwatch_experiment_analysis(
            self.folder_analysis,
            self.folder_videos,
            self.experiment_alias,
            self.settings,
            settings_file_destination,
            log_func=self.log,
            debug_mode=False
        )

        # Save experiment details to JSON
        experiment_details = {
            'root_folder_videos': root_folder_videos,
            'root_folder_analysis': root_folder_analysis,
            'experiment_name': experiment_name,
            'cage_name': cage_name,
            'batch_name': batch_name,
            'settings_file': settings_file_destination  # Use the copied settings file
        }
        json_experiment_path = os.path.join(self.folder_analysis, f"experiment_{self.experiment_alias}.json")
        self.save_experiment_to_json(json_experiment_path, experiment_details)

        #self.config['last_experiment'] = json_experiment_path
        #self.save_config(self.config)
        self.log(f"Experiment details saved to {json_experiment_path}")

        if update_ui_func:
            update_ui_func()

        self.log("New experiment initialized successfully.")


    def save_experiment_to_json(self, json_path, experiment_details):
        """Save experiment details to a JSON file."""
        with open(json_path, 'w') as json_file:
            json.dump(experiment_details, json_file, indent=4)
        self.log(f"Experiment details saved to {json_path}")


    ## Sample functions to go inside ExperimentManager class
    def get_images_from_video(self, force_rerun):
        self.log("Extracting images from video...")
        self.experiment.extract_images_v2(force_rerun)
        
        # Update image listbox
        #self.update_image_listbox()
        self.log("Images extracted and listed.")

    def get_background_from_images(self, force_rerun):
        self.log("Computing background from images...")
        self.experiment.extract_average_background(force_rerun)
        #self.update_median_image_listbox()
        self.log("Background images computed and saved.")

    def draw_cage_borders(self):
        self.log("Drawing cage borders...")
        self.experiment.user_input_draw_borders_cage(force_to_redo=1)
        self.experiment.plot_all_borders()
        self.log("Cage borders drawn and saved.")

    def draw_sugar_feeder_borders(self):
        self.log("Drawing sugar feeder borders...")
        self.experiment.user_input_draw_sugar_feeding(force_to_redo=1)
        self.experiment.plot_all_borders()
        self.log("Sugar feeder borders drawn and saved.")

    def draw_control_borders(self):
        self.log("Drawing control feeder borders...")
        self.experiment.user_input_draw_control_squares(force_to_redo=1)
        self.experiment.plot_all_borders()
        self.log("Control feeder borders drawn and saved.")

    def draw_square_3(self):
        self.log("Drawing square 3 borders...")
        self.experiment.user_input_draw_control_squares_3(force_to_redo=1)
        self.experiment.plot_all_borders()
        self.log("Square 3 borders drawn and saved.")

    def draw_square_4(self):
        self.log("Drawing square 4 borders...")
        self.experiment.user_input_draw_control_squares_4(force_to_redo=1)
        self.experiment.plot_all_borders()
        self.log("Square 4 borders drawn and saved.")

    def update_border_image(self):
        self.log("Udpating backgrund with borders")
        self.experiment.plot_all_borders()

    def run_tracking_analysis(self,video_name):
        self.experiment.run_single_video_analysis(video_name,debug_mode=1)

    def get_final_tracking_path(self, video_name):
        # Specify how you want to derive final tracking file paths from the video name
        final_tracking_folder = os.path.join(self.folder_analysis, "final_tracking")
        final_tracking_files = [os.path.join(final_tracking_folder, f) for f in os.listdir(final_tracking_folder) if video_name in f]
        return final_tracking_files
    
    def get_video_path(self, video_name):
        # Implement this method based on your application structure
        return os.path.join(self.folder_videos, video_name+".mp4")


    def load_tracking_data(self, tracking_file):
        tracking_file_path = os.path.join(self.folder_analysis, "final_tracking_data", tracking_file)
        if not os.path.exists(tracking_file_path):
            self.log(f"Tracking file {tracking_file_path} does not exist.")
            return
        with open(tracking_file_path, 'rb') as f:
            self.mosquito_tracks = pickle.load(f)


    def stop_video_playback(self):
        self.is_playing = False
        if self.cap:
            self.cap.release()
            self.cap = None
    def load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {}

    def save_config(self, config):
        with open(self.config_path, 'w') as f:
            json.dump(config, f)