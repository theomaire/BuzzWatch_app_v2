import sys
import os
import json
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk  # Add this import for handling the logo image
from ui_manager import UIManager
from experiment_manager import ExperimentManager
from video_manager import VideoManager
from image_manager import ImageManager

PATH_TO_APP = os.path.dirname(__file__)
sys.path.append(PATH_TO_APP) 

class VideoAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BuzzWatch_analysis_app")

        # Set the application icon
        logo_path = os.path.join(PATH_TO_APP, "app_logo.png")
        if os.path.exists(logo_path):
            logo_image = Image.open(logo_path)
            logo_photo = ImageTk.PhotoImage(logo_image)
            self.root.iconphoto(False, logo_photo)

        self.config_path = os.path.join(os.path.dirname(__file__), "config.json")
        self.config = self.load_config()

        # Create a unified logging area in the main window
        self.log_frame = tk.Frame(self.root)
        self.log_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        self.log_text = tk.Text(self.log_frame, height=8)
        self.log_text.pack(fill=tk.X, expand=True)
               
        self.last_progress_line_index = None 

        # Initialize managers
        self.experiment_manager = ExperimentManager(self.log)
        self.video_manager = VideoManager(self.log)
        self.image_manager = ImageManager(self.log)
        self.ui_manager = UIManager(self.root, self.experiment_manager, self.video_manager, self.image_manager, self.log)

        self.experiment_manager.ui_manager = self.ui_manager

        # Set central log function to all managers
        self.experiment_manager.log = self.log
        self.video_manager.log = self.log
        self.image_manager.log = self.log
        self.ui_manager.log = self.log

        self.ui_manager.init_tabs()

        #Load the last opened experiment if available
        if 'last_experiment' in self.config:
            try:
                self.experiment_manager.load_experiment(self.config['last_experiment'])
                self.ui_manager.update_ui_after_loading_experiment()
            except Exception:
                self.log("Error loading last experiment")


        #Initialize UI tabs


    def log(self, message):
        """Log messages in the logging area."""
        if "Progress" in message:
            # Determine the start of the last line
            last_line_index = self.log_text.index("end-1c linestart")
            # Delete the current text of the last line
            self.log_text.delete(last_line_index, 'end-1c')
            # Insert the new progress message at the start of the last line
            self.log_text.insert(last_line_index, message)

            self.last_progress_line_index = last_line_index
        else:
            self.log_text.insert(tk.END, message + "\n")
            self.last_progress_line_index = None

        self.log_text.see(tk.END)
        self.log_text.update_idletasks()  # Force the refresh of the UI


    def load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {}

    def save_config(self, config):
        with open(self.config_path, 'w') as f:
            json.dump(config, f)

    def on_closing(self):
        """Handle the window closing event."""
        self.experiment_manager.stop_video_playback()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAnalyzerApp(root)
    root.geometry("1200x800")
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
