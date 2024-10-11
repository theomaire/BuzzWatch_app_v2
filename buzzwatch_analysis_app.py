# buzzwatch_analysis_app.py
import sys
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

from ui_manager import UIManager
from experiment_manager import ExperimentManager
from video_manager import VideoManager
from image_manager import ImageManager
from state_manager import StateManager
from config_manager import ConfigManager

PATH_TO_APP = os.path.dirname(__file__)
sys.path.append(PATH_TO_APP)

class VideoAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BuzzWatch_analysis_app")
        
        # Set the application icon
        self._set_app_icon()

        self._maximize_window()


        # Initialize ConfigManager
        self.config_manager = ConfigManager(os.path.join(PATH_TO_APP, "config.json"))
        self.config = self.config_manager.config

        # Initialize Managers
        self._initialize_managers()

        # Create a unified logging area in the main window
        self._setup_logging_area()

        # Initialize UI tabs
        self.ui_manager.init_tabs()

        # Load the last opened experiment if available
        self._load_last_experiment()

    def _set_app_icon(self):
        logo_path = os.path.join(PATH_TO_APP, "app_logo.png")
        if os.path.exists(logo_path):
            logo_image = Image.open(logo_path)
            logo_photo = ImageTk.PhotoImage(logo_image)
            self.root.iconphoto(False, logo_photo)

    def _initialize_managers(self):
        self.experiment_manager = ExperimentManager(self.log)
        self.video_manager = VideoManager(self.log)
        self.image_manager = ImageManager(self.log)
        self.state_manager = StateManager()
        self.ui_manager = UIManager(self.root, self.experiment_manager, self.video_manager, self.image_manager, self.log, self.state_manager)
        
        self.experiment_manager.ui_manager = self.ui_manager

        # Set central log function to all managers
        self.experiment_manager.log = self.log
        self.video_manager.log = self.log
        self.image_manager.log = self.log
        self.ui_manager.log = self.log

    def _setup_logging_area(self):
        self.log_frame = tk.Frame(self.root)
        self.log_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        self.log_text = tk.Text(self.log_frame, height=8)
        self.log_text.pack(fill=tk.X, expand=True)
               
        self.last_progress_line_index = None

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

    def _load_last_experiment(self):
        if 'last_experiment' in self.config:
            try:
                self.experiment_manager.load_experiment(self.config['last_experiment'])
                self.ui_manager.update_ui_after_loading_experiment()
            except Exception:
                self.log("Error loading last experiment")

    def on_closing(self):
        """Handle the window closing event."""
        self.experiment_manager.stop_video_playback()
        self.root.destroy()
    def _maximize_window(self):
        """Maximize the window to fit the screen."""
        self.root.state('zoomed')  # This will maximize the window for Windows and macOS


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAnalyzerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()