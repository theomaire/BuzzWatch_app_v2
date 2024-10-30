from common_imports import *

from video_tab_manager import VideoTabManager
from analysis_tab_manager import AnalysisTabManager
from batch_processing_tab_manager import BatchProcessingTabManager
from comparison_tab_manager import ComparisonTabManager
from state_manager import StateManager

class UIManager:
    def __init__(self, root, experiment_manager, video_manager, image_manager, log_func, state_manager):
        self.root = root
        self.experiment_manager = experiment_manager
        self.video_manager = video_manager
        self.image_manager = image_manager
        self.log = log_func
        self.state_manager = state_manager

        self.notebook = None
        self.setup_managers()

    def setup_managers(self):
        self.video_tab_manager = VideoTabManager(self.root, self, self.state_manager)
        self.analysis_tab_manager = AnalysisTabManager(self.root, self, self.state_manager)
        self.batch_processing_tab_manager = BatchProcessingTabManager(self.root, self, self.state_manager)
        self.comparison_tab_manager = ComparisonTabManager(self.root, self, self.state_manager)

    def init_tabs(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)
        self.initialize_tabs()
        self.video_tab_manager.init_video_tab(self.tabs['video_tab'])
        self.analysis_tab_manager.init_analysis_tab(self.tabs['analysis_tab'])
        self.batch_processing_tab_manager.init_batch_processing_tab(self.tabs['batch_tab'])
        self.comparison_tab_manager.init_comparison_tab(self.tabs['compare_tab'])

        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)


    def initialize_tabs(self):
        self.tabs = {}
        tab_titles = [
            ('video_tab', 'Video Inspection & Initialization'),
            ('analysis_tab', 'Preliminary Analysis'),
            ('batch_tab', 'Batch Processing'),
            ('compare_tab', 'Compare Experiments')
        ]

        for tab_key, tab_title in tab_titles:
            self.tabs[tab_key] = ttk.Frame(self.notebook)
            self.notebook.add(self.tabs[tab_key], text=tab_title)

    def update_ui_after_loading_experiment(self):
        self.handle_errors(self.video_tab_manager.set_initial_video_list, "Error loading .mp4 videos")
        self.handle_errors(self.analysis_tab_manager.update_image_listbox, "Error loading individual images")
        self.handle_errors(self.analysis_tab_manager.update_median_image_listbox, "Error loading median images")
        self.handle_errors(lambda: self.analysis_tab_manager.draw_borders_and_update(lambda:self.experiment_manager.update_border_image()), "Error displaying image with borders")
        self.handle_errors(self.analysis_tab_manager.update_video_listbox_svtracking,"Error loading median images")
        #self.handle_errors(self.video_tab_manager.update_pkl_listbox, "Error loading tracking files")

    def handle_errors(self, func, error_message):
        try:
            func()
        except Exception as e:
            self.log(f"{error_message}: {e}")

    def create_logging_area(self):
        self.log_frame = tk.Frame(self.root)
        self.log_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        self.log_text = tk.Text(self.log_frame, height=8)
        self.log_text.pack(fill=tk.X, expand=True)

    def log(self, message):
        """Log messages in the logging area."""
        if "Progress" in message:
            self.update_progress_line(message)
        else:
            self.add_log_message(message)

        self.log_text.see(tk.END)
        self.log_text.update_idletasks()  # Force the refresh of the UI

    def update_progress_line(self, message):
        last_line_index = self.log_text.index("end-1c linestart")
        self.log_text.delete(last_line_index, 'end-1c')
        self.log_text.insert(last_line_index, message)
        self.last_progress_line_index = last_line_index

    def add_log_message(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.last_progress_line_index = None


    def on_tab_change(self, event):
        selected_tab = self.notebook.select()
        tab_text = self.notebook.tab(selected_tab, "text")
        self.log(f"Switched to tab: {tab_text}")

        # Stop ongoing video playback
        self.video_manager.is_playing = False
        if self.video_manager.cap:
            self.video_manager.cap.release()  # Properly release any loaded video

        # Reset video display: If you switch to a non-video tab, clear the label
        if self.video_manager.label:
            self.video_manager.label.config(image='')  # Clear any displayed image

        # Assign the appropriate display label and scrollbar upon returning to a video-related tab
        if tab_text == 'Video Inspection & Initialization':
            self.video_manager.set_display_label(self.video_tab_manager.video_label)
            self.video_manager.set_scrollbar(self.video_tab_manager.video_scrollbar)
        elif tab_text == 'Preliminary Analysis':
            self.update_ui_after_loading_experiment()
            self.video_manager.set_display_label(self.analysis_tab_manager.tracking_image_label)
            self.video_manager.set_scrollbar(self.analysis_tab_manager.tracking_scrollbar)
