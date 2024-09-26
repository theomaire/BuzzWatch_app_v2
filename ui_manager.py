import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import filedialog
import json
from logger import MultiLogger
import sys
from threading import Thread
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from tkinter import Toplevel
import pandas as pd
import pickle
import numpy as np
from tkcalendar import DateEntry
PATH_TO_MODULE = "/Users/tmaire/Documents/BuzzWatch_analysis/"
sys.path.append(PATH_TO_MODULE) 
from buzzwatch_data_analysis.single_video_analysis import single_video_analysis  # Add this line


class UIManager:
    def __init__(self, root, experiment_manager, video_manager, image_manager, log_func):
        self.root = root
        self.experiment_manager = experiment_manager
        self.video_manager = video_manager
        self.image_manager = image_manager
        self.log = log_func
        self.video_start_time_label = None  # Add this attribute

        self.experiment_details = []
        self.experiment_dates = {}
        self.experiment_frames = []  # Initialize the ex


    def update_ui_after_loading_experiment(self):
        try:
            self.update_video_listbox()
        except Exception:
            self.log("Error loading .mp4 videos")
        try:
            self.update_image_listbox()
        except Exception:
            self.log("Error individual images")

        try:
            self.update_median_image_listbox()
        except Exception:
            self.log("Error averaged images")

        try:
            self.draw_borders_and_update(lambda:self.experiment_manager.update_border_image())
        except Exception:
            self.log("Error Display image with borders")

        try: 
            self.update_pkl_listbox()
        except Exception:
            self.log("Error loading tracking files")

        try:
            self.update_video_listbox_svtracking()
        except Exception:
            self.log("Error loading video to analyze")

        # Save the loaded experiment path to config


    def init_tabs(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)

        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab_batch = ttk.Frame(self.notebook)
        self.tab_compare = ttk.Frame(self.notebook)


        self.notebook.add(self.tab1, text='Video Inspection & Initialization')
        self.notebook.add(self.tab2, text='Preliminary Analysis')
        self.notebook.add(self.tab_batch, text="Batch Processing")
        self.notebook.add(self.tab_compare, text="Compare Experiments")


        self.init_tab1()
        self.init_tab2()
        self.init_batch_processing_tab(self.tab_batch)
        self.init_comparison_tab(self.tab_compare)



        #self.update_ui_after_loading_experiment()

    def init_tab1(self):
        # Configure the main grid for the tab
        self.tab1.grid_rowconfigure(0, weight=0)  # Top frame
        self.tab1.grid_rowconfigure(1, weight=1)  # Middle frame (video display)
        self.tab1.grid_rowconfigure(2, weight=0)  # Bottom frame (video controls)
        self.tab1.grid_columnconfigure(0, weight=1, minsize=300)  # Left frame
        self.tab1.grid_columnconfigure(1, weight=4)  # Right frame

        # Top frame for experiment control buttons
        top_frame = tk.Frame(self.tab1)
        top_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

        load_experiment_button = tk.Button(top_frame, text="Load Existing Experiment", command=lambda: self.experiment_manager.load_experiment(update_ui_func=self.update_ui_after_loading_experiment))
        load_experiment_button.pack(side=tk.LEFT, padx=5)

        load_button = tk.Button(top_frame, text="Initialize New Experiment", command=lambda: self.experiment_manager.initialize_new_experiment(update_ui_func=self.update_ui_after_loading_experiment))
        load_button.pack(side=tk.LEFT, padx=5)

        # Left frame for video list and load button
        buttons_frame = tk.Frame(self.tab1)
        buttons_frame.grid(row=1, column=0, sticky="ns", padx=10, pady=10)
        buttons_frame.grid_rowconfigure(1, weight=1)  # Make the listbox expand

        video_listbox_label = tk.Label(buttons_frame, text="Available Videos")
        video_listbox_label.grid(row=0, column=0, sticky="w")

        self.video_listbox = tk.Listbox(buttons_frame, selectmode=tk.SINGLE, width=40)  # Increased width
        self.video_listbox.grid(row=1, column=0, sticky="nswe", pady=5)

        load_raw_video_button = tk.Button(buttons_frame, text="Load Raw Video", command=self.load_raw_video)
        load_raw_video_button.grid(row=2, column=0, pady=5)

        # Right frame for video display
        video_display_frame = tk.Frame(self.tab1)
        video_display_frame.grid(row=1, column=1, sticky="nswe", padx=10, pady=10)
        video_display_frame.grid_rowconfigure(0, weight=1)
        video_display_frame.grid_columnconfigure(0, weight=1)

        self.video_label = tk.Label(video_display_frame)
        self.video_label.grid(row=0, column=0, sticky="nswe")
        self.video_label.bind('<Configure>', self.on_resize_video)

        self.video_manager.set_display_label(self.video_label)

        # Label for showing video start date and time
        self.video_start_time_label = tk.Label(video_display_frame, text="")
        self.video_start_time_label.grid(row=1, column=0, sticky="ew")

        # Frame for video controls
        controls_frame = tk.Frame(self.tab1)
        controls_frame.grid(row=2, column=1, sticky="ew", padx=10, pady=10)

        previous_frame_button = tk.Button(controls_frame, text="<<", command=self.video_manager.previous_frame)
        previous_frame_button.pack(side=tk.LEFT, padx=5)

        play_button = tk.Button(controls_frame, text="Play", command=self.video_manager.play_video)
        play_button.pack(side=tk.LEFT, padx=5)

        pause_button = tk.Button(controls_frame, text="Pause", command=self.video_manager.pause_video)
        pause_button.pack(side=tk.LEFT, padx=5)

        next_frame_button = tk.Button(controls_frame, text=">>", command=self.video_manager.next_frame)
        next_frame_button.pack(side=tk.LEFT, padx=5)

        self.video_scrollbar = tk.Scale(controls_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=self.on_scroll)
        self.video_scrollbar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=10)

        self.video_manager.set_scrollbar(self.video_scrollbar)

    def init_tab2(self):
        sub_notebook = ttk.Notebook(self.tab2)
        sub_notebook.pack(fill='both', expand=True)

        extract_images_tab = ttk.Frame(sub_notebook)
        get_background_tab = ttk.Frame(sub_notebook)
        draw_borders_tab = ttk.Frame(sub_notebook)
        single_video_tracking_tab = ttk.Frame(sub_notebook)

        sub_notebook.add(extract_images_tab, text='Extract Images from Video')
        sub_notebook.add(get_background_tab, text='Get Background from Images')
        sub_notebook.add(draw_borders_tab, text='Draw Borders')
        sub_notebook.add(single_video_tracking_tab, text='Single Video Tracking')

        self.init_extract_images_tab(extract_images_tab)
        self.init_get_background_tab(get_background_tab)
        self.init_draw_borders_tab(draw_borders_tab)
        self.init_single_video_tracking_tab(single_video_tracking_tab)

    def init_extract_images_tab(self, tab):
        force_rerun_var = tk.IntVar()

        options_frame = tk.Frame(tab)
        options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        get_images_button = tk.Button(options_frame, text="Run", command=lambda: self.experiment_manager.get_images_from_video(force_rerun_var.get()))
        get_images_button.pack(side=tk.LEFT, padx=5)

        update_images_button = tk.Button(options_frame, text="Update image list", command=lambda: self.update_image_listbox())
        update_images_button.pack(side=tk.LEFT, padx=5)

        force_rerun_check = tk.Checkbutton(options_frame, text="Force Re-run", variable=force_rerun_var)
        force_rerun_check.pack(side=tk.LEFT, padx=5)
        
        image_main_frame = tk.Frame(tab)
        image_main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        image_listbox_frame = tk.Frame(image_main_frame, width=300)
        image_listbox_frame.pack(side=tk.LEFT, fill=tk.Y)

        image_scrollbar = tk.Scrollbar(image_listbox_frame)
        image_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.image_manager.image_listbox = tk.Listbox(image_listbox_frame, yscrollcommand=image_scrollbar.set, selectmode=tk.SINGLE, width=50)
        self.image_manager.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        image_scrollbar.config(command=self.image_manager.image_listbox.yview)

        self.image_manager.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)

        image_display_frame = tk.Frame(image_main_frame)
        image_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.image_manager.image_label = tk.Label(image_display_frame)
        self.image_manager.image_label.pack(fill=tk.BOTH, expand=True)

    def init_get_background_tab(self, tab):
        force_rerun_var = tk.IntVar()

        options_frame = tk.Frame(tab)
        options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        get_background_button = tk.Button(options_frame, text="Run", command=lambda: self.experiment_manager.get_background_from_images(force_rerun_var.get()))
        get_background_button.pack(side=tk.LEFT, padx=5)

        update_background_button = tk.Button(options_frame, text="Update bck image list", command=lambda: self.update_median_image_listbox())
        update_background_button.pack(side=tk.LEFT, padx=5)

        force_rerun_check = tk.Checkbutton(options_frame, text="Force Re-run", variable=force_rerun_var)
        force_rerun_check.pack(side=tk.LEFT, padx=5)

        image_main_frame = tk.Frame(tab)
        image_main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        image_listbox_frame = tk.Frame(image_main_frame, width=300)
        image_listbox_frame.pack(side=tk.LEFT, fill=tk.Y)

        image_scrollbar = tk.Scrollbar(image_listbox_frame)
        image_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.image_manager.median_image_listbox = tk.Listbox(image_listbox_frame, yscrollcommand=image_scrollbar.set, selectmode=tk.SINGLE, width=50)
        self.image_manager.median_image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        image_scrollbar.config(command=self.image_manager.median_image_listbox.yview)

        self.image_manager.median_image_listbox.bind('<<ListboxSelect>>', self.on_median_image_select)

        image_display_frame = tk.Frame(image_main_frame)
        image_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.image_manager.median_image_label = tk.Label(image_display_frame)
        self.image_manager.median_image_label.pack(fill=tk.BOTH, expand=True)

    def on_image_select(self, event):
        if not self.experiment_manager.experiment:
            self.log("No experiment loaded.")
            return

        selected_image = self.image_manager.image_listbox.curselection()
        if not selected_image:
            self.log("No image selected from the list.")
            return

        images_path = os.path.join(self.experiment_manager.experiment.folder_analysis, "individual_images")
        image_file = os.path.join(images_path, self.image_manager.image_listbox.get(selected_image[0]))
        self.log(f"Displaying image: {image_file}")
        self.image_manager.display_image(image_file, self.image_manager.image_label)

    def on_median_image_select(self, event):
        if not self.experiment_manager.experiment:
            self.log("No experiment loaded.")
            return

        selected_image = self.image_manager.median_image_listbox.curselection()
        if not selected_image:
            self.log("No image selected from the list.")
            return

        images_path = os.path.join(self.experiment_manager.experiment.folder_analysis, "images_mortality")
        image_file = os.path.join(images_path, self.image_manager.median_image_listbox.get(selected_image[0]))
        self.log(f"Displaying image: {image_file}")
        self.image_manager.display_image(image_file, self.image_manager.median_image_label)

    def init_draw_borders_tab(self, tab):
        buttons_frame = tk.Frame(tab)
        buttons_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        update_image_border_button = tk.Button(buttons_frame, text="Show background with borders", command=lambda: self.draw_borders_and_update(lambda: self.experiment_manager.update_border_image()))
        update_image_border_button.pack(side=tk.TOP, pady=5)

        draw_cage_button = tk.Button(buttons_frame, text="Draw Cage Borders", command=lambda: self.draw_borders_and_update(lambda: self.experiment_manager.draw_cage_borders()))
        draw_cage_button.pack(side=tk.TOP, pady=5)

        draw_sugar_feeder_button = tk.Button(buttons_frame, text="Draw Sugar Feeder Borders", command=lambda: self.draw_borders_and_update(lambda: self.experiment_manager.draw_sugar_feeder_borders()))
        draw_sugar_feeder_button.pack(side=tk.TOP, pady=5)

        draw_control_button = tk.Button(buttons_frame, text="Draw Control Borders", command=lambda: self.draw_borders_and_update(lambda: self.experiment_manager.draw_control_borders()))
        draw_control_button.pack(side=tk.TOP, pady=5)

        draw_square_3_button = tk.Button(buttons_frame, text="Draw Square 3 Borders", command=lambda: self.draw_borders_and_update(lambda: self.experiment_manager.draw_square_3()))
        draw_square_3_button.pack(side=tk.TOP, pady=5)

        draw_square_4_button = tk.Button(buttons_frame, text="Draw Square 4 Borders", command=lambda: self.draw_borders_and_update(lambda: self.experiment_manager.draw_square_4()))
        draw_square_4_button.pack(side=tk.TOP, pady=5)

        image_display_frame = tk.Frame(tab)
        image_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.borders_image_label = tk.Label(image_display_frame)
        self.borders_image_label.pack(fill=tk.BOTH, expand=True)

    def draw_borders_and_update(self, draw_func):
        draw_func()
        self.display_borders_image()

    def init_single_video_tracking_tab(self, tab):
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=0)

        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=4)


        buttons_frame = tk.Frame(tab)
        buttons_frame.grid(row=0, column=0,rowspan=2, sticky="ns", padx=10, pady=10)

        # Video names available
        pkl_listbox_label = tk.Label(buttons_frame, text="Videos files available for analysis")
        pkl_listbox_label.pack(anchor=tk.W)

        self.video_listbox_svtracking = tk.Listbox(buttons_frame, selectmode=tk.SINGLE, width=50)
        self.video_listbox_svtracking.pack(side=tk.TOP, fill=tk.BOTH, expand=False, pady=5)

        # Create a frame to hold the buttons
        buttons_container = tk.Frame(buttons_frame)
        buttons_container.pack(side=tk.TOP, fill=tk.X, pady=5)

        # Analyze Video Button on the left
        analyze_video_button = tk.Button(buttons_container, text="Analyze Video", command=self.analyze_selected_video)
        analyze_video_button.pack(side=tk.LEFT, padx=5)

        # Update Tracking Results Button on the right
        update_tracking_video_button = tk.Button(buttons_container, text="Update Tracking results", command=self.update_pkl_listbox)
        update_tracking_video_button.pack(side=tk.RIGHT, padx=5)

        # Final Tracking Files Listbox
        pkl_listbox_label = tk.Label(buttons_frame, text="Final Tracking Files:")
        pkl_listbox_label.pack(anchor=tk.W)

        self.pkl_listbox = tk.Listbox(buttons_frame, selectmode=tk.SINGLE, width=50)
        self.pkl_listbox.pack(fill=tk.BOTH, expand=True, pady=5)

        load_tracking_video_button = tk.Button(buttons_frame, text="Load Tracking Video", command=self.load_tracking_video)
        load_tracking_video_button.pack(side=tk.LEFT, pady=5)

        # Inspect Results Button
        inspect_time_series_button = tk.Button(buttons_frame, text="Show time series", command=self.inspect_time_series)
        inspect_time_series_button.pack(side=tk.LEFT, pady=5)

        inspect_hist_button = tk.Button(buttons_frame, text="Show histograms", command=self.inspect_histogram)
        inspect_hist_button.pack(side=tk.LEFT, pady=5)

        inspect_traj_button = tk.Button(buttons_frame, text="Show flight traj", command=self.inspect_trajectories)
        inspect_traj_button.pack(side=tk.LEFT, pady=5)

        # New Button for Flight Metrics
        extract_metrics_button = tk.Button(buttons_frame, text="Extract Flight Metrics", command=self.show_flight_metrics)
        extract_metrics_button.pack(side=tk.LEFT, pady=5)

        # Right frame for video display

        #  and controls
        video_display_frame = tk.Frame(tab)
        video_display_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        video_display_frame.grid_rowconfigure(0, weight=1)
        video_display_frame.grid_columnconfigure(0, weight=1)

        self.tracking_image_label = tk.Label(video_display_frame)
        #self.tracking_image_label.pack(fill=tk.BOTH, expand=True)
        self.tracking_image_label.grid(row=0, column=0, sticky="nswe")
        self.tracking_image_label.bind('<Configure>', self.on_resize_video)

        # Label for showing video start date and time
        self.video_start_time_label = tk.Label(video_display_frame, text="")
        self.video_start_time_label.grid(row=1, column=0, sticky="ew")


        # Controls frame
        controls_frame = tk.Frame(tab)
        controls_frame.grid(row=1, column=1, sticky="ew", padx=10, pady=10)


        play_button = tk.Button(controls_frame, text="Play", command=self.video_manager.play_video)
        play_button.pack(side=tk.LEFT, padx=5)

        pause_button = tk.Button(controls_frame, text="Pause", command=self.video_manager.pause_video)
        pause_button.pack(side=tk.LEFT, padx=5)

        # Left arrow button
        left_button = tk.Button(controls_frame, text="<<", command=self.video_manager.previous_frame)
        left_button.pack(side=tk.LEFT, padx=5)

        # Right arrow button
        right_button = tk.Button(controls_frame, text=">>", command=self.video_manager.next_frame)
        right_button.pack(side=tk.RIGHT, padx=5)

        self.tracking_scrollbar = tk.Scale(controls_frame, from_=0, to=self.video_manager.total_frames-1, orient=tk.HORIZONTAL, command=self.on_scroll)
        self.tracking_scrollbar.pack(fill=tk.X, expand=True, padx=5, pady=5)
        self.video_manager.set_scrollbar(self.tracking_scrollbar)


    def load_raw_video(self):
        selected_video = self.video_listbox.curselection()
        if not selected_video:
            self.log("No video selected from the list.")
            return
        video_path = os.path.join(self.experiment_manager.folder_videos,self.video_listbox.get(selected_video[0]))
        self.log(f"Loading raw video: {video_path}")
        if self.video_manager.load_video(video_path):
            self.video_manager.set_display_label(self.video_label)
            #self.video_manager.load_tracking_data(tracking_file)
            #self.video_manager.set_display_tracking(True)
            self.video_manager.set_display_tracking(False)
            self.video_manager.set_scrollbar(self.video_scrollbar)
            self.video_scrollbar.config(to=self.video_manager.total_frames - 1)
            self.video_manager.show_frame(0)

            # Get and display the video's start datetime
            start_time = self.video_manager.get_datetime_from_file_name(self.video_listbox.get(selected_video[0]))
            self.set_video_start_time(start_time)


    def load_tracking_video(self):
        selected_video = self.pkl_listbox.curselection()
        if selected_video:
            video_name = self.get_video_name_from_tracking_file(self.pkl_listbox.get(selected_video[0]))
            video_path = os.path.join(self.experiment_manager.folder_videos,video_name+".mp4")
            tracking_file = os.path.join(self.experiment_manager.folder_analysis,"final_tracking_data",self.pkl_listbox.get(selected_video[0]))
            if self.video_manager.load_video(video_path):
                self.video_manager.load_tracking_data(tracking_file)
                self.video_manager.label = self.tracking_image_label
                self.video_manager.set_display_tracking(True)
                self.video_manager.set_scrollbar(self.tracking_scrollbar)
                self.tracking_scrollbar.config(to=self.video_manager.total_frames - 1)
                self.video_manager.show_frame(0)

             # Get and display the video's start datetime
            start_time = self.video_manager.get_datetime_from_file_name(video_name)
            self.set_video_start_time(start_time)

    def on_scroll(self, value):
        frame_index = int(value)
        self.video_manager.show_frame(frame_index)

    def on_resize_video(self, event):
        new_width = event.width
        new_height = event.height
        frame_index = int(self.video_scrollbar.get())
        self.video_manager.show_frame(frame_index)

    def set_video_start_time(self, start_time):
        """Display the start date and time of the video."""
        if start_time:
            formatted_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
            self.video_start_time_label.config(text=f"Video Start Time: {formatted_time}")
        else:
            self.video_start_time_label.config(text="")


    def update_video_listbox(self):
        self.video_listbox.delete(0, tk.END)
        self.experiment_manager.video_files = sorted([os.path.join(self.experiment_manager.folder_videos, f) for f in os.listdir(self.experiment_manager.folder_videos) if f.endswith('.mp4') and f.startswith("Cage")])
        for video in self.experiment_manager.video_files:
            self.video_listbox.insert(tk.END, os.path.basename(video))

    def update_image_listbox(self):
        if self.experiment_manager.experiment:
            images_path = os.path.join(self.experiment_manager.experiment.folder_analysis, "individual_images")
            image_files = [f for f in os.listdir(images_path) if f.endswith('.png')]
            image_files.sort()
            self.image_manager.image_listbox.delete(0, tk.END)
            for image in image_files:
                self.image_manager.image_listbox.insert(tk.END, image)

    def update_median_image_listbox(self):
        if self.experiment_manager.experiment:
            images_path = os.path.join(self.experiment_manager.experiment.folder_analysis, "images_mortality")
            median_image_files = [f for f in os.listdir(images_path) if f.endswith('.png')]
            median_image_files.sort()
            self.image_manager.median_image_listbox.delete(0, tk.END)
            for image in median_image_files:
                self.image_manager.median_image_listbox.insert(tk.END, image)

    def display_borders_image(self):
        background_with_borders_path = os.path.join(self.experiment_manager.experiment.folder_analysis, "background_with_borders.png")
        try:
            img = Image.open(background_with_borders_path)
            img = img.resize((int(img.width * self.image_manager.resize_factor), int(img.height * self.image_manager.resize_factor)))
            imgtk = ImageTk.PhotoImage(image=img)
            self.borders_image_label.imgtk = imgtk
            self.borders_image_label.config(image=imgtk)
        except Exception as e:
            self.log(f"Error displaying background with borders: {str(e)}")

    def update_video_listbox_svtracking(self):
        if not self.experiment_manager.folder_videos:
            self.log("Folder path for videos is not set.")
            return
        video_files = [f for f in os.listdir(os.path.join(self.experiment_manager.folder_analysis, "images_mortality")) if f.endswith('.png')]
        video_names = [os.path.splitext(video_file)[0] for video_file in video_files]
        video_names.sort()

        if not video_files:
            self.log("No videos found in the folder.")
            return

        self.video_listbox_svtracking.delete(0, tk.END)
        for video in video_names:
            self.video_listbox_svtracking.insert(tk.END, video)
        self.log("Loaded video list from video folder.")
    
    def update_pkl_listbox(self):
        # Assuming your final tracking files are named in a way you can derive them from the video_name
        if self.experiment_manager.experiment:
            final_tracking_path = os.path.join(self.experiment_manager.experiment.folder_analysis, "final_tracking_data")
            tracking_files = [f for f in os.listdir(final_tracking_path) if f.startswith('forward')]
            tracking_files.sort()
            self.pkl_listbox.delete(0, tk.END)

        for file in tracking_files:
            self.pkl_listbox.insert(tk.END, file)


    def analyze_selected_video(self):
        selected_video = self.video_listbox_svtracking.curselection()
        if not selected_video:
            self.log("No video selected for analysis.")
            return
        video_name = self.video_listbox_svtracking.get(selected_video[0])

        log_file_path = os.path.join(self.experiment_manager.folder_analysis, "log_analysis", f"{video_name}.log")
        # Construct the new path
        log_analysis_path = os.path.join(self.experiment_manager.folder_analysis, "log_analysis")

        # Create the directory (including any necessary parent directories)
        os.makedirs(log_analysis_path, exist_ok=True)
        os.path.join(self.experiment_manager.folder_analysis, "log_analysis")
        logger = MultiLogger(self.log, log_file_path)

        # Redirect stdout to the MultiLogger
        old_stdout = sys.stdout
        sys.stdout = logger

        #self.experiment_manager.run_tracking_analysis(video_name)

        # Ensure the processing happens in a separate thread to keep UI responsive
        
        def run_analysis():
            self.experiment_manager.run_tracking_analysis(video_name)
            self.root.after(0, lambda: [self.log(f"Finished analyzing {video_name}"), self.update_log_text()])


        # Start the analysis in a new thread
        analysis_thread = Thread(target=run_analysis)
        analysis_thread.start()

        # Reset stdout
        sys.stdout = old_stdout

    def update_log_text(self):
        self.log_text.see(tk.END)  # Scroll to the end of the Text widget
        self.log_text.update_idletasks()  # Force the refresh of the UI

    def get_video_name_from_tracking_file(self,tracking_file_name):
        """
        Extracts the base video name from the tracking file name.

        Example:
            input: "forward_mosq_tracks_Cage04_KUM__240819_mosquipi4_140452_v12"
            output: "Cage04_KUM__240819_mosquipi4_140452"
        """
        # Remove the prefix
        if tracking_file_name.startswith("forward_mosq_tracks_"):
            base_name = tracking_file_name[len("forward_mosq_tracks_"):]
        else:
            base_name = tracking_file_name


        return base_name
    
    def inspect_time_series(self):
        # Create a new top-level window
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)  # Adjust the space between plots

        # Flatten the 2D array of axes for easier access
        axs = axs.flatten()

        # Population Variables Plots
        data = self.video_manager.mosquito_tracks.population_variables  # Retrieve the population data

        time_intervals = np.arange(len(data['numb_mosquitos_flying']))
        print(len(data['numb_mosquitos_flying']))
        minutes = time_intervals / (60*25)

        axs[0].plot(minutes,data['numb_mosquitos_flying'], color='blue')
        axs[0].set_title("Number of Mosquitos Flying Over Time")
        axs[0].set_xlabel("minutes")
        axs[0].set_ylabel("Count")

        axs[1].plot(minutes,data['numb_mosquitos_sugar'], color='green')
        axs[1].plot(minutes,data['numb_mosquitos_hs'], color='orange')
        axs[1].plot(minutes,data['numb_mosquitos_left_ctrl'], color='black')
        axs[1].plot(minutes,data['numb_mosquitos_right_ctrl'], color='grey')
        axs[1].set_title("Number of Mosquitos at on side windows")
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Count")


        # # Individual Statistics Histograms
        # individual_data = self.video_manager.mosquito_tracks.individual_variables

        # axs[1].hist(individual_data["flight_duration"], bins=20, color='purple', edgecolor='black')
        # axs[1].set_title("Histogram of Flight Duration")
        # axs[1].set_xlabel("Duration")
        # axs[1].set_ylabel("Frequency")

        # axs[5].hist(individual_data["average_speed"], bins=20, color='brown', edgecolor='black')
        # axs[5].set_title("Histogram of Flight Speed")
        # axs[5].set_xlabel("Speed")
        # axs[5].set_ylabel("Frequency")

        plt.tight_layout()
    
        # Now, since we aren't using a canvas, we can show the figure in a separate window
        plt.show()



    def inspect_histogram(self):
                # Create a new top-level window
        fig, axs = plt.subplots(3, 1, figsize=(5, 8))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)  # Adjust the space between plots

        # Flatten the 2D array of axes for easier access
        axs = axs.flatten()

        # Population Variables Plots
        data = self.video_manager.mosquito_tracks.individual_variables  # Retrieve the population data


        axs[0].hist(data["average_speed"], bins=20, color='purple', edgecolor='black')
        axs[0].set_title("Histogram of Flight Duration")
        axs[0].set_xlabel("Duration")
        axs[0].set_ylabel("Frequency")

        axs[1].hist(data["flight_duration"], bins=20, color='purple', edgecolor='black')
        axs[1].set_title("Histogram of Flight Duration")
        axs[1].set_xlabel("Duration")
        axs[1].set_ylabel("Frequency")

        data = self.video_manager.mosquito_tracks.resting_variables  # Retrieve the population data
        axs[2].hist(data["resting_duration"], bins=20, color='purple', edgecolor='black')
        axs[2].set_title("Histogram of Flight Duration")
        axs[2].set_xlabel("Duration")
        axs[2].set_ylabel("Frequency")
        plt.tight_layout()
    
        # Now, since we aren't using a canvas, we can show the figure in a separate window
        plt.show()


    def inspect_trajectories(self):

        fig, axes = plt.subplots(3, 6,dpi=200)
        fig.subplots_adjust(hspace=0., wspace=0.)
        fig.set_figheight(20)
        fig.set_figwidth(20)
        axes  = axes.reshape(-1)

        # Population Variables Plots
        mosquito_tracks = self.video_manager.mosquito_tracks  # Retrieve the population data

        axes = self.experiment_manager.experiment.plot_sample_flight_trajectories_from_video(axes,mosquito_tracks)
        plt.show()


        
    def save_config(self, config):
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f)

    def load_config(self):
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    

############## Third tab for batch processing
    def init_batch_processing_tab(self, tab):
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)
        
        # Create a frame for batch processing controls
        controls_frame = tk.Frame(tab)
        controls_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        generate_script_button = tk.Button(controls_frame, text="Generate Batch Processing Script", command=self.generate_batch_script)
        generate_script_button.pack(side=tk.TOP, fill=tk.X, expand=True, pady=5)

        concatenate_button = tk.Button(controls_frame, text="Concatenate and Save Data", command=self.concatenate_and_save_plots)
        concatenate_button.pack(side=tk.TOP, fill=tk.X, expand=True, pady=5)

        load_data_button = tk.Button(controls_frame, text="Load Data", command=self.load_data)
        load_data_button.pack(side=tk.TOP, fill=tk.X, expand=True, pady=5)
        
        plot_rolling_avg_button = tk.Button(controls_frame, text="Plot Rolling Average", command=self.plot_rolling_average)
        plot_rolling_avg_button.pack(side=tk.TOP, fill=tk.X, expand=True, pady=5)

        plot_histograms_button = tk.Button(controls_frame, text="Plot Histograms", command=self.plot_histograms)
        plot_histograms_button.pack(side=tk.TOP, fill=tk.X, expand=True, pady=5)

        plot_scatter_bar_button = tk.Button(controls_frame, text="Plot Scatter and Bar", command=self.plot_scatter_and_bar)
        plot_scatter_bar_button.pack(side=tk.TOP, fill=tk.X, expand=True, pady=5)

        self.script_status_label = tk.Label(controls_frame, text='', fg='blue')
        self.script_status_label.pack(side=tk.TOP, pady=5)

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
        folder_analysis = self.experiment_manager.folder_analysis
        video_files = [f.replace(".png", "") for f in os.listdir(folder_analysis + "/images_mortality") if f.endswith(".png") and f.startswith("Cage")]

        all_population_data = []
        all_individual_data = []
        all_flight_metrics_data = []  # New list to store flight metrics around resting
        summary_data = []

        for video_name in video_files:
            final_data_path = os.path.join(folder_analysis, "final_tracking_data", f"forward_mosq_tracks_{video_name}")
            if os.path.exists(final_data_path):
                with open(final_data_path, 'rb') as f:
                    video_data = pickle.load(f)

                all_population_data.append(video_data.population_variables.resample('1T', label='right').mean())
                all_individual_data.append(video_data.individual_variables)
                
                # Check and append the new flight metrics data
                if hasattr(video_data, 'flight_metrics_around_resting'):
                    all_flight_metrics_data.append(video_data.flight_metrics_around_resting)

                start_time = video_data.time_stamp[0]  # Get the start time of the video
                avg_fraction_flying = video_data.population_variables['numb_mosquitos_flying'].mean()  # Get the average fraction flying per video
                total_nb_tracks = len(video_data.objects.keys())

                summary_data.append([start_time, avg_fraction_flying, total_nb_tracks])  # Append to summary data list

        if all_population_data:
            full_population_data = pd.concat(all_population_data)
            full_individual_data = pd.concat(all_individual_data)
            full_flight_metrics_data = pd.concat(all_flight_metrics_data) if all_flight_metrics_data else pd.DataFrame()

            summary_df = pd.DataFrame(summary_data, columns=['start_time', 'avg_fraction_flying', 'total_nb_tracks'])

            # Save concatenated data into a single .pkl file
            final_output_path = os.path.join(folder_analysis, "analyzed_data.pkl")
            with open(final_output_path, 'wb') as f:
                pickle.dump({
                    'population_data': full_population_data,
                    'individual_data': full_individual_data,
                    'summary_data': summary_df,
                    'flight_metrics_data': full_flight_metrics_data  # Include the new data
                }, f)

            # Notify completion
            self.log("Concatenation complete. Data saved to:")
            self.log(final_output_path)
        else:
            self.log("No data found for concatenation.")



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


######################### Comparison between experiments tab            

    def init_comparison_tab(self, tab):
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)
        
        # Frame for managing experiments
        exp_frame = tk.Frame(tab)
        exp_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")

        num_experiments_label = tk.Label(exp_frame, text="Number of Experiments:")
        num_experiments_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.num_experiments_entry = tk.Entry(exp_frame)
        self.num_experiments_entry.pack(side=tk.LEFT, padx=5)

        add_experiments_button = tk.Button(exp_frame, text="Add Experiments", command=self.add_experiment_inputs)
        add_experiments_button.pack(side=tk.LEFT, padx=5)

        self.exp_details_frame = tk.Frame(tab)
        self.exp_details_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        # Frame for selecting plot variables and options
        options_frame = tk.Frame(tab)
        options_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")

        var_label = tk.Label(options_frame, text="Select Variable(s) to Plot:")
        var_label.pack(side=tk.TOP, pady=5)

        # Listbox for multiple variable selection
        self.var_listbox = tk.Listbox(options_frame, selectmode=tk.MULTIPLE, height=6)
        for var in ['numb_mosquitos_flying', 'numb_mosquitos_sugar', 'numb_mosquitos_hs', 'numb_mosquitos_left_ctrl', 'numb_mosquitos_right_ctrl', 'flight_duration', 'average_speed']:
            self.var_listbox.insert(tk.END, var)
        self.var_listbox.pack(side=tk.LEFT, pady=5)

        resample_frame = tk.Frame(options_frame)
        resample_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        resample_label = tk.Label(resample_frame, text="Resample Interval (e.g., '1T' for 1 minute):")
        resample_label.pack(side=tk.LEFT, padx=1)

        self.resample_entry = tk.Entry(resample_frame)
        self.resample_entry.pack(side=tk.LEFT, padx=5)

        moving_avg_frame = tk.Frame(options_frame)
        moving_avg_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        moving_avg_label = tk.Label(moving_avg_frame, text="Moving Average Window (e.g., 5):")
        moving_avg_label.pack(side=tk.LEFT, padx=5)

        self.moving_avg_entry = tk.Entry(moving_avg_frame)
        self.moving_avg_entry.pack(side=tk.LEFT, padx=5)

        threshold_frame = tk.Frame(options_frame)
        threshold_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        threshold_label = tk.Label(threshold_frame, text="Threshold for Flight Duration and Average Speed:")
        threshold_label.pack(side=tk.LEFT, padx=5)

        self.threshold_entry = tk.Entry(threshold_frame)
        self.threshold_entry.pack(side=tk.LEFT, padx=5)


        # Time selection frame
        time_frame = tk.Frame(options_frame)
        time_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        start_hour_label = tk.Label(time_frame, text="Start Hour:")
        start_hour_label.pack(side=tk.LEFT, padx=5)

        self.start_hour_scale = tk.Scale(time_frame, from_=0, to=23, orient=tk.HORIZONTAL)
        self.start_hour_scale.set(0)  # Default start hour
        self.start_hour_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        end_hour_label = tk.Label(time_frame, text="End Hour:")
        end_hour_label.pack(side=tk.LEFT, padx=5)

        self.end_hour_scale = tk.Scale(time_frame, from_=0, to=23, orient=tk.HORIZONTAL)
        self.end_hour_scale.set(23)  # Default end hour
        self.end_hour_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)


        # Frame for date selection
        date_frame = tk.Frame(tab)
        date_frame.grid(row=3, column=0, padx=10, pady=5, sticky="nsew")

        start_date_frame = tk.Frame(date_frame)
        start_date_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        start_date_label = tk.Label(start_date_frame, text="Start Date:")
        start_date_label.pack(side=tk.LEFT, padx=5)

        self.start_date_combobox = ttk.Combobox(start_date_frame)
        self.start_date_combobox.pack(side=tk.LEFT, padx=5)

        end_date_frame = tk.Frame(date_frame)
        end_date_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        end_date_label = tk.Label(end_date_frame, text="End Date:")
        end_date_label.pack(side=tk.LEFT, padx=5)

        self.end_date_combobox = ttk.Combobox(end_date_frame)
        self.end_date_combobox.pack(side=tk.LEFT, padx=5)

        num_days_avg_frame = tk.Frame(date_frame)
        num_days_avg_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        num_days_avg_label = tk.Label(num_days_avg_frame, text="Number of Days to Average:")
        num_days_avg_label.pack(side=tk.LEFT, padx=5)

        self.num_days_avg_entry = tk.Entry(num_days_avg_frame)
        self.num_days_avg_entry.pack(side=tk.LEFT, padx=5)

        # Frame for buttons
        buttons_frame = tk.Frame(tab)
        buttons_frame.grid(row=4, column=0, padx=10, pady=5, sticky="nsew")

        save_load_frame = tk.Frame(buttons_frame)
        save_load_frame.pack(side=tk.LEFT, fill=tk.X, pady=5)

        save_button = tk.Button(save_load_frame, text="Save Settings", command=self.save_comparison_settings)
        save_button.pack(side=tk.LEFT, padx=5)

        load_button = tk.Button(save_load_frame, text="Load Settings", command=self.load_comparison_settings)
        load_button.pack(side=tk.LEFT, padx=5)

        plot_frame = tk.Frame(buttons_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.X, pady=5)

        plot_button = tk.Button(plot_frame, text="Plot Data", command=self.plot_experiment_comparison)
        plot_button.pack(side=tk.LEFT, padx=5)

        plot_daily_avg_button = tk.Button(plot_frame, text="Plot Daily Average", command=self.plot_daily_average)
        plot_daily_avg_button.pack(side=tk.LEFT, padx=5)

        plot_avg_over_days_button = tk.Button(plot_frame, text="Plot Avg Over Days", command=self.plot_avg_over_days)
        plot_avg_over_days_button.pack(side=tk.LEFT, padx=5)

        # Frame for experiment inclusion checkboxes
        self.exp_inclusion_frame = tk.Frame(tab)
        self.exp_inclusion_frame.grid(row=5, column=0, padx=10, pady=5, sticky="nsew")


    def add_experiment_inputs(self):
        num_experiments = int(self.num_experiments_entry.get())
        for frame in self.experiment_frames:
            frame.pack_forget()

        self.experiment_frames = []
        self.experiment_details = []
        self.experiment_dates = {}

        for i in range(num_experiments):
            frame = tk.Frame(self.exp_details_frame)
            frame.pack(side=tk.TOP, pady=5, fill=tk.X)

            exp_label = tk.Label(frame, text=f"Experiment {i + 1} Path:")
            exp_label.pack(side=tk.LEFT, padx=5)

            exp_path_entry = tk.Entry(frame, width=35)
            exp_path_entry.pack(side=tk.LEFT, padx=5)

            color_label = tk.Label(frame, text="Color:")
            color_label.pack(side=tk.LEFT, padx=5)

            color_combobox = ttk.Combobox(frame, values=['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
            color_combobox.pack(side=tk.LEFT, padx=5)

            alias_label = tk.Label(frame, text="Alias:")
            alias_label.pack(side=tk.LEFT, padx=5)

            alias_entry = tk.Entry(frame, width=20)
            alias_entry.pack(side=tk.LEFT, padx=5)

            include_var = tk.IntVar(value=1)  # Defaults to included
            include_checkbox = tk.Checkbutton(frame, variable=include_var)
            include_checkbox.pack(side=tk.LEFT, padx=5)

            pre_load_button = tk.Button(frame, text="Pre-load", command=lambda e=exp_path_entry, c=color_combobox: self.preload_experiment_data(e, c))
            pre_load_button.pack(side=tk.LEFT, padx=5)

            self.experiment_frames.append(frame)
            self.experiment_details.append((exp_path_entry, color_combobox, include_var, alias_entry))  # Store alias entry




    def update_date_comboboxes(self):
        if not self.experiment_dates:
            return
        
        common_dates = None
        for dates in self.experiment_dates.values():
            date_range = pd.date_range(dates[0], dates[1])
            if common_dates is None:
                common_dates = set(date_range)
            else:
                common_dates &= set(date_range)

        if common_dates:
            sorted_dates = sorted(list(common_dates))
            date_strings = [date.strftime('%Y-%m-%d') for date in sorted_dates]
            self.start_date_combobox['values'] = date_strings
            self.end_date_combobox['values'] = date_strings

            if date_strings:
                self.start_date_combobox.set(date_strings[0])
                self.end_date_combobox.set(date_strings[-1])

    def preload_experiment_data(self, exp_path_entry, color_combobox):
        exp_path = exp_path_entry.get()
        if not exp_path:
            return

        try:

            #folder_analysis = self.experiment_manager.folder_analysis
            analyzed_data_path = os.path.join(exp_path, "analyzed_data.pkl")

            if os.path.exists(analyzed_data_path):
                with open(analyzed_data_path, 'rb') as f:
                    analyzed_data = pickle.load(f)

                full_population_data = analyzed_data['population_data']
                full_individual_data = analyzed_data['individual_data']
                summary_data = analyzed_data['summary_data']

            #full_population_data = pd.read_pickle(os.path.join(exp_path, "full_population_data.pkl"))
            min_date = full_population_data.index.min()
            max_date = full_population_data.index.max()
            self.experiment_dates[exp_path] = (min_date, max_date)
            self.update_date_comboboxes()
            self.log(f"Pre-loaded data for {exp_path}: {min_date} to {max_date}")
        except Exception as e:
            self.log(f"Error loading data from {exp_path}: {e}")

    def save_comparison_settings(self):
        settings = {
            "num_experiments": self.num_experiments_entry.get(),
            "experiments": [{
                "path": exp_path_entry.get(),
                "color": color_combobox.get(),
                "include": include_var.get(),
                "alias": alias_entry.get()
            } for exp_path_entry, color_combobox, include_var, alias_entry in self.experiment_details],
            "selected_vars": [self.var_listbox.get(i) for i in self.var_listbox.curselection()],
            "resample_interval": self.resample_entry.get(),
            "moving_avg_window": self.moving_avg_entry.get(),
            "start_date": self.start_date_combobox.get(),
            "end_date": self.end_date_combobox.get(),
            "num_days_avg": self.num_days_avg_entry.get(),
            "threshold": self.threshold_entry.get(),
        }
        save_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(settings, f)
            self.log(f"Settings saved to {save_path}")



    def load_comparison_settings(self):
        load_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if load_path:
            with open(load_path, 'r') as f:
                settings = json.load(f)
            
            self.num_experiments_entry.delete(0, tk.END)
            self.num_experiments_entry.insert(0, settings["num_experiments"])
            self.add_experiment_inputs()

            for i, (exp_frame, exp_settings) in enumerate(zip(self.experiment_details, settings["experiments"])):
                exp_path_entry, color_combobox, include_var, alias_entry = exp_frame
                exp_path_entry.delete(0, tk.END)
                exp_path_entry.insert(0, exp_settings["path"])
                color_combobox.set(exp_settings["color"])
                include_var.set(exp_settings["include"])
                alias_entry.delete(0, tk.END)
                alias_entry.insert(0, exp_settings["alias"])
                self.preload_experiment_data(exp_path_entry, color_combobox)

            selected_vars = settings["selected_vars"]
            self.var_listbox.selection_clear(0, tk.END)
            for var in selected_vars:
                index = self.var_listbox.get(0, tk.END).index(var)
                self.var_listbox.selection_set(index)

            self.resample_entry.delete(0, tk.END)
            self.resample_entry.insert(0, settings["resample_interval"])

            self.moving_avg_entry.delete(0, tk.END)
            self.moving_avg_entry.insert(0, settings["moving_avg_window"])

            self.start_date_combobox.set(settings["start_date"])
            self.end_date_combobox.set(settings["end_date"])
            
            self.num_days_avg_entry.delete(0, tk.END)
            self.num_days_avg_entry.insert(0, settings["num_days_avg"])

            self.threshold_entry.delete(0, tk.END)
            self.threshold_entry.insert(0, settings.get("threshold", ""))

            self.log(f"Settings loaded from {load_path}")



    def plot_experiment_comparison(self):
        selected_vars = [self.var_listbox.get(i) for i in self.var_listbox.curselection()]
        resample_interval = self.resample_entry.get()
        moving_avg_window = self.moving_avg_entry.get()
        start_date_str = self.start_date_combobox.get()
        end_date_str = self.end_date_combobox.get()
        threshold = self.threshold_entry.get()  # Get the threshold value

        if not start_date_str or not end_date_str:
            self.log("Start date or end date not selected.")
            return

        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        line_styles = ['-', '--', '-.', ':']  # Different line styles for variables

        for exp_path_entry, color_combobox, include_var, alias_entry in self.experiment_details:
            if not include_var.get():
                continue  # Skip if not included

            exp_path = exp_path_entry.get()
            color = color_combobox.get()
            alias = alias_entry.get()

            if not exp_path or not color or not alias:
                continue
            
            try:
                analyzed_data_path = os.path.join(exp_path, "analyzed_data.pkl")
                if os.path.exists(analyzed_data_path):
                    with open(analyzed_data_path, 'rb') as f:
                        analyzed_data = pickle.load(f)

                    full_population_data = analyzed_data['population_data']
                    full_individual_data = analyzed_data['individual_data']
                    summary_data = analyzed_data['summary_data']
                #full_population_data = pd.read_pickle(os.path.join(exp_path, "full_population_data.pkl"))
                #full_individual_data = pd.read_pickle(os.path.join(exp_path, "full_individual_data.pkl"))

                for i, var in enumerate(selected_vars):
                    if var in full_population_data.columns:
                        var_data = full_population_data[var]
                    elif var in full_individual_data.columns:
                        var_data = full_individual_data[var]
                        # Apply threshold filtering for "flight_duration" and "average_speed"
                        if var in ["flight_duration", "average_speed"] and threshold:
                            threshold_value = float(threshold)
                            var_data = var_data[var_data > threshold_value]
                    else:
                        self.log(f"Variable '{var}' not found in experiment data.")
                        continue

                    var_data = self.crop_time(start_date, end_date, var_data)

                    if resample_interval:
                        var_data = var_data.resample(resample_interval).mean()

                    if moving_avg_window:
                        moving_avg_window = int(moving_avg_window)
                        var_data = var_data.rolling(window=moving_avg_window).mean()

                    style = line_styles[i % len(line_styles)]  # Cycle through line styles
                    var_data.plot(ax=ax, label=f"{alias} - {var}", color=color, linestyle=style)

            except Exception as e:
                self.log(f"Error loading data from {exp_path}: {e}")

        ax.set_title(f"Comparison of Selected Variables")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()

        plt.show()


    def plot_daily_average(self):
        selected_vars = [self.var_listbox.get(i) for i in self.var_listbox.curselection()]
        resample_interval = self.resample_entry.get()
        moving_avg_window = self.moving_avg_entry.get()
        start_date_str = self.start_date_combobox.get()
        end_date_str = self.end_date_combobox.get()
        threshold = self.threshold_entry.get() 

        if not start_date_str or not end_date_str:
            self.log("Start date or end date not selected.")
            return

        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        line_styles = ['-', '--', '-.', ':']  # Different line styles for variables

        for exp_path_entry, color_combobox, include_var, alias_entry in self.experiment_details:
            if not include_var.get():
                continue  # Skip if not included

            exp_path = exp_path_entry.get()
            color = color_combobox.get()
            alias = alias_entry.get()

            if not exp_path or not color or not alias:
                continue

            try:
                analyzed_data_path = os.path.join(exp_path, "analyzed_data.pkl")
                if os.path.exists(analyzed_data_path):
                    with open(analyzed_data_path, 'rb') as f:
                        analyzed_data = pickle.load(f)

                    full_population_data = analyzed_data['population_data']
                    full_individual_data = analyzed_data['individual_data']
                    summary_data = analyzed_data['summary_data']

                    for i, var in enumerate(selected_vars):
                        if var in full_population_data.columns:
                            var_data = full_population_data[var]

                        elif var in full_individual_data.columns:
                            var_data = full_individual_data[var]

                            if var in ["flight_duration", "average_speed"] and threshold:
                                threshold_value = float(threshold)
                                var_data = var_data[var_data > threshold_value]

                        # Make sure the data includes all times between start and end dates
                        data_cropped = var_data
                        data_cropped = self.crop_time(start_date, end_date, data_cropped)

                        if resample_interval:
                            data_cropped = data_cropped.resample(resample_interval).mean()
                            data_cropped = data_cropped.interpolate()

                        if moving_avg_window:
                            moving_avg_window = int(moving_avg_window)
                            data_cropped = data_cropped.rolling(window=moving_avg_window).mean()

                        avg_data = self.daily_average(start_date, end_date, data_cropped)

                        avg = avg_data.mean(axis=1)
                        std_dev = avg_data.std(axis=1)

                        style = line_styles[i % len(line_styles)]  # Cycle through line styles
                        ax.plot(avg.index, avg.values, label=f"{alias} - {var}", color=color, linestyle=style)
                        ax.fill_between(avg.index, avg - std_dev, avg + std_dev, color=color, alpha=0.3)

            except Exception as e:
                self.log(f"Error loading data from {exp_path}: {e}")

        ax.set_title(f"Daily Average of Selected Variables")
        ax.set_xlabel("Hour of the Day")
        ax.set_ylabel("Value")
        ax.legend()

        plt.show()



    def daily_average(self, start_date, end_date, df):
        nb_days = (end_date - start_date).days

        all_days = []
        for day in range(nb_days):
            day_start = start_date + pd.Timedelta(days=day)
            day_end = day_start + pd.Timedelta(days=1)

            day_data = self.crop_time(day_start, day_end, df)
            all_days.append(day_data.values)

        df_all = pd.DataFrame(all_days).T
        df_all.index = day_data.index  # Ensure the index is consistent
        return df_all



    def crop_time(self,t_i,t_f,df):
        mask = (df.index > t_i) & (df.index < t_f)
        df = df.loc[mask]
        return df.groupby(df.index).mean()
    
    # def crop_time(self, t_i, t_f, df):
    #     full_index = pd.date_range(start=t_i, end=t_f, freq='T')
    #     df = df.reindex(full_index)
    #     df = df[~df.index.duplicated(keep='first')]
    #     return df
    

    def plot_avg_over_days(self):
        selected_vars = [self.var_listbox.get(i) for i in self.var_listbox.curselection()]
        resample_interval = self.resample_entry.get()
        start_date_str = self.start_date_combobox.get()
        end_date_str = self.end_date_combobox.get()
        start_hour = self.start_hour_scale.get()
        end_hour = self.end_hour_scale.get()

        if not start_date_str or not end_date_str:
            self.log("Start date or end date not selected.")
            return

        if start_hour >= end_hour:
            self.log("Start hour must be less than end hour.")
            return

        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
        
        fig, ax = plt.subplots(figsize=(10, 6))

        for exp_path_entry, color_combobox, include_var, alias_entry in self.experiment_details:
            if not include_var.get():
                continue  # Skip if not included

            exp_path = exp_path_entry.get()
            color = color_combobox.get()
            alias = alias_entry.get()

            if not exp_path or not color or not alias:
                continue
                
            try:
                analyzed_data_path = os.path.join(exp_path, "analyzed_data.pkl")
                if os.path.exists(analyzed_data_path):
                    with open(analyzed_data_path, 'rb') as f:
                        analyzed_data = pickle.load(f)

                    full_population_data = analyzed_data['population_data']
                    full_individual_data = analyzed_data['individual_data']
                    summary_data = analyzed_data['summary_data']

                    for i, var in enumerate(selected_vars):
                        if var in full_population_data.columns:
                            var_data = full_population_data[var]
                        elif var in full_individual_data.columns:
                            var_data = full_individual_data[var]
                            # Apply threshold filtering for "flight_duration" and "average_speed"
                            if var in ["flight_duration", "average_speed"]:
                                threshold_value = float(self.threshold_entry.get() if self.threshold_entry.get() else 0)
                                var_data = var_data[var_data > threshold_value]
                        else:
                            self.log(f"Variable '{var}' not found in experiment data.")
                            continue

                        var_data = self.crop_time(start_date, end_date, var_data)

                        if resample_interval:
                            var_data = var_data.resample(resample_interval).mean()

                        # Filter data for the selected time interval
                        var_data = var_data.between_time(f'{start_hour}:00', f'{end_hour}:00')

                        daily_means = var_data.resample('D').mean()  # Resample by day, taking the mean, ignoring NaNs
                        daily_means.plot(ax=ax, label=f"{alias} - {var}", color=color, marker='.', linestyle='--')

            except Exception as e:
                self.log(f"Error loading data from {exp_path}: {e}")

        ax.set_title(f"Average of Selected Variables Over Days (from {start_hour}:00 to {end_hour}:00 each day)")
        ax.set_xlabel("Day")
        ax.set_ylabel("Average Value")
        ax.legend()

        plt.show()


    def show_flight_metrics(self):
        # Get selected tracking video
        selected_video = self.pkl_listbox.curselection()
        if not selected_video:
            self.log("No tracking video selected.")
            return
        
        tracking_file = self.pkl_listbox.get(selected_video[0])
        try:
            # Fetch the video name and load the corresponding tracking data
            video_name = self.get_video_name_from_tracking_file(tracking_file)
            # Initialize a single_video_analysis object
            video_analysis = single_video_analysis(self.experiment_manager.experiment, video_name, debug_mode=False)
            video_analysis.mosquito_tracks = self.video_manager.mosquito_tracks
            
            # Perform the analysis to extract flight metrics around resting points
            video_analysis.extract_flight_metrics_around_resting()
            flight_metrics_df = video_analysis.mosquito_tracks.flight_metrics_around_resting
            
            if flight_metrics_df.empty:
                self.log("No flight metrics found.")
                return

            # Prepare the data for plotting
            flight_metrics_df['landing_duration'] = flight_metrics_df['landing'].apply(lambda x: x[0])
            flight_metrics_df['landing_speed'] = flight_metrics_df['landing'].apply(lambda x: x[1])
            flight_metrics_df['takeoff_duration'] = flight_metrics_df['takeoff'].apply(lambda x: x[0])
            flight_metrics_df['takeoff_speed'] = flight_metrics_df['takeoff'].apply(lambda x: x[1])

             # Define colors for each zone
            zone_colors = {
                'sugar': 'blue',
                'hs': 'green',
                'left_ctrl': 'red',
                'right_ctrl': 'orange'
            }

            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot take-off duration as a function of resting time
            for zone, color in zone_colors.items():
                zone_data = flight_metrics_df[flight_metrics_df['zone'] == zone]
                ax.scatter(zone_data['resting_time'], zone_data['takeoff_duration'], color=color, label=zone)

            ax.set_title("Takeoff Duration as a function of Resting Time")
            ax.set_xlabel("Resting Time (s)")
            ax.set_ylabel("Takeoff Duration (s)")
            ax.legend(title="Zone")

            plt.show()

        except Exception as e:
            self.log(f"Error loading tracking data from {tracking_file}: {e}")