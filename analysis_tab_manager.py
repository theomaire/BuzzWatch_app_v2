from common_imports import *
from logger import MultiLogger

from threading import Thread


class AnalysisTabManager:
    def __init__(self, root, ui_manager, state_manager):
        self.root = root
        self.ui_manager = ui_manager
        self.state_manager = state_manager
        self.experiment_manager = ui_manager.experiment_manager
        self.log = ui_manager.log
        self.video_manager = ui_manager.video_manager
        self.video_scrollbar = None

    def init_analysis_tab(self, tab):
        self.tab = tab
        self.sub_notebook = ttk.Notebook(tab)
        self.sub_notebook.pack(fill='both', expand=True)

        self.create_extract_images_tab()
        self.create_get_background_tab()
        self.create_draw_borders_tab()
        self.create_single_video_tracking_tab()

    def create_extract_images_tab(self):
        extract_images_tab = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(extract_images_tab, text='Extract Images from Video')
        
        force_rerun_var = tk.IntVar()
        options_frame = tk.Frame(extract_images_tab)
        options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        get_images_button = tk.Button(options_frame, text="Run", command=lambda: self.experiment_manager.get_images_from_video(force_rerun_var.get()))
        get_images_button.pack(side=tk.LEFT, padx=5)
        
        update_images_button = tk.Button(options_frame, text="Update image list", command=self.update_image_listbox)
        update_images_button.pack(side=tk.LEFT, padx=5)

        force_rerun_check = tk.Checkbutton(options_frame, text="Force Re-run", variable=force_rerun_var)
        force_rerun_check.pack(side=tk.LEFT, padx=5)
        self.state_manager.set('force_rerun_var', force_rerun_var)

        image_main_frame = tk.Frame(extract_images_tab)
        image_main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        image_listbox_frame = tk.Frame(image_main_frame, width=300)
        image_listbox_frame.pack(side=tk.LEFT, fill=tk.Y)

        image_scrollbar = tk.Scrollbar(image_listbox_frame)
        image_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.ui_manager.image_manager.image_listbox = tk.Listbox(image_listbox_frame, yscrollcommand=image_scrollbar.set, selectmode=tk.SINGLE, width=50)
        self.ui_manager.image_manager.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        image_scrollbar.config(command=self.ui_manager.image_manager.image_listbox.yview)

        self.ui_manager.image_manager.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)
        
        image_display_frame = tk.Frame(image_main_frame)
        image_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.ui_manager.image_manager.image_label = tk.Label(image_display_frame)
        self.ui_manager.image_manager.image_label.pack(fill=tk.BOTH, expand=True)

    def create_get_background_tab(self):
        get_background_tab = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(get_background_tab, text='Get Background from Images')
        
        force_rerun_var = tk.IntVar()
        options_frame = tk.Frame(get_background_tab)
        options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        get_background_button = tk.Button(options_frame, text="Run", command=lambda: self.experiment_manager.get_background_from_images(force_rerun_var.get()))
        get_background_button.pack(side=tk.LEFT, padx=5)

        update_background_button = tk.Button(options_frame, text="Update bck image list", command=self.update_median_image_listbox)
        update_background_button.pack(side=tk.LEFT, padx=5)

        force_rerun_check = tk.Checkbutton(options_frame, text="Force Re-run", variable=force_rerun_var)
        force_rerun_check.pack(side=tk.LEFT, padx=5)
        self.state_manager.set('force_rerun_var', force_rerun_var)

        image_main_frame = tk.Frame(get_background_tab)
        image_main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        image_listbox_frame = tk.Frame(image_main_frame, width=300)
        image_listbox_frame.pack(side=tk.LEFT, fill=tk.Y)

        image_scrollbar = tk.Scrollbar(image_listbox_frame)
        image_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.ui_manager.image_manager.median_image_listbox = tk.Listbox(image_listbox_frame, yscrollcommand=image_scrollbar.set, selectmode=tk.SINGLE, width=50)
        self.ui_manager.image_manager.median_image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        image_scrollbar.config(command=self.ui_manager.image_manager.median_image_listbox.yview)

        self.ui_manager.image_manager.median_image_listbox.bind('<<ListboxSelect>>', self.on_median_image_select)

        image_display_frame = tk.Frame(image_main_frame)
        image_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.ui_manager.image_manager.median_image_label = tk.Label(image_display_frame)
        self.ui_manager.image_manager.median_image_label.pack(fill=tk.BOTH, expand=True)

    def create_draw_borders_tab(self):
        draw_borders_tab = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(draw_borders_tab, text='Draw Borders')
        
        buttons_frame = tk.Frame(draw_borders_tab)
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

        image_display_frame = tk.Frame(draw_borders_tab)
        image_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.borders_image_label = tk.Label(image_display_frame)
        self.borders_image_label.pack(fill=tk.BOTH, expand=True)

    def create_single_video_tracking_tab(self):

        tab= ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(tab, text='Single Video Tracking')

        tab.grid_rowconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=0)
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_columnconfigure(2, weight=4)

        # Left column for buttons
        buttons_frame = tk.Frame(tab)
        buttons_frame.grid(row=0, column=0, rowspan=2, sticky="ns", padx=5, pady=5)

        analyze_video_button = tk.Button(buttons_frame, text="Analyze Video", command=self.analyze_selected_video)
        analyze_video_button.pack(anchor=tk.W, pady=2)

        update_tracking_video_button = tk.Button(buttons_frame, text="Update Tracking results", command=self.update_pkl_listbox)
        update_tracking_video_button.pack(anchor=tk.W, pady=2)

        load_tracking_video_button = tk.Button(buttons_frame, text="Load Tracking Video", command=self.load_tracking_video)
        load_tracking_video_button.pack(anchor=tk.W, pady=2)

        inspect_time_series_button = tk.Button(buttons_frame, text="Show time series", command=self.inspect_time_series)
        inspect_time_series_button.pack(anchor=tk.W, pady=2)

        inspect_hist_button = tk.Button(buttons_frame, text="Show histograms", command=self.inspect_histogram)
        inspect_hist_button.pack(anchor=tk.W, pady=2)

        inspect_traj_button = tk.Button(buttons_frame, text="Show flight traj", command=self.inspect_trajectories)
        inspect_traj_button.pack(anchor=tk.W, pady=2)

        extract_metrics_button = tk.Button(buttons_frame, text="Extract Flight Metrics", command=self.show_flight_metrics)
        extract_metrics_button.pack(anchor=tk.W, pady=2)

        # Middle column for list boxes
        listbox_frame = tk.Frame(tab)
        listbox_frame.grid(row=0, column=1, rowspan=2, sticky="ns", padx=5, pady=5)

        pkl_listbox_label = tk.Label(listbox_frame, text="Videos files available for analysis")
        pkl_listbox_label.pack(anchor=tk.W)

        self.video_listbox_svtracking = tk.Listbox(listbox_frame, selectmode=tk.SINGLE, width=20)
        self.video_listbox_svtracking.pack(side=tk.TOP, fill=tk.BOTH, expand=False, pady=5)

        pkl_listbox_label = tk.Label(listbox_frame, text="Final Tracking Files:")
        pkl_listbox_label.pack(anchor=tk.W, pady=(10, 0))

        self.pkl_listbox = tk.Listbox(listbox_frame, selectmode=tk.SINGLE, width=20)
        self.pkl_listbox.pack(fill=tk.BOTH, expand=False, pady=5)

        # Right column for video display
        video_display_frame = tk.Frame(tab)
        video_display_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        video_display_frame.grid_rowconfigure(0, weight=1)
        video_display_frame.grid_columnconfigure(0, weight=1)

        self.tracking_image_label = tk.Label(video_display_frame)
        self.tracking_image_label.grid(row=0, column=0, sticky="nswe")
        self.tracking_image_label.bind('<Configure>', self.on_resize_video)

        self.video_start_time_label = tk.Label(video_display_frame, text="")
        self.video_start_time_label.grid(row=1, column=0, sticky="ew")

        controls_frame = tk.Frame(tab)
        controls_frame.grid(row=1, column=2, sticky="ew", padx=5, pady=5)

        play_button = tk.Button(controls_frame, text="Play", command=self.video_manager.play_video)
        play_button.pack(side=tk.LEFT, padx=5)

        pause_button = tk.Button(controls_frame, text="Pause", command=self.video_manager.pause_video)
        pause_button.pack(side=tk.LEFT, padx=5)

        left_button = tk.Button(controls_frame, text="<<", command=self.video_manager.previous_frame)
        left_button.pack(side=tk.LEFT, padx=5)

        right_button = tk.Button(controls_frame, text=">>", command=self.video_manager.next_frame)
        right_button.pack(side=tk.RIGHT, padx=5)

        self.tracking_scrollbar = tk.Scale(controls_frame, from_=0, to=self.video_manager.total_frames-1, orient=tk.HORIZONTAL, command=self.on_scroll)
        self.tracking_scrollbar.pack(fill=tk.X, expand=True, padx=5, pady=5)
        self.video_manager.set_scrollbar(self.tracking_scrollbar)


    def draw_borders_and_update(self, draw_func):
        try:
            draw_func()
            self.display_borders_image()
        except Exception as e:
            self.log(f"Error drawing borders: {e}")

    def display_borders_image(self):
        background_with_borders_path = os.path.join(self.experiment_manager.experiment.folder_analysis, "background_with_borders.png")
        try:
            img = Image.open(background_with_borders_path)
            img = img.resize((int(img.width * self.ui_manager.image_manager.resize_factor), int(img.height * self.ui_manager.image_manager.resize_factor)))
            imgtk = ImageTk.PhotoImage(image=img)
            self.borders_image_label.imgtk = imgtk
            self.borders_image_label.config(image=imgtk)
        except Exception as e:
            self.log(f"Error displaying background with borders: {str(e)}")

    def on_image_select(self, event):
        if not self.experiment_manager.experiment:
            self.log("No experiment loaded.")
            return

        selected_image = self.ui_manager.image_manager.image_listbox.curselection()
        if not selected_image:
            self.log("No image selected from the list.")
            return

        images_path = os.path.join(self.experiment_manager.experiment.folder_analysis, "individual_images")
        image_file = os.path.join(images_path, self.ui_manager.image_manager.image_listbox.get(selected_image[0]))
        self.log(f"Displaying image: {image_file}")
        self.ui_manager.image_manager.display_image(image_file, self.ui_manager.image_manager.image_label)

    def on_median_image_select(self, event):
        if not self.experiment_manager.experiment:
            self.log("No experiment loaded.")
            return

        selected_image = self.ui_manager.image_manager.median_image_listbox.curselection()
        if not selected_image:
            self.log("No image selected from the list.")
            return

        images_path = os.path.join(self.experiment_manager.experiment.folder_analysis, "images_mortality")
        image_file = os.path.join(images_path, self.ui_manager.image_manager.median_image_listbox.get(selected_image[0]))
        self.log(f"Displaying image: {image_file}")
        self.ui_manager.image_manager.display_image(image_file, self.ui_manager.image_manager.median_image_label)

    def update_image_listbox(self):
        if self.experiment_manager.experiment:
            images_path = os.path.join(self.experiment_manager.experiment.folder_analysis, "individual_images")
            self.ui_manager.image_manager.update_image_list(self.ui_manager.image_manager.image_listbox, images_path)
        else:
            self.log("No exp loaded")

    def update_median_image_listbox(self):
        if self.experiment_manager.experiment:
            images_path = os.path.join(self.experiment_manager.experiment.folder_analysis, "images_mortality")
            self.ui_manager.image_manager.update_median_image_list(self.ui_manager.image_manager.median_image_listbox, images_path)





###################### Function for single_video_tracking_tab

    def analyze_selected_video(self):
        selected_video = self.video_listbox_svtracking.curselection()
        if not selected_video:
            self.log("No video selected for analysis.")
            return
        video_name = self.video_listbox_svtracking.get(selected_video[0])
        self.log("Selected_"+video_name)

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


    def update_video_listbox_svtracking(self):
        if not self.experiment_manager.folder_videos:
            self.log("Folder path for videos is not set.")
            return
        video_files = [f for f in os.listdir(os.path.join(self.experiment_manager.folder_analysis, "images_mortality")) if f.endswith('.png') and f.startswith('Cage')]
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
    def update_scrollbar(self):
        total_frames = self.video_manager.get_total_frames()
        self.tracking_scrollbar.configure(to=total_frames - 1)
        self.video_manager.show_frame(0)

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
                self.update_scrollbar()

             # Get and display the video's start datetime
            start_time = self.video_manager.get_datetime_from_file_name(video_name)
            self.set_video_start_time(start_time)

    def inspect_time_series(self):
        # Create a new top-level window
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)  # Adjust the space between plots

        # Flatten the 2D array of axes for easier access
        axs = axs.flatten()

        # Population Variables Plots
        data = self.video_manager.mosquito_tracks.population_variables  # Retrieve the population data

        time_intervals = np.arange(len(data['numb_mosquitos_flying']))
        #print(len(data['numb_mosquitos_flying']))
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


    def on_scroll(self, value):
        frame_index = int(value)
        self.video_manager.show_frame(frame_index)

    def on_resize_video(self, event):
        if self.tracking_scrollbar is not None:
            frame_index = int(self.tracking_scrollbar.get())
            self.video_manager.show_frame(frame_index)
        else:
            self.log("Scrollbar not initialized.")

    def set_video_start_time(self, start_time):
        """Display the start date and time of the video."""
        if start_time:
            formatted_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
            self.video_start_time_label.config(text=f"Video Start Time: {formatted_time}")
        else:
            self.video_start_time_label.config(text="")