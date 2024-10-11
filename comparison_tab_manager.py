

from common_imports import *

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from plot_manager import PlotManager
from statistical_analysis_manager import StatisticalAnalysisManager
from statsmodels.stats.multitest import multipletests
from matplotlib.markers import MarkerStyle
from matplotlib import colors

# At the beginning of your script or top of your class
AVAILABLE_COLORS = list(colors.TABLEAU_COLORS.keys())
AVAILABLE_MARKERS = list(MarkerStyle.markers.keys())


class ComparisonTabManager:
    def __init__(self, root, ui_manager, state_manager):
        self.root = root
        self.ui_manager = ui_manager
        self.state_manager = state_manager
        self.experiment_manager = ui_manager.experiment_manager
        self.log = ui_manager.log
        self.grouped_experiments = {}

        # Initialize the lists to track experiment and group widgets
        self.exp_path_entries = []
        self.alias_entries = []
        self.group_comboboxes = []
        self.experiments_frame = None
        self.variable_vars = []  # Initialize here

        # Initialize lists to hold widget entries
        self.group_alias_entries = []
        self.group_color_entries = []
        self.group_marker_entries = []
        self.category_alias_entries = []
        self.category_color_entries = []
        self.category_marker_entries = []
        self.plot_manager = PlotManager(self.log,
                                        group_colors=self.collect_group_colors(),
                                        group_markers=self.collect_group_markers(),
                                        category_colors=self.collect_category_colors(),
                                        category_markers=self.collect_category_markers())

        self.stat_analysis_manager = StatisticalAnalysisManager(self.log)  # Initialize StatisticalAnalysisManager
        self.exp_start_date_comboboxes = []
        self.exp_end_date_comboboxes = []

        self.variable_name_mapping = {
        'numb_mosquitos_flying': 'numb_mosquitos_flying',
        'numb_mosquitos_sugar': 'numb_mosquitos_sugar',
        'numb_mosquitos_hs': 'numb_mosquito_control',
        'numb_mosquitos_left_ctrl': 'numb_mosquitos_left_ctrl',
        'numb_mosquitos_right_ctrl': 'numb_mosquitos_right_ctrl',
        'flight_duration': 'flight_duration',
        'average_speed': 'average_speed'
    }

        

        # Initialize group and category assignments
        self.exp_group_assignments = {}
        self.exp_category_assignments = {}  # New addition

    def collect_group_colors(self):
        return {entry.get(): color.get() for entry, color in zip(self.group_alias_entries, self.group_color_entries)}

    def collect_group_markers(self):
        return {entry.get(): marker.get() for entry, marker in zip(self.group_alias_entries, self.group_marker_entries)}

    def collect_category_colors(self):
        return {entry.get(): color.get() for entry, color in zip(self.category_alias_entries, self.category_color_entries)}

    def collect_category_markers(self):
        return {entry.get(): marker.get() for entry, marker in zip(self.category_alias_entries, self.category_marker_entries)}
    def init_comparison_tab(self, tab):
        self.tab = tab
        self.notebook = ttk.Notebook(self.tab)
        self.notebook.pack(fill='both', expand=True)

        self.create_experiment_grouping_tab()
        self.create_flight_activity_analysis_tab()
        self.create_flight_statistics_analysis_tab()
        # self.create_sugar_feeding_analysis_tab()
        # self.create_flight_stastistics_analysis_tab()

    def save_settings(self):
        settings = {
            "root_folder": self.root_folder_entry.get(),
            "num_experiments": self.num_experiments_entry.get(),
            "experiments": [{
                "path": exp_path_entry.get(),
                "alias": alias_entry.get(),
                "group": self.exp_group_assignments[index].get(),
                "category": self.exp_category_assignments[index].get(),
                "start_date": self.exp_start_date_comboboxes[index].get(),
                "end_date": self.exp_end_date_comboboxes[index].get()
            } for index, (exp_path_entry, alias_entry) in enumerate(zip(self.exp_path_entries, self.alias_entries))],
            "num_groups": self.num_groups_entry.get(),
            "groups": [{
                "alias": group_alias_entry.get(),
                "color": self.group_color_entries[i].get(),
                "marker": self.group_marker_entries[i].get()
            } for i, group_alias_entry in enumerate(self.group_alias_entries)],
            "num_categories": self.num_categories_entry.get(),
            "categories": [{
                "alias": category_alias_entry.get(),
                "color": self.category_color_entries[i].get(),
                "marker": self.category_marker_entries[i].get()
            } for i, category_alias_entry in enumerate(self.category_alias_entries)]
        }
        save_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(settings, f)
            self.log(f"Settings saved to {save_path}")

            
            # Create a folder named after the JSON file (without the extension)
            analysis_folder_name = os.path.splitext(os.path.basename(save_path))[0]
            self.analysis_folder = os.path.join(os.path.dirname(save_path), analysis_folder_name)
            os.makedirs(self.analysis_folder, exist_ok=True)

            self.common_save_dir = os.path.join(self.analysis_folder, 'analysis_results')
            os.makedirs(self.common_save_dir, exist_ok=True)
            self.common_plots_dir = os.path.join(self.analysis_folder, 'plots')
            os.makedirs(self.common_plots_dir, exist_ok=True)

            self.plot_manager.save_dir = self.common_plots_dir

    def load_settings(self):
        load_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if load_path:
            with open(load_path, 'r') as f:
                settings = json.load(f)

            analysis_folder_name = os.path.splitext(os.path.basename(load_path))[0]
            self.analysis_folder = os.path.join(os.path.dirname(load_path), analysis_folder_name)
            os.makedirs(self.analysis_folder, exist_ok=True)


            self.root_folder_entry.delete(0, tk.END)
            self.root_folder_entry.insert(0, settings.get("root_folder", ""))

            self.num_experiments_entry.delete(0, tk.END)
            self.num_experiments_entry.insert(0, settings["num_experiments"])
            self.add_experiment_inputs()

            for i, exp_settings in enumerate(settings["experiments"]):
                exp_path_entry = self.exp_path_entries[i]
                alias_entry = self.alias_entries[i]
                start_combobox = self.exp_start_date_comboboxes[i]
                end_combobox = self.exp_end_date_comboboxes[i]

                exp_path_entry.delete(0, tk.END)
                exp_path_entry.insert(0, exp_settings["path"])

                alias_entry.delete(0, tk.END)
                alias_entry.insert(0, exp_settings["alias"])

                self.exp_group_assignments[i].set(exp_settings["group"])
                self.exp_category_assignments[i].set(exp_settings.get("category", ""))
                self.root.after(100, lambda i=i, exp_settings=exp_settings: self.update_date_comboboxes_tab(i, exp_settings))

            self.num_groups_entry.delete(0, tk.END)
            self.num_groups_entry.insert(0, settings["num_groups"])
            self.add_group_inputs()

            for i, group in enumerate(settings["groups"]):
                group_alias_entry = self.group_alias_entries[i]
                group_alias_entry.delete(0, tk.END)
                group_alias_entry.insert(0, group["alias"])
                self.group_color_entries[i].set(group.get("color", ""))
                self.group_marker_entries[i].set(group.get("marker", ""))

            self.num_categories_entry.delete(0, tk.END)
            self.num_categories_entry.insert(0, settings.get("num_categories", 0))
            self.add_category_inputs()

            for i, category in enumerate(settings.get("categories", [])):
                category_alias_entry = self.category_alias_entries[i]
                category_alias_entry.delete(0, tk.END)
                category_alias_entry.insert(0, category["alias"])
                self.category_color_entries[i].set(category.get("color", ""))
                self.category_marker_entries[i].set(category.get("marker", ""))

            self.update_group_comboboxes()
            self.populate_selection_checkbuttons()
            self.log(f"Settings loaded from {load_path}")

    def update_date_comboboxes_tab(self, index, exp_settings):
        self.exp_start_date_comboboxes[index].set(exp_settings.get("start_date", ""))
        self.exp_end_date_comboboxes[index].set(exp_settings.get("end_date", ""))


#################  Experiment group tab ###########
    def create_experiment_grouping_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Experiment Grouping")

        self.configure_experiment_grouping_sub_tab(tab)

    def select_root_folder(self):
        folder_selected = filedialog.askdirectory(title="Select root folder for analysis")
        if folder_selected:
            self.root_folder_entry.delete(0, tk.END)
            self.root_folder_entry.insert(0, folder_selected)
    def on_frame_configure(self, event):
        """
        Reset the scroll region to encompass the inner frame
        """
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    # Function to update the canvas width when the canvas size changes
    def on_canvas_configure(self, event):
        """
        Configure canvas scroll region to update with canvas size changes
        """
        canvas_width = event.width
        canvas_height = event.height
        #self.canvas.itemconfig(self.exp_details_frame_id, width=canvas_width, height=canvas_height)

    def configure_experiment_grouping_sub_tab(self, tab):
        # Set grid weights to control dynamic resizing
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_rowconfigure(2, weight=1)
        tab.grid_rowconfigure(3, weight=1)
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=1)

        # Root folder frame
        root_folder_frame = tk.Frame(tab)
        root_folder_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

        root_folder_frame.grid_columnconfigure(0, weight=1)

        load_button = tk.Button(root_folder_frame, text="Load Settings", command=self.load_settings)
        load_button.pack(side=tk.LEFT, padx=5)

        save_button = tk.Button(root_folder_frame, text="Save Settings", command=self.save_settings)
        save_button.pack(side=tk.LEFT, padx=5)

        root_folder_label = tk.Label(root_folder_frame, text="Root Folder for Analysis:")
        root_folder_label.pack(side=tk.LEFT, padx=5)

        self.root_folder_entry = tk.Entry(root_folder_frame, width=10)
        self.root_folder_entry.pack(side=tk.LEFT, padx=5)

        select_root_folder_button = tk.Button(root_folder_frame, text="Select Folder", command=self.select_root_folder)
        select_root_folder_button.pack(side=tk.LEFT, padx=5)

        num_experiments_label = tk.Label(root_folder_frame, text="Number of Experiments:")
        num_experiments_label.pack(side=tk.LEFT, padx=5)

        self.num_experiments_entry = tk.Entry(root_folder_frame, width=5)
        self.num_experiments_entry.pack(side=tk.LEFT, padx=5)

        add_experiments_button = tk.Button(root_folder_frame, text="Add Experiments", command=self.add_experiment_inputs)
        add_experiments_button.pack(side=tk.LEFT, padx=5)

        canvas_frame = tk.Frame(tab)
        canvas_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        vertical_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        vertical_scrollbar.grid(row=0, column=1, sticky="ns")
        
        horizontal_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        horizontal_scrollbar.grid(row=1, column=0, sticky="ew")

        self.canvas.configure(yscrollcommand=vertical_scrollbar.set, xscrollcommand=horizontal_scrollbar.set)

        self.exp_details_frame = tk.Frame(self.canvas)
        self.exp_details_frame_id = self.canvas.create_window((0, 0), window=self.exp_details_frame, anchor='nw')

        self.exp_details_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        # Group management frame
        group_frame = tk.LabelFrame(tab, text="Manage Groups")
        group_frame.grid(row=2, column=0, padx=10, pady=2, sticky="nsew")

        num_groups_label = tk.Label(group_frame, text="Number of Groups:")
        num_groups_label.pack(side=tk.LEFT, padx=5, pady=2)

        self.num_groups_entry = tk.Entry(group_frame, width=5)
        self.num_groups_entry.pack(side=tk.LEFT, padx=5)

        add_groups_button = tk.Button(group_frame, text="Add Groups", command=self.add_group_inputs)
        add_groups_button.pack(side=tk.LEFT, padx=5, pady=2)

        # Group categories frame
        category_frame = tk.LabelFrame(tab, text="Manage Group Categories")
        category_frame.grid(row=2, column=1, padx=10, pady=2, sticky="nsew")

        num_categories_label = tk.Label(category_frame, text="Number of Group Categories:")
        num_categories_label.pack(side=tk.LEFT, padx=5, pady=2)

        self.num_categories_entry = tk.Entry(category_frame, width=5)
        self.num_categories_entry.pack(side=tk.LEFT, padx=5)

        add_categories_button = tk.Button(category_frame, text="Add Group Categories", command=self.add_category_inputs)
        add_categories_button.pack(side=tk.LEFT, padx=5, pady=2)

        # Group frame for displaying group and category inputs together
        self.groups_frame = tk.Frame(tab)
        self.groups_frame.grid(row=3, column=0, padx=10, pady=5, sticky="nsew")

        # Group frame for displaying group and category inputs together
        self.category_frame = tk.Frame(tab)
        self.category_frame.grid(row=3, column=1, padx=10, pady=5, sticky="nsew")

        self.exp_group_assignments = {}
        self.group_alias_entries = []
        self.category_alias_entries = []

        # Finalize Button
        self.finalize_button = tk.Button(root_folder_frame, text="Finalize data",bg="red", command=self.finalize_groups)
        self.finalize_button.pack(side=tk.LEFT, padx=5)

    def finalize_groups(self):
        if self.analysis_folder:
            self.common_save_dir = os.path.join(self.analysis_folder, 'analysis_results')
            os.makedirs(self.common_save_dir, exist_ok=True)
            self.common_plots_dir = os.path.join(self.analysis_folder, 'plots')
            os.makedirs(self.common_plots_dir, exist_ok=True)
            self.plot_manager.save_dir = self.common_plots_dir
        else:
            self.log("Please save settings first.")
            return

        self.grouped_experiments = {}
        unified_index = None

        num_experiments = int(self.num_experiments_entry.get())
        all_min_dates = []
        all_durations = []

        root_folder = self.root_folder_entry.get()

        for i in range(num_experiments):
            exp_path_entry = self.exp_path_entries[i]
            start_combobox = self.exp_start_date_comboboxes[i]
            end_combobox = self.exp_end_date_comboboxes[i]
            exp_path = os.path.join(root_folder, exp_path_entry.get())
            
            if not exp_path:
                continue

            try:
                analyzed_data_path = os.path.join(exp_path, "analyzed_data.pkl")
                if os.path.exists(analyzed_data_path):
                    with open(analyzed_data_path, 'rb') as f:
                        analyzed_data = pickle.load(f)

                    full_population_data = analyzed_data.get('population_data', None)
                    #print(full_population_data)
                    if full_population_data is not None:
                        all_min_dates.append(full_population_data.index.min())
                        start_date_a = pd.to_datetime(start_combobox.get())
                        end_date_a = pd.to_datetime(end_combobox.get())
                        duration = (end_date_a - start_date_a).total_seconds() / 60  # Duration in minutes
                        all_durations.append(duration)
                        
                else:
                    self.log(f"No analyzed_data.pkl found for {exp_path}.")
            except Exception as e:
                self.log(f"Error loading data from {exp_path}: {e}")

        if all_min_dates and all_durations:
            common_start_date = min(all_min_dates).normalize()  # Set earliest start date to 00:00
            max_duration_days = max(all_durations) // 1440  # Duration in days
            global_end_date = common_start_date + pd.Timedelta(days=max_duration_days) + pd.Timedelta(hours=23, minutes=59)
            unified_index = pd.date_range(start=common_start_date, end=global_end_date, freq='T')


        if unified_index is None:
            self.log("No valid experiment data found to determine a unified index.")
            return

        for i in range(num_experiments):
            exp_path_entry = self.exp_path_entries[i]
            alias_entry = self.alias_entries[i]
            group_combobox = self.exp_group_assignments[i]
            category_combobox = self.exp_category_assignments[i]
            start_combobox = self.exp_start_date_comboboxes[i]
            end_combobox = self.exp_end_date_comboboxes[i]

            exp_path = os.path.join(root_folder, exp_path_entry.get())
            alias = alias_entry.get()
            group_name = group_combobox.get()
            category_name = category_combobox.get()
            start_date = pd.to_datetime(start_combobox.get())
            end_date = pd.to_datetime(end_combobox.get())

            if category_name not in self.grouped_experiments:
                self.grouped_experiments[category_name] = {}

            if group_name not in self.grouped_experiments[category_name]:
                self.grouped_experiments[category_name][group_name] = []

            if not exp_path:
                continue

            try:
                analyzed_data_path = os.path.join(exp_path, "analyzed_data.pkl")
                if os.path.exists(analyzed_data_path):
                    with open(analyzed_data_path, 'rb') as f:
                        analyzed_data = pickle.load(f)

                    full_population_data = analyzed_data.get('population_data', None)
                    death_count_file = next((f for f in os.listdir(exp_path) if 'death_count' in f and f.endswith('.xlsx')), None)
                    
                    if not death_count_file:
                        self.log(f"No death count file found for {exp_path}.")
                        continue

                    death_count_path = os.path.join(exp_path, death_count_file)
                    df_dead = pd.read_excel(death_count_path)
                    df_dead['Date'] = pd.to_datetime(df_dead['Date'])
                    df_dead.set_index('Date', inplace=True)
                    df_dead = df_dead.resample('T').interpolate()



                    dead_mosquito_column = df_dead.columns[1]

                    if full_population_data is not None:
                        full_population_data = full_population_data[(full_population_data.index >= start_date) & (full_population_data.index <= end_date)]
                        normalized_start_date = full_population_data.index.min().normalize()  # Account for cropped data
                        shift_offset = common_start_date - normalized_start_date
                        full_population_data.index = full_population_data.index + shift_offset
                        full_population_data = full_population_data[~full_population_data.index.duplicated(keep='first')]
                        full_population_data = full_population_data.reindex(unified_index).interpolate(method='time')


                        initial_mosquito_count = 40  # adjust based on your setup
                        alive_mosquitos = initial_mosquito_count - df_dead[dead_mosquito_column].reindex(full_population_data.index, method='nearest')
                        for column in full_population_data.columns:
                            full_population_data[column] = full_population_data[column].div(alive_mosquitos).fillna(0)

                    full_individual_data = analyzed_data.get('individual_data', None)
                    if full_individual_data is not None:
                        full_individual_data = full_individual_data[(full_individual_data.index >= start_date) & (full_individual_data.index <= end_date)]
                        normalized_start_date = full_individual_data.index.min().normalize()  # Account for cropped data
                        shift_offset = common_start_date - normalized_start_date
                        full_individual_data.index = full_individual_data.index + shift_offset
                        full_individual_data = full_individual_data[~full_individual_data.index.duplicated(keep='first')]
                        full_individual_data = full_individual_data.reindex(unified_index).interpolate(method='time')

                    summary_data = analyzed_data.get('summary_data', None)

                    self.state_manager.set(exp_path, {
                        'population_data': full_population_data,
                        'individual_data': full_individual_data,
                        'summary_data': summary_data,
                        'alias': alias
                    })

                    self.grouped_experiments[category_name][group_name].append({
                        'path': exp_path,
                        'alias': alias,
                        'population_data': full_population_data,
                        'individual_data': full_individual_data,
                        'summary_data': summary_data,
                        'min_date': common_start_date,
                        'max_date': global_end_date
                    })
                    self.log(f"Loaded data for {exp_path}: {common_start_date} to {global_end_date}")
                else:
                    self.log(f"No analyzed_data.pkl found for {exp_path}.")
                self.finalize_button.config(bg="green")
            except Exception as e:
                self.log(f"Error loading data from {exp_path}: {e}")

        plot_data_path = os.path.join(self.common_save_dir, 'plot_data.pkl')
        with open(plot_data_path, 'wb') as f:
            pickle.dump(self.grouped_experiments, f)
        self.log(f"Plot data structure saved to {plot_data_path}")

            # Update PlotManager with latest colors and markers
        self.plot_manager.group_colors = self.collect_group_colors()
        self.plot_manager.group_markers = self.collect_group_markers()
        self.plot_manager.category_colors = self.collect_category_colors()
        self.plot_manager.category_markers = self.collect_category_markers()
        self.update_date_comboboxes()
        self.populate_selection_checkbuttons()


# Example usage
# finalize_groups() should be called within its proper context




    def validate_group(self, group_alias_entry):
        group_alias = group_alias_entry.get()
        if group_alias:
            self.log(f"Group '{group_alias}' validated.")
            self.update_group_comboboxes()

    def validate_category(self, category_alias_entry):
        category_alias = category_alias_entry.get()
        if category_alias:
            self.log(f"Category '{category_alias}' validated.")
            self.update_group_comboboxes()

    def assign_experiment_to_group_and_category(self, experiment_index, selected_group, selected_category):
        # Update the experiment and group relationships including categories
        if selected_category not in self.grouped_experiments:
            self.grouped_experiments[selected_category] = {}

        if selected_group not in self.grouped_experiments[selected_category]:
            self.grouped_experiments[selected_category][selected_group] = []

        self.grouped_experiments[selected_category][selected_group].append(experiment_index)


#################################################
    def update_date_comboboxes(self):
        all_dates = []
        for category in self.grouped_experiments.values():
            for group in category.values():
                for exp_data in group:
                    population_data = exp_data['population_data']
                    if population_data is not None and not population_data.empty:
                        all_dates.append(population_data.index.min())
                        all_dates.append(population_data.index.max())

        if all_dates:
            start_date = min(all_dates)
            end_date = max(all_dates)
            date_range = pd.date_range(start_date, end_date)
            date_strings = [date.strftime('%Y-%m-%d') for date in date_range]

            self.start_date_combobox['values'] = date_strings
            self.end_date_combobox['values'] = date_strings

            if date_strings:
                self.start_date_combobox.set(date_strings[0])
                self.end_date_combobox.set(date_strings[-1])
    def add_experiment_inputs(self):
        num_experiments = int(self.num_experiments_entry.get())

        # Clear out previous entries to avoid duplication
        self.exp_path_entries.clear()
        self.alias_entries.clear()
        self.exp_group_assignments.clear()
        self.exp_category_assignments.clear()
        self.exp_start_date_comboboxes.clear()
        self.exp_end_date_comboboxes.clear()

        for widget in self.exp_details_frame.winfo_children():
            widget.destroy()

        for i in range(num_experiments):
            frame = tk.Frame(self.exp_details_frame)
            frame.grid(row=i, column=0, sticky="nsew", pady=5)

            exp_label = tk.Label(frame, text=f"Exp {i + 1}:")
            exp_label.grid(row=0, column=0, padx=5)

            exp_path_entry = tk.Entry(frame, width=5)
            exp_path_entry.grid(row=0, column=1, padx=5)
            self.exp_path_entries.append(exp_path_entry)

            select_folder_button = tk.Button(frame, text="Select Folder", command=lambda e=exp_path_entry: self.select_experiment_folder(e))
            select_folder_button.grid(row=0, column=2, padx=5)

            alias_label = tk.Label(frame, text="Alias:")
            alias_label.grid(row=0, column=3, padx=5)

            alias_entry = tk.Entry(frame, width=15)
            alias_entry.grid(row=0, column=4, padx=5)
            self.alias_entries.append(alias_entry)

            start_date_label = tk.Label(frame, text="Start:")
            start_date_label.grid(row=0, column=5, padx=5)

            start_date_combobox = ttk.Combobox(frame, width=10)
            start_date_combobox.grid(row=0, column=6, padx=5)
            self.exp_start_date_comboboxes.append(start_date_combobox)

            end_date_label = tk.Label(frame, text="End:")
            end_date_label.grid(row=0, column=7, padx=5)

            end_date_combobox = ttk.Combobox(frame, width=10)
            end_date_combobox.grid(row=0, column=8, padx=5)
            self.exp_end_date_comboboxes.append(end_date_combobox)

            load_button = tk.Button(frame, text="Load", command=lambda e=exp_path_entry, s=start_date_combobox, ed=end_date_combobox: self.load_experiment_data(e, s, ed))
            load_button.grid(row=0, column=9, padx=5)

            group_label = tk.Label(frame, text="Group:")
            group_label.grid(row=0, column=10, padx=5)

            group_combobox = ttk.Combobox(frame, width=10)
            group_combobox.grid(row=0, column=11, padx=5)
            self.exp_group_assignments[i] = group_combobox

            # Add category selection elements
            category_label = tk.Label(frame, text="Category:")
            category_label.grid(row=0, column=12, padx=5)

            category_combobox = ttk.Combobox(frame, width=10)
            category_combobox.grid(row=0, column=13, padx=5)
            self.exp_category_assignments[i] = category_combobox

        # Update selection listbox whenever experiments are added
        self.populate_selection_checkbuttons()

    def update_group_comboboxes(self):
        groups = [e.get() for e in self.group_alias_entries if e.get().strip()]
        categories = [e.get() for e in self.category_alias_entries if e.get().strip()]

        for index, combobox in self.exp_group_assignments.items():
            combobox['values'] = groups
        
        for index, combobox in self.exp_category_assignments.items():
            combobox['values'] = categories

        # Update the scrollable region in the canvas

    def add_group_inputs(self):
            num_groups = int(self.num_groups_entry.get())
            
            # Clear previous widgets
            for widget in self.groups_frame.winfo_children():
                widget.destroy()

            self.group_alias_entries.clear()
            self.group_color_entries.clear()
            self.group_marker_entries.clear()

            # Creating canvas for the scrolling functionality
            canvas = tk.Canvas(self.groups_frame)
            scrollbar = tk.Scrollbar(self.groups_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = tk.Frame(canvas)

            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            for i in range(num_groups):
                frame = tk.Frame(scrollable_frame)
                frame.grid(row=i, column=0, padx=10, pady=2, sticky="nsew")

                group_label = tk.Label(frame, text=f"Group {i + 1} Alias:")
                group_label.pack(side=tk.LEFT, padx=5)

                group_alias_entry = tk.Entry(frame, width=8)
                group_alias_entry.pack(side=tk.LEFT, padx=5)
                self.group_alias_entries.append(group_alias_entry)

                validate_button = tk.Button(frame, text="Validate", command=lambda e=group_alias_entry: self.validate_group(e))
                validate_button.pack(side=tk.LEFT, padx=5)

                # Add color and marker ComboBox
                color_combobox = ttk.Combobox(frame, values=AVAILABLE_COLORS, width=5)
                color_combobox.pack(side=tk.LEFT, padx=5)
                self.group_color_entries.append(color_combobox)

                marker_combobox = ttk.Combobox(frame, values=AVAILABLE_MARKERS, width=5)
                marker_combobox.pack(side=tk.LEFT, padx=5)
                self.group_marker_entries.append(marker_combobox)

            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            # Update selection listbox whenever groups are added
            self.populate_selection_checkbuttons()

    def add_category_inputs(self):
        num_categories = int(self.num_categories_entry.get())

        # Clear previous widgets
        for widget in self.category_frame.winfo_children():
            widget.destroy()

        self.category_alias_entries.clear()
        self.category_color_entries.clear()
        self.category_marker_entries.clear()

        # Creating canvas for the scrolling functionality
        canvas = tk.Canvas(self.category_frame)
        scrollbar = tk.Scrollbar(self.category_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        for i in range(num_categories):
            frame = tk.Frame(scrollable_frame)
            frame.grid(row=i, column=1, padx=10, pady=2, sticky="nsew")

            category_label = tk.Label(frame, text=f"Category {i + 1} Alias:")
            category_label.pack(side=tk.LEFT, padx=5)

            category_alias_entry = tk.Entry(frame, width=20)
            category_alias_entry.pack(side=tk.LEFT, padx=5)
            self.category_alias_entries.append(category_alias_entry)

            validate_button = tk.Button(frame, text="Validate", command=lambda e=category_alias_entry: self.validate_category(e))
            validate_button.pack(side=tk.LEFT, padx=5)

            # Add color and marker ComboBox
            color_combobox = ttk.Combobox(frame, values=AVAILABLE_COLORS, width=5)
            color_combobox.pack(side=tk.LEFT, padx=5)
            self.category_color_entries.append(color_combobox)

            marker_combobox = ttk.Combobox(frame, values=AVAILABLE_MARKERS, width=5)
            marker_combobox.pack(side=tk.LEFT, padx=5)
            self.category_marker_entries.append(marker_combobox)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")


        # Update selection listbox whenever groups are added
        self.populate_selection_checkbuttons()

    def select_experiment_folder(self, exp_path_entry):
        current_root_folder = self.root_folder_entry.get()
        folder_selected = filedialog.askdirectory(title="Select folder where the experiment is located",initialdir=current_root_folder if current_root_folder else None)
        if folder_selected:
            # Use relative path if root folder is defined
            if current_root_folder and os.path.commonpath([current_root_folder, folder_selected]) == current_root_folder:
                relative_path = os.path.relpath(folder_selected, current_root_folder)
                exp_path_entry.delete(0, tk.END)
                exp_path_entry.insert(0, relative_path)
            else:
                exp_path_entry.delete(0, tk.END)
                exp_path_entry.insert(0, folder_selected)

    def load_experiment_data(self, exp_path_entry, start_combobox, end_combobox):
        exp_path = exp_path_entry.get()
        current_root_folder = self.root_folder_entry.get()
        if not exp_path:
            self.log("No experiment path specified.")
            return
        try:
            analyzed_data_path = os.path.join(current_root_folder, exp_path, "analyzed_data.pkl")
            if os.path.exists(analyzed_data_path):
                with open(analyzed_data_path, 'rb') as f:
                    analyzed_data = pickle.load(f)

                full_population_data = analyzed_data.get('population_data', None)
                full_individual_data = analyzed_data.get('individual_data', None)
                summary_data = analyzed_data.get('summary_data', None)

                min_date = full_population_data.index.min()
                max_date = full_population_data.index.max()
                self.state_manager.set(exp_path, {
                    'population_data': full_population_data,
                    'individual_data': full_individual_data,
                    'summary_data': summary_data,
                    'alias': exp_path_entry.get()
                })

                self.log(f"Pre-loaded data for {exp_path}: {min_date} to {max_date}")

                # Update combobox values and selection
                date_range = pd.date_range(min_date, max_date)
                date_strings = [date.strftime('%Y-%m-%d') for date in date_range]
                start_combobox['values'] = date_strings
                end_combobox['values'] = date_strings
                start_combobox.set(date_strings[0])
                end_combobox.set(date_strings[-1])

                self.update_group_comboboxes()
            else:
                self.log("No analyzed_data.pkl found.")
        except Exception as e:
            self.log(f"Error loading data from {exp_path}: {e}")


    def validate_group(self, group_alias_entry):
        group_alias = group_alias_entry.get()
        if group_alias:
            self.log(f"Group '{group_alias}' validated.")
            self.update_group_comboboxes()

    def validate_category(self, category_alias_entry):
        category_alias = category_alias_entry.get()
        if category_alias:
            self.log(f"Category '{category_alias}' validated.")
            self.update_group_comboboxes()

    def populate_selection_checkbuttons(self):
        # Destroy existing widgets
        for widget in self.experiments_frame.winfo_children():
            widget.destroy()

        self.experiment_vars = []  # Clear the list of experiment Checkbutton variables

        canvas = tk.Canvas(self.experiments_frame)
        scrollbar = ttk.Scrollbar(self.experiments_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Set up the positioning variables
        row, col = 0, 0
        max_items_per_col = 15  # Max items before moving to the next column

        for category_name, groups in self.grouped_experiments.items():
            if row >= max_items_per_col:
                row = 0
                col += 1

            # Category Checkbutton
            var_category = tk.BooleanVar()
            chk_category = tk.Checkbutton(
                scrollable_frame, text=f"Category: {category_name}", 
                variable=var_category, bg="lightblue", anchor='w'
            )
            chk_category.grid(row=row, column=col, sticky='w', padx=5, pady=2)
            self.experiment_vars.append((var_category, category_name))
            row += 1

            for group_name, experiments in groups.items():
                if row >= max_items_per_col:
                    row = 0
                    col += 1

                # Indent Group Checkbutton
                var_group = tk.BooleanVar()
                chk_group = tk.Checkbutton(
                    scrollable_frame, text=f"  Group: {group_name}",  # Indent text
                    variable=var_group, bg="lightgreen", anchor='w'
                )
                chk_group.grid(row=row, column=col, sticky='w', padx=20, pady=1)
                self.experiment_vars.append((var_group, group_name))
                row += 1

                for exp_data in experiments:
                    if row >= max_items_per_col:
                        row = 0
                        col += 1

                    # Further indent Experiment Checkbutton
                    var_experiment = tk.BooleanVar()
                    chk_experiment = tk.Checkbutton(
                        scrollable_frame, text=f"    Experiment: {exp_data['alias']}", 
                        variable=var_experiment, bg="lightyellow", anchor='w'
                    )
                    chk_experiment.grid(row=row, column=col, sticky='w', padx=40, pady=0)
                    self.experiment_vars.append((var_experiment, exp_data))
                    row += 1

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.experiments_frame.grid_rowconfigure(0, weight=1)  # Allow row to expand
        self.experiments_frame.grid_columnconfigure(0, weight=1)  # Allow column to expand

#################  Flight activity tab ##################
    def create_flight_activity_analysis_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Flight Activity Analysis")

        self.configure_flight_activity_analysis_tab(tab)

    def configure_flight_activity_analysis_tab(self, tab):
        # Configure grid weights
        tab.grid_rowconfigure(0, weight=1)  # Increase space for selection frame
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # Selection frame for experiments/groups
        selection_frame = tk.Frame(tab)
        selection_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        selection_frame.grid_rowconfigure(0, weight=1)  # Make row expandable
        selection_frame.grid_columnconfigure(0, weight=1)  # Make column expandable

        self.experiments_frame = tk.LabelFrame(selection_frame, text="Select Experiments/Groups to Plot")
        self.experiments_frame.grid(row=0, column=0, sticky="nsew")  # Ensure it fills available space

        # Compact plot options
# Compact plot options
        options_frame = tk.Frame(tab)
        options_frame.grid(row=1, column=0, padx=5, pady=5, sticky="we")
        #options_frame.grid_columnconfigure(index=0, weight=1)

        tk.Label(options_frame, text="Resample (1T = 1 min):").grid(row=0, column=0, sticky='w', padx=(5, 0))
        self.resample_entry = tk.Entry(options_frame, width=3)
        self.resample_entry.grid(row=0, column=1, padx=(0, 5))
        self.resample_entry.insert(0, "1T")  # Default to 1T for 1 minute

        tk.Label(options_frame, text="Avg Win:").grid(row=0, column=2, sticky='w', padx=(5, 0))
        self.moving_avg_entry = tk.Entry(options_frame, width=3)
        self.moving_avg_entry.grid(row=0, column=3, padx=(0, 5))
        self.moving_avg_entry.insert(0, "20")  # Default average window

        tk.Label(options_frame, text="Threshold:").grid(row=0, column=4, sticky='w', padx=(5, 0))
        self.threshold_entry = tk.Entry(options_frame, width=3)
        self.threshold_entry.grid(row=0, column=5, padx=(0, 5))

        self.normalize_var = tk.BooleanVar()
        normalize_chk = tk.Checkbutton(options_frame, text="Normalize", variable=self.normalize_var)
        normalize_chk.grid(row=0, column=6, padx=(5, 0))
        #self.threshold_entry.insert(0, "0")  # Default threshold

        self.normalize_var = tk.BooleanVar()
        tk.Checkbutton(options_frame, text="Normalize by total day activity", variable=self.normalize_var).grid(row=0, column=6, padx=5)

        # Compact variables selection
        variables_frame = tk.LabelFrame(tab, text="Select Variable(s)")
        variables_frame.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

        # Distribute checkboxes horizontally across the same row
        # Distribute checkboxes horizontally across the same row
        for i, (internal_name, display_name) in enumerate(self.variable_name_mapping.items()):
            var_chk = tk.BooleanVar()
            if internal_name == 'numb_mosquitos_flying':
                var_chk.set(True)  # Set this checkbox to be selected by default

            chk = tk.Checkbutton(variables_frame, text=display_name, variable=var_chk)
            chk.grid(row=0, column=i, sticky='w', padx=5)  # Place all items in row=0
            self.variable_vars.append((var_chk, internal_name, display_name))  # Add display name as third element

        # Compact date and time selection
        time_frame = tk.Frame(tab)
        time_frame.grid(row=3, column=0, padx=5, pady=5, sticky="we")
        time_frame.grid_columnconfigure(index=1, weight=1)
        time_frame.grid_columnconfigure(index=3, weight=1)

        tk.Label(time_frame, text="Start Date:").grid(row=0, column=0, sticky='w', padx=5)
        self.start_date_combobox = ttk.Combobox(time_frame, width=10)
        self.start_date_combobox.grid(row=0, column=1, padx=5)

        tk.Label(time_frame, text="End Date:").grid(row=0, column=2, sticky='w', padx=5)
        self.end_date_combobox = ttk.Combobox(time_frame, width=10)
        self.end_date_combobox.grid(row=0, column=3, padx=5)

        tk.Label(time_frame, text="Start Hour:").grid(row=0, column=4, sticky='w', padx=5)
        self.start_hour_scale = tk.Scale(time_frame, from_=0, to=23, orient=tk.HORIZONTAL, width=15, length=300)
        self.start_hour_scale.grid(row=0, column=5, padx=5, sticky='w')

        tk.Label(time_frame, text="End Hour:").grid(row=0, column=6, sticky='w', padx=5)
        self.end_hour_scale = tk.Scale(time_frame, from_=0, to=23, orient=tk.HORIZONTAL, width=15, length=300)
        self.end_hour_scale.grid(row=0, column=7, padx=5, sticky='w')
        self.end_hour_scale.set(23)  # Set default value to 23

        # Action frame
        action_frame = tk.Frame(tab)
        action_frame.grid(row=4, column=0, padx=5, pady=5, sticky="ew")

        plot_button = tk.Button(action_frame, text="Plot Data", command=self.plot_data)
        plot_button.pack(side=tk.LEFT, padx=5)

        plot_type_label = tk.Label(action_frame, text="Plot Type:")
        plot_type_label.pack(side=tk.LEFT, padx=5)

        self.plot_type_combobox = ttk.Combobox(action_frame, values=["Entire Time Series", "Daily Average", "Average Over Days", "Scatter plot variability"], width=20)
        self.plot_type_combobox.set("Entire Time Series")
        self.plot_type_combobox.pack(side=tk.LEFT, padx=5)

        # Add a checkbutton for stacked plot option
        self.stacked_plot_var = tk.BooleanVar()
        stacked_plot_chk = tk.Checkbutton(action_frame, text="Stacked Subplots", variable=self.stacked_plot_var)
        stacked_plot_chk.pack(side=tk.LEFT, padx=5)


        self.populate_selection_checkbuttons()

    def plot_data(self):
        # Retrieve selected internal variable names and display names
        selected_vars = [(internal_name, display_name) for var_chk, internal_name, display_name in self.variable_vars if var_chk.get()]

        if not selected_vars:
            self.log("No variables selected for plotting.")
            return

        resample_interval = self.resample_entry.get()
        moving_avg_window = self.moving_avg_entry.get()
        start_date_str = self.start_date_combobox.get()
        end_date_str = self.end_date_combobox.get()
        threshold = self.threshold_entry.get()

        start_hour = self.start_hour_scale.get()
        end_hour = self.end_hour_scale.get()
        normalize = self.normalize_var.get()

        if not start_date_str or not end_date_str:
            self.log("Start date or end date not selected.")
            return

        if start_hour >= end_hour:
            self.log("Start hour must be less than end hour.")
            return

        experiments_to_plot = []
        groups_to_plot = []
        categories_to_plot = []

        for var, item in self.experiment_vars:
            if var.get():
                if isinstance(item, str):
                    if item in self.grouped_experiments:
                        categories_to_plot.append(item)
                    else:
                        groups_to_plot.append(item)
                else:
                    experiments_to_plot.append(item)

        plot_type = self.plot_type_combobox.get()
        plot_args_base = {
            "resample_interval": resample_interval,
            "moving_avg_window": moving_avg_window,
            "start_date_str": start_date_str,
            "end_date_str": end_date_str,
            "threshold": threshold,
            "experiments_to_plot": experiments_to_plot,
            "groups_to_plot": groups_to_plot,
            "categories_to_plot": categories_to_plot,
            "grouped_experiments": self.grouped_experiments,
            "start_hour": start_hour,
            "end_hour": end_hour,
            "normalize": normalize
        }

        if self.stacked_plot_var.get():
            num_vars = len(selected_vars)
            fig, axes = plt.subplots(num_vars, 1, figsize=(10, 3 * num_vars), sharex=True, constrained_layout=True)

            if num_vars == 1:
                axes = [axes]

            for ax, (internal_name, display_name) in zip(axes, selected_vars):
                plot_args = {**plot_args_base, "selected_vars": [internal_name], "ax": ax, "display_names": display_name}
                if plot_type == "Entire Time Series":
                    self.plot_manager.plot_entire_time_series(**plot_args)
                elif plot_type == "Daily Average":
                    self.plot_manager.plot_daily_average(**plot_args)
                elif plot_type == "Average Over Days":
                    self.plot_manager.plot_avg_over_days(**plot_args)
                elif plot_type == "Scatter plot variability":
                    self.plot_manager.plot_scatter_variability(**plot_args)

            filename = f"{plot_type}_stacked_subplots"
            self.plot_manager.save_plot(fig, filename)
            plt.show()
            plt.close(fig)

        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_args = {**plot_args_base, "selected_vars": [internal_name for internal_name, display_name in selected_vars], "ax": ax, "display_names": [display_name for internal_name, display_name in selected_vars]}

            if plot_type == "Entire Time Series":
                self.plot_manager.plot_entire_time_series(**plot_args)
            elif plot_type == "Daily Average":
                self.plot_manager.plot_daily_average(**plot_args)
            elif plot_type == "Average Over Days":
                self.plot_manager.plot_avg_over_days(**plot_args)
            elif plot_type == "Scatter plot variability":
                self.plot_manager.plot_scatter_variability(**plot_args)

        # Base of the filename
        exp_names = [exp_data['alias'] for exp_data in experiments_to_plot]
        vars_part = "_".join([display_name.replace(' ', '-') for _, display_name in selected_vars])
        exp_part = "_and_".join(exp_names) if exp_names else "NoExperiment"
        group_part = "_and_".join(groups_to_plot) if groups_to_plot else "NoGroup"
        category_part = "_and_".join(categories_to_plot) if categories_to_plot else "NoCategory"

        exp_group_cat_part = "_and_".join(filter(None, [exp_part, group_part, category_part]))
        filename = f"{plot_type}_{vars_part}_{exp_group_cat_part}_Normalize_{normalize}"

        self.plot_manager.save_plot(fig, filename)
        plt.show()
        plt.close(fig)

    def _execute_plot_method(self, plot_type, plot_args):
        if plot_type == "Entire Time Series":
            self.plot_manager.plot_entire_time_series(**plot_args)
        elif plot_type == "Daily Average":
            self.plot_manager.plot_daily_average(**plot_args)
        elif plot_type == "Average Over Days":
            self.plot_manager.plot_avg_over_days(**plot_args)
        elif plot_type == "Scatter plot variability":
            self.plot_manager.plot_scatter_variability(**plot_args)


#################  Flight statistics tab ##################

    def create_flight_statistics_analysis_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Flight Statistics Analysis")

        self.configure_statistics_analysis_tab(tab)

    def configure_statistics_analysis_tab(self, tab):
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_rowconfigure(2, weight=1)
        tab.grid_rowconfigure(3, weight=1)
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=1)

        # Dimensionality reduction section
        red_frame = tk.LabelFrame(tab, text="Dimensionality Reduction")
        red_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

        reduction_method_label = tk.Label(red_frame, text="Reduction Method:")
        reduction_method_label.pack(side=tk.LEFT, padx=5)
        
        self.reduction_method_combobox = ttk.Combobox(red_frame, values=["PCA", "t-SNE", "UMAP"],width=10)
        self.reduction_method_combobox.set("PCA")
        self.reduction_method_combobox.pack(side=tk.LEFT, padx=5)

        data_level_label = tk.Label(red_frame, text="Data Level:")
        data_level_label.pack(side=tk.LEFT, padx=5)

        self.data_level_combobox = ttk.Combobox(red_frame, values=["Experiments", "Groups", "Categories"],width=10)
        self.data_level_combobox.set("Experiments")
        self.data_level_combobox.pack(side=tk.LEFT, padx=5)
        
        interval_label = tk.Label(red_frame, text="Interval (minutes):")
        interval_label.pack(side=tk.LEFT, padx=5)

        self.interval_entry = tk.Entry(red_frame, width=5)
        self.interval_entry.insert(0, "20")  # Default to 20 minutes
        self.interval_entry.pack(side=tk.LEFT, padx=5)



        self.normalize_dim_reduction_var = tk.BooleanVar()
        normalize_checkbox = tk.Checkbutton(red_frame, text="Normalize by Total Daily Activity", variable=self.normalize_dim_reduction_var)
        normalize_checkbox.pack(side=tk.LEFT, padx=5)

        self.plot_days_annotation = tk.BooleanVar()
        annotate_checkbox = tk.Checkbutton(red_frame, text="Annotate days", variable=self.plot_days_annotation)
        annotate_checkbox.pack(side=tk.LEFT, padx=5)

        dimensionality_button = tk.Button(red_frame, text="Run Dimensionality Reduction", command=self.run_dimensionality_reduction)
        dimensionality_button.pack(side=tk.LEFT, padx=5)



        # Analysis method section
        analysis_frame = tk.LabelFrame(tab, text="Statistical Analysis")
        analysis_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        analysis_frame.grid_columnconfigure(0, weight=1)
        analysis_frame.grid_columnconfigure(1, weight=1)

        analysis_method_label = tk.Label(analysis_frame, text="Analysis Method:")
        analysis_method_label.grid(row=0, column=0, padx=5, sticky="w")

        self.analysis_method_combobox = ttk.Combobox(analysis_frame, values=["ANOVA", "GLMM"])
        self.analysis_method_combobox.set("ANOVA")
        self.analysis_method_combobox.grid(row=0, column=1, padx=5, sticky="w")

        interval_size_label = tk.Label(analysis_frame, text="Day Interval Size:")
        interval_size_label.grid(row=1, column=0, padx=5, sticky="w")

        # Add 'All Days' as an option
        self.day_interval_size_combobox = ttk.Combobox(analysis_frame, values=[str(i) for i in range(1, 15)] + ["All Days"])
        self.day_interval_size_combobox.set("5")
        self.day_interval_size_combobox.grid(row=1, column=1, padx=5, sticky="w")

        time_interval_size_label = tk.Label(analysis_frame, text="Time Interval Size (hours):")
        time_interval_size_label.grid(row=2, column=0, padx=5, sticky="w")

        self.time_interval_size_combobox = ttk.Combobox(analysis_frame, values=[str(i) for i in range(1, 24)])  # Interval size options
        self.time_interval_size_combobox.set("4")
        self.time_interval_size_combobox.grid(row=2, column=1, padx=5, sticky="w")

        run_analysis_button = tk.Button(analysis_frame, text="Run Statistical Analysis", 
                                        command=lambda: self.run_automated_stat_analysis(method=self.analysis_method_combobox.get().lower()))
        run_analysis_button.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

        load_results_button = tk.Button(analysis_frame, text="Load and Visualize Results", command=self.load_and_visualize_results)
        load_results_button.grid(row=4, column=0, padx=5, pady=5, sticky="ew")


        # New section for Simple Tests with Bar and Scatter Plots
        simple_test_frame = tk.LabelFrame(tab, text="Simple Tests Between Conditions")
        simple_test_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        simple_test_frame.grid_columnconfigure(0, weight=1)
        simple_test_frame.grid_columnconfigure(1, weight=1)

        start_date_label = tk.Label(simple_test_frame, text="Start Date:")
        start_date_label.grid(row=0, column=0, padx=5, sticky="w")
        self.start_date_entry = tk.Entry(simple_test_frame)
        self.start_date_entry.grid(row=0, column=1, padx=5, sticky="w")

        end_date_label = tk.Label(simple_test_frame, text="End Date:")
        end_date_label.grid(row=1, column=0, padx=5, sticky="w")
        self.end_date_entry = tk.Entry(simple_test_frame)
        self.end_date_entry.grid(row=1, column=1, padx=5, sticky="w")

        start_time_label = tk.Label(simple_test_frame, text="Start Time (Hour):")
        start_time_label.grid(row=2, column=0, padx=5, sticky="w")
        self.start_time_entry = tk.Entry(simple_test_frame)
        self.start_time_entry.grid(row=2, column=1, padx=5, sticky="w")

        end_time_label = tk.Label(simple_test_frame, text="End Time (Hour):")
        end_time_label.grid(row=3, column=0, padx=5, sticky="w")
        self.end_time_entry = tk.Entry(simple_test_frame)
        self.end_time_entry.grid(row=3, column=1, padx=5, sticky="w")

        # plot_bar_button = tk.Button(simple_test_frame, text="Plot Bar Graph", command=self.plot_bar_graph)
        # plot_bar_button.grid(row=4, column=0, padx=5, pady=5, sticky="ew")

        # plot_scatter_button = tk.Button(simple_test_frame, text="Plot Scatter Plot", command=self.plot_scatter_plot)
        # plot_scatter_button.grid(row=4, column=1, padx=5, pady=5, sticky="ew")

    def run_dimensionality_reduction(self):
        selected_var = 'numb_mosquitos_flying'
        data_level = self.data_level_combobox.get().lower()
        interval_hours = int(self.interval_entry.get())
        
        combined_data = self.get_combined_data_for_dimension_reduction(selected_var, data_level)

        # if normalize:
        #     combined_data = self.plot_manager.normalize_by_daily_total(combined_data)
        #print(combined_data)
        # Ensure the index is set correctly as a DatetimeIndex
        if not isinstance(combined_data.index, pd.DatetimeIndex):
            combined_data.set_index('Date', inplace=True)

        if combined_data.empty:
            self.log("No data available for dimensionality reduction.")
            return
        #print(interval_hours)

            # Remove the last day
        max_date = combined_data.index.max().date()  # Get the last day
        combined_data = combined_data[combined_data.index.date < max_date]  # Exclude entries from the last day
        #print(combined_data)

        interval_features = self.prepare_interval_features(combined_data, selected_var, interval_hours)
        interval_features.dropna(inplace=True)

        days = interval_features['Date']
        experiments = interval_features['Experiment']
        #print(experiments)
        feature_data = interval_features.drop(columns=['Date', 'Experiment'])
        #print(feature_data)
        #print(feature_data)
        method = self.reduction_method_combobox.get().lower()
        
        reduced_data = self.stat_analysis_manager.run_dimensionality_reduction(feature_data, method=method)
        #print(reduced_data)
        #print(reduced_data)
        # Prepare reduced dataframe (for example purposes) to include in visualization call
        reduced_df = pd.DataFrame(reduced_data, index=experiments)

        plot_days = self.plot_days_annotation.get()
        # Use PlotManager to visualize the reduced data
        self.plot_manager.visualize_reduced_data(reduced_df, self.grouped_experiments, method, days,plot_days,interval_hours,data_level,self.normalize_dim_reduction_var.get())


    def get_combined_data_for_dimension_reduction(self, var, data_level):
        combined_records = []
        
        if data_level == "experiments":
            combined_df = self.get_combined_data([var])
            combined_records = combined_df.reset_index().to_dict('records')
        elif data_level == "groups":
            for category_name, groups in self.grouped_experiments.items():
                for group_name in groups:
                    experiments = groups[group_name]
                    group_data_df = self.aggregate_experiment_data(
                        experiments, var, level_identifier="group", name=group_name
                    )
                    combined_records.extend(group_data_df)
        elif data_level == "categories":
            for category_name, groups in self.grouped_experiments.items():
                category_experiments = [exp for group in groups.values() for exp in group]
                category_data_df = self.aggregate_experiment_data(
                    category_experiments, var, level_identifier="category", name=category_name
                )
                combined_records.extend(category_data_df)

        # Convert to DataFrame
        combined_df = pd.DataFrame(combined_records)
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        combined_df.set_index('Date', inplace=True)
        
        return combined_df

    def calculate_average_for_group_or_category(self, name, var, level_type):
        result = []
        if level_type == "group":
            for _, group in self.grouped_experiments.items():  # passing through categories and checking group
                if name in group:
                    group_data = self.aggregate_experiment_data(group[name], var, level_type,name)
                    result.extend(group_data)
        elif level_type == "category":
            if name in self.grouped_experiments:
                for group_name, group in self.grouped_experiments[name].items():
                    group_data = self.aggregate_experiment_data(group, var, level_type,group_name)
                    result.extend(group_data)
        return result

    def aggregate_experiment_data(self, experiments, var, level_identifier, name):
        data_collection = []

        for exp_data in experiments:
            population_data = exp_data['population_data']
            individual_data = exp_data['individual_data']

            var_data = self.get_var_data(var, population_data, individual_data, None)
            var_data = var_data[~var_data.index.duplicated(keep='first')]
            data_collection.append(var_data)

        if not data_collection:
            return []

        # Determine the common time range across all data
        overall_start_date = min(data.index.min() for data in data_collection)
        overall_end_date = max(data.index.max() for data in data_collection)

        unified_index = pd.date_range(start=overall_start_date, end=overall_end_date, freq='T')

        # Aggregate Data
        aggregated_data_frames = [data.reindex(unified_index, fill_value=np.nan) for data in data_collection]
        all_data_df = pd.concat(aggregated_data_frames, axis=1)
        avg_data = all_data_df.mean(axis=1, skipna=True)
        
        # Create DataFrame with combined information
        result_collection = [{
        'Timestamp': dt,
        'Date': dt.date(),
        'Day': dt.day,
        'Experiment': name,
        var: value
    } for dt, value in avg_data.items()]
        
        return result_collection



    def prepare_interval_features(self, df, variable, interval_minutes):
        # Ensure the Timestamp column is properly set as a Datetime object
        if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # Define start times from 00:00 to 23:59 with specified minute interval
        interval_delta = pd.Timedelta(minutes=interval_minutes)
        start_times = pd.date_range("00:00", "23:59", freq=interval_delta)
        features = []
        keys = ['mean', 'std', 'min', 'max']
        
        for (date, experiment), group in df.groupby([df['Timestamp'].dt.date, 'Experiment']):
            stats_dict = {'Date': date, 'Experiment': experiment}
            
            for i, start_time in enumerate(start_times):
                # Calculate actual interval start and end times
                interval_start = time(start_time.hour, start_time.minute)
                if i + 1 < len(start_times):
                    interval_end = time(start_times[i + 1].hour, start_times[i + 1].minute)
                else:
                    interval_end = time(23, 59)

                # Filter the group using the Timestamp column
                interval_df = group[(group['Timestamp'].dt.time >= interval_start) & (group['Timestamp'].dt.time < interval_end)]
                    
                if not interval_df.empty:
                    stats = interval_df[variable].agg(keys)
                    for key in keys:
                        stats_dict[f'{interval_start}-{interval_end}_{key}'] = stats[key]
                else:
                    for key in keys:
                        stats_dict[f'{interval_start}-{interval_end}_{key}'] = np.nan
                    
            features.append(stats_dict)

        features_df = pd.DataFrame(features)
        features_df.fillna(method='ffill', inplace=True)
        features_df.reset_index(drop=True, inplace=True)
        features_df.dropna(axis=1, how='all', inplace=True)

        return features_df



    def reduce_dimensionality(self,data, method='pca', n_components=2):
        if method == 'pca':
            model = PCA(n_components=n_components)
        elif method == 'tsne':
            model = TSNE(n_components=n_components, metric='euclidean')
        elif method == 'umap':
            model = umap.UMAP(n_components=n_components, metric='euclidean')
        else:
            raise ValueError(f"Unknown method: {method}")
            
        reduced_data = model.fit_transform(data)
        return reduced_data

    def prepare_daily_features(self,df, variable):
        df['Date'] = df.index.date
        
        daily_features = df.groupby(['Date', 'Experiment'])[variable].agg(['mean', 'std', 'max', 'min', np.median]).reset_index()
        daily_features.columns = ['Date', 'Experiment', 'mean', 'std', 'max', 'min', 'median']
        return daily_features
    




    def generate_time_intervals(self,time_interval_size):
        start_hours = np.arange(24)
        time_intervals = []
        for start_hour in start_hours:
            end_hour = (start_hour + time_interval_size) % 24
            if start_hour < end_hour:
                time_intervals.append((start_hour, end_hour, False))
            else:
                time_intervals.append((start_hour, end_hour, True))
        return time_intervals
    

    def run_automated_stat_analysis(self, method='glmm'):
        selected_var = 'numb_mosquitos_flying'  # Variable to analyze
        fixed_effects = ["Condition"]  # Only Condition is the fixed effect
        random_effects = ["Day", "Experiment"]  # Day and Experiment are random effects
        results = {factor: {"effect_size": [], "p_value": [], "z_value": []} for factor in fixed_effects}  # Initialize results

        combined_data = self.get_combined_data([selected_var])
        if combined_data.empty:
            self.log("Combined data is empty. Aborting analysis.")
            return

        start_date = combined_data.index.min().date()
        end_date = combined_data.index.max().date()

        days = pd.date_range(start_date, end_date, freq='D')
        day_interval_size = self.day_interval_size_combobox.get()

            # Define day intervals
        if day_interval_size == "All Days":
            day_intervals = [(days[0], days[-1])]
        else:
            day_interval_size = int(day_interval_size)
            day_intervals = [(days[i], days[min(i + day_interval_size - 1, len(days) - 1)]) for i in range(len(days) - day_interval_size + 1)]

        time_interval_size = int(self.time_interval_size_combobox.get())
        time_intervals = self.generate_time_intervals(time_interval_size)


        for day_interval in day_intervals:
            for start_hour, end_hour, span_midnight in time_intervals:
                segment_data = self.get_data_for_interval(selected_var, day_interval[0], day_interval[1], start_hour, end_hour, span_midnight)
                #print(segment_data)
                if segment_data.empty:
                    self.log(f"No data for interval {day_interval[0]} to {day_interval[1]}, {start_hour:02d}:00 to {end_hour:02d}:00.")
                    continue

                #result = self.stat_analysis_manager.analyze_variability(
                    segment_data, selected_var, start_hour, end_hour, fixed_effects=fixed_effects, random_effects=random_effects, method=method
                #)
                result = self.stat_analysis_manager.perform_glmm(segment_data, selected_var, fixed_effects, random_effects, self.common_save_dir, f"glmm_{day_interval[0]}_{start_hour}_{end_hour}")
                #print(result)
        #         if result is not None:
        #             for factor in fixed_effects:
        #                 try:
        #                     index = 0#result["Factor"].index(factor)
        #                     effect_size = result["coefficient"][index]
        #                     p_value = result["p_value"][index]
        #                     z_value = result["z_value"][index]
        #                 except (ValueError, KeyError) as e:
        #                     self.log(f"Error extracting factor '{factor}': {e}")
        #                     effect_size, p_value, z_value = np.nan, np.nan, np.nan
                        
        #                 results[factor]["effect_size"].append({
        #                     "day_interval": (day_interval[0].strftime('%Y-%m-%d'), day_interval[1].strftime('%Y-%m-%d')),
        #                     "time_interval": f"{start_hour:02d}:00-{end_hour:02d}:00",
        #                     "effect_size": effect_size,
        #                 })
        #                 results[factor]["p_value"].append({
        #                     "day_interval": (day_interval[0].strftime('%Y-%m-%d'), day_interval[1].strftime('%Y-%m-%d')),
        #                     "time_interval": f"{start_hour:02d}:00-{end_hour:02d}:00",
        #                     "p_value": p_value,
        #                 })
        #                 results[factor]["z_value"].append({
        #                     "day_interval": (day_interval[0].strftime('%Y-%m-%d'), day_interval[1].strftime('%Y-%m-%d')),
        #                     "time_interval": f"{start_hour:02d}:00-{end_hour:02d}:00",
        #                     "z_value": z_value,
        #                 })

        # # Apply multiple testing correction after collecting all results
        # for factor in fixed_effects:
        #     if results[factor]["p_value"]:
        #         p_values = [entry["p_value"] for entry in results[factor]["p_value"] if np.isfinite(entry["p_value"])]
                
        #         corrected_p_values = []
        #         if p_values:
        #             _, corrected_p_values_tmp, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        #             corrected_p_values_iter = iter(corrected_p_values_tmp)
                    
        #             for entry in results[factor]["p_value"]:
        #                 if np.isfinite(entry["p_value"]):
        #                     entry["corrected_p_value"] = next(corrected_p_values_iter)
        #                 else:
        #                     entry["corrected_p_value"] = np.nan

        #         effect_size_df = pd.DataFrame(results[factor]["effect_size"])
        #         p_values_df = pd.DataFrame(results[factor]["p_value"])
        #         z_values_df = pd.DataFrame(results[factor]["z_value"])
        #         p_values_df['z_value'] = z_values_df['z_value']
                
        #         result_dir = self.common_save_dir
        #         if result_dir:
        #             effect_size_df.to_csv(os.path.join(result_dir, f"effect_size_{factor}_{method}.csv"), index=False)
        #             p_values_df.to_csv(os.path.join(result_dir, f"p_values_{factor}_{method}.csv"), index=False)
        #             self.log(f"{method.upper()} results saved in {result_dir}")


    def load_and_visualize_results(self):
        result_dir = self.common_save_dir

        method = self.analysis_method_combobox.get().lower()

        if method == "glmm":
            factors = ["Condition"]
        elif method == 'anova':
            factors  = ["Condition","Day", "Experiment"]
       
        if not result_dir:
            self.log("Common result directory is not set. Please finalize group settings first.")
            return

        for factor in factors:
            effect_size_path = os.path.join(result_dir, f"effect_size_{factor}_{method}.csv")
            p_values_path = os.path.join(result_dir, f"p_values_{factor}_{method}.csv")

            if os.path.exists(effect_size_path) and os.path.exists(p_values_path):
                effect_size_df = pd.read_csv(effect_size_path)
                p_values_df = pd.read_csv(p_values_path)
                
                # Ensure 'neg_log_p_value' and 'z_value' columns
                p_values_df['neg_log_p_value'] = p_values_df['p_value'].apply(lambda x: -np.log10(max(x, 0.0001)))
                if 'z_value' not in p_values_df.columns:
                    p_values_df['z_value'] = np.nan
                
                # Calculate the start date
                start_date = pd.to_datetime(effect_size_df['day_interval'].str.extract(r"'(\d{4}-\d{2}-\d{2})'")[0]).min().normalize()
                
                # Calculate days_since_start
                effect_size_df['days_since_start'] = effect_size_df['day_interval'].apply(lambda x: (pd.to_datetime(x.split(',')[0].strip("('")) - start_date).days)
                p_values_df['days_since_start'] = p_values_df['day_interval'].apply(lambda x: (pd.to_datetime(x.split(',')[0].strip("('")) - start_date).days)

                # Calculate median time interval
                effect_size_df['median_time_interval'] = effect_size_df['time_interval'].apply(self.plot_manager.calculate_median_time_interval)
                p_values_df['median_time_interval'] = p_values_df['time_interval'].apply(self.plot_manager.calculate_median_time_interval)

                # Determine start and end dates as strings
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = (start_date + pd.Timedelta(days=effect_size_df['days_since_start'].max())).strftime('%Y-%m-%d')

                # Call the PlotManager's method
                self.plot_manager.plot_heatmaps(effect_size_df, p_values_df, p_values_df, factor, method, start_date_str, end_date_str)
            else:
                self.log("Effect size or p-values file not found in the common result directory.")

    def get_data_for_interval(self, var, start_day, end_day, start_hour, end_hour, span_midnight=False):
        combined_data = []
        for group_name, experiments in self.grouped_experiments.items():
            for exp_data in experiments:
                alias = exp_data['alias']
                condition = group_name
                population_data = exp_data['population_data']
                individual_data = exp_data['individual_data']

                var_data = self.get_var_data(var, population_data, individual_data, None)
                if var_data is None or var_data.empty:
                    self.log(f"No data available for variable '{var}' in experiment '{alias}' for the given time interval.")
                    continue

                var_data = var_data[start_day:end_day]
                if span_midnight:
                    var_data = pd.concat([var_data.between_time(f'{start_hour:02d}:00', '23:59'), var_data.between_time('00:00', f'{end_hour:02d}:00')])
                else:
                    var_data = var_data.between_time(f'{start_hour:02d}:00', f'{end_hour:02d}:00')
                    
                
                if var_data.empty:
                    continue
                


                # Calculate daily average for the specified time interval
                daily_avg = var_data.groupby(var_data.index.date).mean()
                #print(daily_avg)
                # Append the averaged data to the combined_data list

                for timestamp, value in daily_avg.items():
                    combined_data.append({
                        'Timestamp': timestamp,
                        'Day': f"Day {timestamp.day}",
                        'Experiment': alias,
                        'Condition': condition,
                        var: value
                    })

        if not combined_data:
            self.log("No data found for the given interval.")
            return pd.DataFrame()

        combined_df = pd.DataFrame(combined_data)
        if 'Timestamp' not in combined_df.columns:
            self.log("Timestamp column is missing in combined_df.")
            return pd.DataFrame()

        combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'])
        combined_df.set_index('Timestamp', inplace=True)
        return combined_df


    def get_var_data(self, var, population_data, individual_data, threshold):
        if var in population_data.columns:
            x = population_data[var]
            x = x.interpolate()
            x = x.resample("1T").mean()
            x = x.rolling(window="5T").mean()

            normalize = self.normalize_dim_reduction_var.get()

            if normalize:
                x = self.plot_manager.normalize_by_daily_total(x)
            return x
        elif var in individual_data.columns:
            var_data = individual_data[var]
            if var in ["flight_duration", "average_speed"] and threshold:
                threshold_value = float(threshold)
                var_data = var_data[var_data > threshold_value]
            return var_data
        else:
            self.log(f"Variable '{var}' not found in experiment data.")
            return pd.Series()

    def crop_time(self, t_i, t_f, df):
        mask = (df.index > t_i) & (df.index < t_f)
        df = df.loc[mask]
        return df
    
    def get_combined_data(self, selected_vars):
        combined_data = []
        
        for category_name, groups in self.grouped_experiments.items():
            for group_name, experiments in groups.items():
                for exp_data in experiments:
                    alias = exp_data['alias']
                    condition = group_name  # Assuming group_name acts as a condition identifier
                    
                    population_data = exp_data.get('population_data')
                    individual_data = exp_data.get('individual_data')

                    if not population_data.empty:
                        for var in selected_vars:
                            var_data = self.get_var_data(var, population_data, individual_data, None)
                            for timestamp, value in var_data.items():
                                combined_data.append({
                                    'Timestamp': timestamp,
                                    'Date': timestamp.date(),
                                    'Day': timestamp.day,
                                    'Experiment': alias,
                                    'Condition': condition,
                                    var: value
                                })

        combined_df = pd.DataFrame(combined_data)
        combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'])
        combined_df.set_index('Timestamp', inplace=True)
        return combined_df
    


    # def plot_bar_graph(self):
    #     start_date = self.start_date_entry.get()
    #     end_date = self.end_date_entry.get()
    #     start_time = int(self.start_time_entry.get())
    #     end_time = int(self.end_time_entry.get())

    #     # Extract the required data
    #     data = self.get_data_for_interval('numb_mosquitos_flying', start_date, end_date, start_time, end_time, span_midnight=start_time > end_time)
        
    #     # Call the PlotManager's method
    #     self.plot_manager.plot_bar_graph(data, 'numb_mosquitos_flying', start_date, end_date, start_time, end_time)

    # def plot_scatter_plot(self):
    #     start_date = self.start_date_entry.get()
    #     end_date = self.end_date_entry.get()
    #     start_time = int(self.start_time_entry.get())
    #     end_time = int(self.end_time_entry.get())

    #     # Extract the required data
    #     data = self.get_data_for_interval('numb_mosquitos_flying', start_date, end_date, start_time, end_time, span_midnight=start_time > end_time)
        
    #     # Call the PlotManager's method
    #     self.plot_manager.plot_scatter_plot(data, 'numb_mosquitos_flying', start_date, end_date, start_time, end_time)


    # def visualize_reduced_data(self, reduced_data, labels, method, days):
    #     reduced_df = pd.DataFrame(reduced_data, index=labels)
    #     plt.figure(figsize=(10, 8))

    #     marker_map = {
    #         'Ae_aegypti_formosus': 'o',  # Circle marker
    #         'Ae_aegypti_aegypti': 's',   # Square marker
    #         'Ae_albopictus': 'd',        # Diamond marker
    #         'An_stephensi': '^',         # Triangle marker
    #         'An_gambiae': 'v',           # Inverted triangle marker
    #         'An_coluzzi': 'p',           # Pentagon marker
    #     }

    #     group_color_map = {
    #         'KPP': 'tab:blue', 'CAY': 'tab:cyan', 'COL': 'tab:purple', 'GUA': 'tab:red',
    #         'PHN': 'tab:pink', 'ZIK': 'tab:olive', 'RAB': 'tab:green', 'KUM': 'tab:orange',
    #         'KED': 'tab:brown', 'KAK': 'tab:gray'
    #     }

    #     experiment_to_group_category = {}
    #     for category_name, groups in self.grouped_experiments.items():
    #         for group_name, experiment_list in groups.items():
    #             for exp_data in experiment_list:
    #                 alias = exp_data.get('alias')
    #                 if alias:
    #                     experiment_to_group_category[alias] = (group_name, category_name)

    #     handles, legend_labels = [], []
    #     start_date = pd.to_datetime(days[0])  # Assume days are datetime strings

    #     for idx, (label, row) in enumerate(reduced_df.iterrows()):
    #         if label in experiment_to_group_category:
    #             group_name, category_name = experiment_to_group_category[label]
    #         elif label in self.grouped_experiments:
    #             category_name = label
    #             group_name = None
    #         else:
    #             group_name = label
    #             for cat_name, groups in self.grouped_experiments.items():
    #                 if group_name in groups:
    #                     category_name = cat_name
    #                     break
    #             else:
    #                 category_name = "Unknown"

    #         marker = marker_map.get(category_name, 'o')
    #         color = group_color_map.get(group_name, 'gray') if group_name else 'gray'

    #         sc = plt.scatter(
    #             row[0], row[1],
    #             color=color,
    #             marker=marker,
    #             s=50,
    #             alpha=0.5,
    #             label=f"{group_name}_{category_name}" if f"{group_name}_{category_name}" not in legend_labels else ""
    #         )
    #         if f"{group_name}_{category_name}" not in legend_labels:
    #             handles.append(sc)
    #             legend_labels.append(f"{group_name}_{category_name}")

    #         # Annotate with the day
    #         current_date = pd.to_datetime(days[idx])
    #         days_since_start = (current_date - start_date).days
    #         plt.annotate(days_since_start, (row[0], row[1]), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=8)


    #     plt.title(f'{method.upper()} Dimensionality Reduction')
    #     plt.xlabel("Component 1")
    #     plt.ylabel("Component 2")
    #     plt.legend(handles=handles, labels=legend_labels, title='Group/Category', fontsize='small', loc='best')
    #     plt.grid(True)
    #     plt.show()