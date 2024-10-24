

from common_imports import *

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from plot_manager import PlotManager
from statistical_analysis_manager import StatisticalAnalysisManager
from statsmodels.stats.multitest import multipletests
from matplotlib.markers import MarkerStyle
from matplotlib import colors
from glmm_analysis_manager import GLMMAnalysisManager
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import matplotlib.colors as mcolors
import datetime

warnings.filterwarnings("ignore", category=ConvergenceWarning)

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
            'sugar_feeding_index': 'Sugar Feeding Activity',  # New variable
            'numb_mosquitos_left_ctrl': 'numb_mosquitos_left_ctrl',
            'numb_mosquitos_right_ctrl': 'numb_mosquitos_right_ctrl',
            'flight_duration': 'flight_duration',
            'average_speed': 'average_speed'
        }

        

        # Initialize group and category assignments
        self.exp_group_assignments = {}
        self.exp_category_assignments = {}  # New addition

                # Initialize GLMM results directory
        self.glmm_results_dir = os.path.join(os.getcwd(), 'glmm_results')
        self.glmm_analysis_manager = GLMMAnalysisManager(self.log, self.glmm_results_dir)

    def compute_sugar_feeding_index(self, population_data):
        sugar_feeding_index = population_data['numb_mosquitos_sugar'] + population_data['numb_mosquitos_hs']
        # Optionally apply more complex transformations or filters if needed
        return sugar_feeding_index
    
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
                        sugar_feeding_index = self.compute_sugar_feeding_index(full_population_data)
                        full_population_data['sugar_feeding_index'] = sugar_feeding_index
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
            if internal_name in ['numb_mosquitos_left_ctrl', 'numb_mosquitos_right_ctrl']:
                continue  # Skip the rest of the loop for these variables

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

        # Define specific dimensions (width, single_subplot_height) for each plot type
        plot_base_dimensions = {
            "Entire Time Series": (15, 3),
            "Daily Average": (8, 5),
            "Average Over Days": (8, 5),
            "Scatter plot variability": (8, 4)
        }

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

            # Set figure size for stacked plots using base height times the number of variables
            base_width, single_height = plot_base_dimensions.get(plot_type, (10, 3))  # Default to (10, 3)
            total_height = single_height * num_vars
            fig, axes = plt.subplots(num_vars, 1, figsize=(base_width, total_height), sharex=True, constrained_layout=True)

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
            # Look up the figsize for the non-stacked plot
            base_width, single_height = plot_base_dimensions.get(plot_type, (10, 6))  # Default to (10, 6) for single plots
            fig, ax = plt.subplots(figsize=(base_width, single_height))
            plot_args = {**plot_args_base,
                        "selected_vars": [internal_name for internal_name, display_name in selected_vars],
                        "ax": ax,
                        "display_names": [display_name for internal_name, display_name in selected_vars]}

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
        analysis_frame = tk.LabelFrame(tab, text="GLMM Analysis Configuration")
        analysis_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        analysis_frame.grid_columnconfigure(0, weight=1)
        analysis_frame.grid_columnconfigure(1, weight=1)

        # Variable Selection
        variable_label = tk.Label(analysis_frame, text="Variable:")
        variable_label.grid(row=0, column=0, padx=5, sticky="w")

        # Update combo box with display names
        self.variable_combobox = ttk.Combobox(
            analysis_frame, 
            values=list(self.variable_name_mapping.values())
        )
        self.variable_combobox.set("numb_mosquitos_flying")  # Ensure this matches a display name
        self.variable_combobox.grid(row=0, column=1, padx=5, sticky="w")

        # Day Interval Size
        day_interval_label = tk.Label(analysis_frame, text="Day Interval Size (days):")
        day_interval_label.grid(row=1, column=0, padx=5, sticky="w")

        self.day_interval_size_combobox = ttk.Combobox(analysis_frame, values=[str(i) for i in range(3, 8)] + ["All Days"])
        self.day_interval_size_combobox.set("All Days")
        self.day_interval_size_combobox.grid(row=1, column=1, padx=5, sticky="w")

        # adjust the logic to enforce a 10-minute update
        time_interval_label = tk.Label(analysis_frame, text="Time Interval Size (minutes):")
        time_interval_label.grid(row=2, column=0, padx=5, sticky="w")

        self.time_interval_size_combobox = ttk.Combobox(analysis_frame, values=[str(i) for i in range(10, 61, 10)])
        self.time_interval_size_combobox.set("30")
        self.time_interval_size_combobox.grid(row=2, column=1, padx=5, sticky="w")

        # Fixed Effects
        fixed_effects_label = tk.Label(analysis_frame, text="Fixed Effects (comma-separated):")
        fixed_effects_label.grid(row=3, column=0, padx=5, sticky="w")

        self.fixed_effects_entry = tk.Entry(analysis_frame)
        self.fixed_effects_entry.insert(0, "Category")
        self.fixed_effects_entry.grid(row=3, column=1, padx=5)

        # Random Effects
        random_effects_label = tk.Label(analysis_frame, text="Random Effects (comma-separated):")
        random_effects_label.grid(row=4, column=0, padx=5, sticky="w")

        self.random_effects_entry = tk.Entry(analysis_frame)
        self.random_effects_entry.insert(0, "Experiment")
        self.random_effects_entry.grid(row=4, column=1, padx=5)


        self.normalize_dim_reduction_var_glmm = tk.BooleanVar()
        normalize_checkbox_glmm = tk.Checkbutton(analysis_frame, text="Normalize by Total Daily Activity", variable=self.normalize_dim_reduction_var_glmm)
        # Changed from pack to grid, using row 5, col 0, spanning 2 columns
        normalize_checkbox_glmm.grid(row=5, column=0, padx=5, columnspan=2, sticky="w")

        # Run Analysis Button
        run_analysis_button = tk.Button(analysis_frame, text="Run GLMM Analysis", command=self.run_glmm_analysis)
        # Updated row number to 6 since checkbox occupies row 5 now
        run_analysis_button.grid(row=6, column=0, padx=5, pady=5, columnspan=2, sticky="ew")



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
    


############# GLMM functions

    def run_glmm_analysis(self):
        # Retrieve the selected display name from the combobox
        display_name = self.variable_combobox.get()
        
        # Perform reverse lookup to get the internal variable name
        variable_name = None
        for key, value in self.variable_name_mapping.items():
            if value == display_name:
                variable_name = key
                break

        if variable_name is None:
            self.log(f"No internal name found for variable: {display_name}")
            return

        day_interval_size = self.day_interval_size_combobox.get()
        time_interval = int(self.time_interval_size_combobox.get())
        fixed_effects = self.fixed_effects_entry.get() or "Group"
        random_effects = self.random_effects_entry.get() or "Day"

        # Parse effects
        fixed_effects = [eff.strip() for eff in fixed_effects.split(',') if eff.strip()]
        random_effects = [eff.strip() for eff in random_effects.split(',') if eff.strip()]

        # Prepare combined data
        combined_df = self.prepare_combined_data(variable_name)
        # Create day intervals
        day_intervals = self.create_day_intervals(combined_df, day_interval_size)

        # Structure to hold GLMM results
        results = []
        all_plot_data = []

        try:
            # Execute GLMM
            results.extend(self.execute_glmm(combined_df, fixed_effects, random_effects, day_intervals, variable_name, time_interval))
            
            for result in results:
                plot_data_entry = {
                    'Minute': result['minute'],
                    'Day Interval': result['day_interval'],
                    'Start Day': result['start_day'],
                    'End Day': result['end_day'],
                    'Effect': result['effect'],
                    'Level': result['level'],
                    'Coefficient': result['coefficient'],
                    'Z-Score': result['z_score'],
                    'P-Value': result['p_value']
                }
                all_plot_data.append(plot_data_entry)

             # Assign unique filename components
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            effects_summary = '_'.join(eff[:3] for eff in fixed_effects)  # e.g., "Gro" for Group
            base_filename = f"glmm_{variable_name}_{effects_summary}_{timestamp}"


            # Plot results
            if isinstance(day_interval_size, str) and day_interval_size.lower() != "all days":
                start_date_str = self.start_date_combobox.get()
                end_date_str = self.end_date_combobox.get()
                self.plot_heatmaps(results, fixed_effects, start_date_str, end_date_str,base_filename)
            else:
                self.plot_glmm_results(results, fixed_effects, grouped_experiments=self.grouped_experiments,base_filename=base_filename)
            
            # Save plot data and caption
            filename_prefix = f"glmm_{variable_name}"
            self.save_plot_data_csv(all_plot_data, filename_prefix, self.common_plots_dir)

            start_date_str = self.start_date_combobox.get()
            end_date_str = self.end_date_combobox.get()

            caption_text = (
                f"GLMM analysis on {display_name} using fixed effects ({', '.join(fixed_effects)}) "
                f"and random effects ({', '.join(random_effects)}). "
                "Z-scores signify the strength and direction relative to the baseline, "
                "typically the reference category. P-Values indicate the probability that the effect size is due to chance. "
                f"Analyzed across groups: {', '.join(entry.get() for entry in self.group_alias_entries)}, \n"
                f"Categories: {', '.join(entry.get() for entry in self.category_alias_entries)}, \n"
                f"Experiments: {', '.join(alias_entry.get() for alias_entry in self.alias_entries)}. "
                f"Time interval size: {time_interval}, Day interval size: {day_interval_size}. "
                f"Normalization applied: {'Yes' if self.normalize_dim_reduction_var_glmm.get() else 'No'}. "
                f"Data spans from {start_date_str} to {end_date_str}."
            )

            self.save_analysis_caption(caption_text, filename_prefix, self.common_plots_dir)
            
            self.log("GLMM analysis, plotting, and reporting completed.")

        except Exception as e:
            self.log(f"Error during GLMM analysis: {e}")

    def save_plot_data_csv(self, plot_data, filename_prefix, save_dir):
        """Save the plot data to a CSV file."""
        df = pd.DataFrame(plot_data)
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        csv_filename = f"{filename_prefix}_plot_data_{timestamp}.csv"
        csv_filepath = os.path.join(save_dir, csv_filename)

        df.to_csv(csv_filepath, index=False)
        self.log(f"Plot data saved to {csv_filepath}")

    def save_analysis_caption(self, caption, filename_prefix, save_dir):
        """Save the scientific caption to a text file."""
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        text_filename = f"{filename_prefix}_caption_{timestamp}.txt"
        text_filepath = os.path.join(save_dir, text_filename)

        with open(text_filepath, 'w') as file:
            file.write(caption)

        self.log(f"Caption saved to {text_filepath}")



    def aggregate_daily_data(self, variable_name, experiments):
        # Combine and calculate daily average for the given variable across all experiments
        combined_data = []
        
        for exp_data in experiments:
            population_data = exp_data['population_data']
            var_data = population_data[variable_name]

            # Resampling to daily data and calculating the average
            daily_avg = var_data.resample('D').mean()
            combined_data.append(daily_avg)
        
        combined_df = pd.concat(combined_data, axis=1)
        overall_avg = combined_df.mean(axis=1)
        overall_std = combined_df.std(axis=1)
        return overall_avg, overall_std

    def extract_data_records(self, exp_data, variable_name, category_name, group_name):
        """Extracts records for a given experiment and prepares them with necessary metadata."""
        records = []
        
        # Extract variable data from the experiment
        var_data = exp_data['population_data'][variable_name]

        normalize = self.normalize_dim_reduction_var_glmm.get()

        if normalize:
            var_data = self.plot_manager.normalize_by_daily_total(var_data)

        start_date = var_data.index.min().date()  # Reference start date for each experiment

        # Iterate through the time series of the variable
        for timestamp, value in var_data.items():
            # Compute the day index relative to the start date
            day_index = (timestamp.date() - start_date).days + 1

            # Append the data to records with all relevant metadata
            record = {
                'Timestamp': timestamp,
                'Day': f"Day {day_index}",
                'Experiment': exp_data['alias'],  # Consider using an experiment alias or ID
                'Category': category_name,        # Assigned category for the experiment
                'Group': group_name,              # Assigned group for the experiment
                variable_name: value              # Value of the variable at the time point
            }
            records.append(record)
        
        return records

    def prepare_combined_data(self, variable_name):
        combined_records = []
        for category_name, groups in self.grouped_experiments.items():
            for group_name, experiments in groups.items():
                for exp_data in experiments:
                    records = self.extract_data_records(exp_data, variable_name, category_name, group_name)
                    combined_records.extend(records)
        
        combined_df = pd.DataFrame(combined_records)
        combined_df['Date'] = pd.to_datetime(combined_df['Timestamp'].dt.date)
        combined_df.set_index('Timestamp', inplace=True)
        last_day = combined_df['Date'].max()
        return combined_df[combined_df['Date'] < last_day]

    def execute_glmm(self, data, fixed_effects, random_effects, day_intervals, variable_name, minute_interval):
        results = []
        dropna_columns = [variable_name] + fixed_effects
        data = data.dropna(subset=dropna_columns)

        time_intervals = self.create_time_intervals(minute_interval)

        for effect in fixed_effects + random_effects:
            data[effect] = data[effect].astype('category')

        for idx, (start_day, end_day) in enumerate(day_intervals):
            start_day_ts = pd.Timestamp(start_day)
            end_day_ts = pd.Timestamp(end_day) + pd.Timedelta(hours=23, minutes=59)

            day_segment = data[(data.index >= start_day_ts) & (data.index <= end_day_ts)]

            for start_minute, end_minute in time_intervals:
                interval_segment = self.segment_data_by_interval(day_segment, start_minute, end_minute)
                
                if not interval_segment.empty:
                    numeric_cols = interval_segment.select_dtypes(include='number').columns
                    means = interval_segment.groupby(['Day', 'Experiment'])[numeric_cols].mean().reset_index()

                    means['Category'] = interval_segment.groupby(['Day', 'Experiment'])['Category'].first().values
                    means['Group'] = interval_segment.groupby(['Day', 'Experiment'])['Group'].first().values

                    formula = f"{variable_name} ~ {' + '.join(fixed_effects)}"

                    try:
                        model = MixedLM.from_formula(formula, groups=means[random_effects[0]], data=means)
                        fit_result = model.fit(method='lbfgs', maxiter=500, full_output=True)
                        self.log(f"Successfully fitted GLMM for {start_minute}:00 on interval {idx}.")

                        # Dynamic extraction of coefficients for each fixed effect
                        for effect in fixed_effects:
                            # Handle factor levels to extract relevant terms dynamically
                            levels = means[effect].cat.categories
                            for level in levels[1:]:  # Skip the reference level
                                key = f"{effect}[T.{level}]"
                                coeff = fit_result.params.get(key, np.nan)
                                p_value = fit_result.pvalues.get(key, np.nan)
                                std_err = fit_result.bse.get(key, np.nan)
                                z_score = coeff / std_err if std_err != 0 else np.nan
                                
                                if coeff is not np.nan and p_value is not np.nan and z_score is not np.nan:
                                    results.append({
                                        'minute': start_minute,
                                        'day_interval': f"{idx}",
                                        'start_day':start_day_ts,
                                        'end_day':end_day_ts,
                                        'effect': effect,
                                        'level': level,
                                        'coefficient': coeff,
                                        'p_value': p_value,
                                        'z_score': z_score
                                    })
                    except Exception as e:
                        self.log(f"GLMM fitting failed for {start_minute}:00 on interval {idx}: {e}")

        return results

    def create_time_intervals(self, interval_minutes):
        intervals = []
        # Create intervals by incrementing start_minute by the given interval size
        for start_minute in range(0, 24 * 60, interval_minutes):
            end_minute = start_minute + interval_minutes
            # Only add the interval if it ends before or exactly at midnight
            if end_minute <= 24 * 60:  # 24*60 denotes the total minutes in a day
                intervals.append((start_minute, end_minute))
        return intervals
    
    def segment_data_by_interval(self, data, start_minute, end_minute):
        start_time = pd.to_datetime(start_minute, unit='m').time()
        end_time = pd.to_datetime(end_minute, unit='m').time()
        return data.between_time(start_time, end_time)

    def save_glmm_results(self, fit_result, variable, start_day, end_day, start_hour, end_hour):
        filename_prefix = f"glmm_{variable}_{start_day}_{end_day}_{start_hour}_{end_hour}"
        output_path = os.path.join(self.common_save_dir, f"{filename_prefix}_results.csv")
        try:
            summary_frame = fit_result.summary().tables[1]
            summary_frame.to_csv(output_path, index=False)
            self.log(f"GLMM results saved to {output_path}")
        except Exception as e:
            self.log(f"Error saving GLMM results: {e}")

    def create_day_intervals(self, data, day_interval_size):
        """Creates intervals of days from the data based on the given interval size.

        Args:
            data (DataFrame): The combined data containing a DateTime index.
            day_interval_size (str or int): The size of each interval in days, or "All Days" to use the entire range.

        Returns:
            list: A list of tuples, where each tuple contains the start and end of a day interval.
        """

        # Convert day_interval_size to integer if it's not "All Days"
        if isinstance(day_interval_size, str) and day_interval_size.lower() == "all days":
            start_date = data.index.min().date()
            end_date = data.index.max().date()
            return [(start_date, end_date)]

        interval_size = int(day_interval_size)

        # Get the unique days present in the data's index
        unique_days = pd.date_range(start=data.index.min().date(), end=data.index.max().date(), freq='D')

        # Implement sliding window
        intervals = []
        for i in range(0, len(unique_days) - interval_size + 1):
            start_day = unique_days[i]
            end_day = unique_days[i + interval_size - 1]
            intervals.append((start_day, end_day))

        print(intervals)

        return intervals
    

    def plot_glmm_results(self, results, fixed_effects, grouped_experiments,base_filename):
        # Retrieve the selected display name from the combobox
        display_name = self.variable_combobox.get()

        # Perform reverse lookup to get the internal variable name
        internal_var_name = None
        for key, value in self.variable_name_mapping.items():
            if value == display_name:
                internal_var_name = key
                break
        
        if internal_var_name is None:
            self.log(f"No internal name found for variable: {display_name}")
            return
        
        # Retrieve start date from user input
        try:
            start_date_str = self.end_date_combobox.get()
            start_date = pd.to_datetime(start_date_str) - timedelta(days=1)  # Convert the start date string to a datetime object
        except Exception as e:
            self.log(f"Error parsing start date: {e}")
            return

        # Create a figure with 4 vertically aligned subplots sharing the same x-axis
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(5, 15),sharex=True)

    # Extract information for GLMM plots
        minutes = [res['minute'] for res in results]
        
        # Convert minutes to datetime objects starting from the user-specified start date
        datetime_values = [start_date + timedelta(minutes=min) for min in minutes]

        coefficients = [res['coefficient'] for res in results]
        z_scores = [res['z_score'] for res in results]
        p_values = [res['p_value'] for res in results]

        # Plot Coefficients
        axes[0].plot(datetime_values , coefficients, marker='o', linestyle='-')
        axes[0].set_ylabel('Coefficient Value')

        # Plot Z-Scores
        axes[1].plot(datetime_values , z_scores, marker='o', linestyle='-')
        axes[1].set_ylabel('Z-Score')

        # Plot P-Values in -log10 scale
        log_p_values = [-np.log10(p) if p > 0 else np.nan for p in p_values]
        axes[2].plot(datetime_values , log_p_values, marker='o', linestyle='-')
        axes[2].set_ylabel('-Log10(P-Value)')
        axes[2].axhline(-np.log10(0.01), color='r', linestyle='--', label='p = 0.01')
        axes[2].legend()

        # Configure the x-axis to show labels on the last subplot to avoid overlap
        #axes[3].set_xticks(hours)
        #axes[3].set_xticklabels(time_labels, rotation=45)  # Rotate labels for better readability

        # Configure the daily average plot in the 4th subplot
        plot_args = {
            'selected_vars': [internal_var_name],
            'resample_interval': self.resample_entry.get(),
            'moving_avg_window': self.moving_avg_entry.get(),
            'start_date_str': self.start_date_combobox.get(),
            'end_date_str': self.end_date_combobox.get(),
            'threshold': self.threshold_entry.get(),
            'experiments_to_plot': [],
            'groups_to_plot': [],
            'categories_to_plot': list(grouped_experiments.keys()) if 'Category' in fixed_effects else [],
            'grouped_experiments': grouped_experiments,
            'start_hour': self.start_hour_scale.get(),
            'end_hour': self.end_hour_scale.get(),
            'ax': axes[3],
            'normalize': self.normalize_dim_reduction_var_glmm.get(),
            'display_names': [display_name]
        }
        
        # Add groups if "Group" is in fixed_effects
        if 'Group' in fixed_effects:
            plot_args['groups_to_plot'] = [grp for category in grouped_experiments.values() for grp in category.keys()]

        self.plot_manager.plot_daily_average(**plot_args)

        # Adding titles and labels for daily averages subplot
        axes[3].set_title('Daily Average of Selected Variables')
        axes[3].set_xlabel('Hour of the Day')
        axes[3].set_ylabel('Average Value')
        axes[3].set_title(None)
        axes[3].legend(fontsize=5)  # Example of specifying a numerical font size

        
        plt.tight_layout()


        filename = base_filename + "_results"
        self.plot_manager.save_plot(fig, filename)
        plt.show()
        plt.close(fig)

    def plot_heatmaps(self, results, fixed_effects, start_date_str, end_date_str,base_filename):
        try:
            # Convert results to DataFrame
            results_df = pd.DataFrame(results)

            # Aggregate by taking the mean of duplicates
            results_df = results_df.groupby(['minute', 'effect', 'level', 'day_interval']).mean().reset_index()

            # Convert day_interval to integers for proper sorting
            results_df['day_interval'] = results_df['day_interval'].astype(int)
            results_df = results_df.sort_values('day_interval')

            # Convert minutes to hour format
            results_df['hour'] = results_df['minute'].apply(lambda x: f"{x // 60:02}:{x % 60:02}")

            # For each fixed effect, create and plot heatmaps
            for effect in fixed_effects:
                # Filter results for this effect
                effect_results = results_df[results_df['effect'] == effect]

                # Calculate median day for interval labels
                median_days = effect_results.groupby('day_interval')['day_interval'].median()
                median_day_labels = {int(idx): f" {int(day+1)}" for idx, day in median_days.items()}

                # Create pivot tables for z-scores and p-values
                z_value_heatmap_data = effect_results.pivot(index='day_interval', columns='hour', values='z_score')
                neg_log_p_heatmap_data = effect_results.pivot(index='day_interval', columns='hour', values='p_value').applymap(lambda p: -np.log10(p) if p > 0 else np.nan)

                # Custom colormap
                colors = [(0, "blue"), (2/6, "white"), (1, "red")]
                cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

                # Normalize p-values using log scales
                norm = mcolors.Normalize(vmin=0, vmax=6)

                # Plotting
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

                # Z-Score Heatmap
                sns.heatmap(z_value_heatmap_data, annot=False, cmap="viridis",
                            cbar_kws={'label': 'Z Value'}, ax=axes[0])
                axes[0].set_title(f'Z-Value for Factor: {effect}', fontsize=14)
                axes[0].set_xlabel("Time Interval (hour:minute)", fontsize=12)
                axes[0].set_ylabel("Median Day", fontsize=12)
                axes[0].set_yticklabels([median_day_labels.get(int(label.get_text()), '') for label in axes[0].get_yticklabels()])

                # -log10(P-Value) Heatmap with custom colormap
                sns.heatmap(neg_log_p_heatmap_data, annot=False, cmap=cmap,
                            cbar_kws={'label': '-log10(p-value)'}, ax=axes[1], norm=norm)
                axes[1].set_title(f'-log10(p-value) for Factor: {effect}', fontsize=14)
                axes[1].set_xlabel("Time Interval (hour:minute)", fontsize=12)
                axes[1].set_ylabel("Median Day", fontsize=12)
                axes[1].set_yticklabels([median_day_labels.get(int(label.get_text()), '') for label in axes[1].get_yticklabels()])

                plt.tight_layout()

                # Save or display the plots
                filename = base_filename + f"_heatmap_{effect}"
                self.plot_manager.save_plot(fig, filename)
                plt.show()
                plt.close(fig)

        except Exception as e:
            self.log(f"Error during plotting heatmaps: {e}")





    # def plot_glmm_results(self, results):
    #     minutes = [res['minute'] for res in results]
    #     coefficients = [res['coefficient'] for res in results]
    #     p_values = [res['p_value'] for res in results]

    #     # Convert p-values to -log10 scale
    #     log_p_values = [-np.log10(p) if p > 0 else np.nan for p in p_values]

    #     fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    #     ax[0].plot(minutes, coefficients, marker='o', linestyle=None)
    #     ax[0].set_title('GLMM Coefficients for Category by Time of Day')
    #     ax[0].set_ylabel('Coefficient Value')

    #     ax[1].plot(minutes, log_p_values, marker='o', linestyle=None)
    #     ax[1].set_title('GLMM P-Values in Log Scale for Category by Time of Day')
    #     ax[1].set_ylabel('-Log10(P-Value)')
    #     ax[1].set_xlabel('Time of Day')

    #     # Convert minute ticks to "Time of Day"
    #     def minutes_to_time(minute):
    #         return (datetime.min + timedelta(minutes=minute)).time().strftime('%H:%M')

    #     # Convert minutes into "HH:MM" format for x-ticks
    #     time_ticks = [minutes_to_time(minute) for minute in minutes]
    #     ax[1].set_xticks(minutes)
    #     ax[1].set_xticklabels(time_ticks, rotation=45)  # Rotate for better readability

    #     ax[1].axhline(-np.log10(0.01), color='r', linestyle='--', label='p = 0.01')  # Optional: mark significance threshold
    #     ax[1].legend()

    #     plt.tight_layout()
    #     plt.show()









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