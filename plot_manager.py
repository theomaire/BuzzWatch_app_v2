from common_imports import *
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from scipy.stats import sem
import matplotlib.dates as mdates
from PIL import Image
import matplotlib.lines as mlines
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap




class PlotManager:
    def __init__(self, log, save_dir="plots",group_colors=None, group_markers=None, category_colors=None, category_markers=None):
        self.log = log
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.set_default_plot_properties()
        self.errorbar_props = {
            'markersize': 10,
            'capsize': 4,
            'fmt': '.',
            'linestyle': '--',
        }
                # Store color and marker information
        self.group_colors = group_colors or {}
        self.group_markers = group_markers or {}
        self.category_colors = category_colors or {}
        self.category_markers = category_markers or {}

    def set_plot_size(self, BIGGER_SIZE):
        plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=BIGGER_SIZE)     # fontsize of the x and y labels
        plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)   # figure title fontsize

    def set_default_plot_properties(self):
        self.set_plot_size(10)
        mpl.rcParams['axes.linewidth'] = 1
        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        mpl.rcParams['font.family'] = 'Arial'
        plt.rcParams.update({
            'font.family': 'Arial',
            'lines.linewidth': 1,
        })

    def initialize_figure(self, subplots_shape=(1, 1), figsize=(8, 6), dpi=100, tight_layout=True):
        """Initialize a matplotlib figure and axes with consistent styling."""
        fig, axs = plt.subplots(*subplots_shape, figsize=figsize, dpi=dpi)
        if tight_layout:
            plt.tight_layout()
        return fig, axs.flatten()


        


    def save_plot(self, fig, filename):
        png_path = os.path.join(self.save_dir, f"{filename}.png")
        pdf_path = os.path.join(self.save_dir, f"{filename}.pdf")
        fig.savefig(png_path, format='png', bbox_inches='tight')
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight',dpi=300,transparent=False)
        self.log(f"Plots saved as {png_path} and {pdf_path}")


        # Assuming the Zeitgeber time conversion starts at 08:00 as ZT0
    def convert_to_zeitgeber_time(self,index):
        # Shift time to match ZT0 at 08:00, wrapping around correctly
        #zt_index = index.shift(-8, 'H')
        zt_values = (index.hour + index.minute / 60) % 24 - 8 - 20/60 # With 20 min correction (first video is 01, not 00)
        
        # Map ZT to the desired range [-8, 15] for plotting
        #zt_values = [(v - 16 if v >= 16 else v) for v in zt_values]
        return zt_values

    def plot_entire_time_series(self, selected_vars, resample_interval, moving_avg_window, start_date_str, end_date_str, threshold, experiments_to_plot, groups_to_plot, categories_to_plot, grouped_experiments, start_hour=None, end_hour=None, normalize=False, ax=None, display_names=None):
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
        
        # Calculate number of days and adjust figure properties
        num_days = (end_date - start_date).days
        
        # Scale font size based on number of days
        font_size = max(6, min(10, 14 - (num_days / 30)))  # Decrease font size as days increase, min 6pt
        plt.rcParams.update({
            'font.size': font_size,
            'axes.labelsize': font_size,
            'axes.titlesize': font_size + 2,
            'xtick.labelsize': font_size - 1,
            'ytick.labelsize': font_size - 1,
            'legend.fontsize': font_size - 1
        })
        
        # Adjust line width based on number of days
        line_width = max(0.5, min(1.5, 2 - (num_days / 30)))  # Thinner lines for longer periods
        plt.rcParams['lines.linewidth'] = line_width


        line_styles = ['-', '--', '-.', ':']

        vars_part = "_".join(selected_vars)
        exp_names = []
        group_names = []
        category_names = []

        # Plot individual experiments
        for exp_data in experiments_to_plot:
            alias = exp_data['alias']
            exp_names.append(alias)
            
            population_data = exp_data['population_data']
            individual_data = exp_data['individual_data']

            for i, (var, display_name) in enumerate(zip(selected_vars, display_names)):
                var_data = self.get_var_data(var, population_data, individual_data, threshold)
                var_data = self.crop_time(start_date, end_date, var_data)

                if resample_interval:
                    var_data = var_data.resample(resample_interval).mean()

                if moving_avg_window:
                    moving_avg_window = int(moving_avg_window)
                    var_data = var_data.rolling(window=moving_avg_window).mean()

                if normalize:
                    var_data = self.normalize_by_daily_total(var_data)


                style = line_styles[i % len(line_styles)]
                var_data.plot(ax=ax, label=f"{alias} - {display_name}", linestyle=style)

        # Plot averaged groups
        for category_name, groups in grouped_experiments.items():
            for group_name in groups_to_plot:
                if group_name in groups:
                    group_names.append(group_name)

                    # Use self.group_colors and self.group_markers
                    color = self.group_colors.get(group_name, 'tab:blue')
                    marker = self.group_markers.get(group_name, 'o')
                    
                    group_data = {var: [] for var in selected_vars}

                    for exp_data in groups[group_name]:
                        alias = exp_data['alias']
                        population_data = exp_data['population_data']
                        individual_data = exp_data['individual_data']

                        for var in selected_vars:
                            var_data = self.get_var_data(var, population_data, individual_data, threshold)
                            var_data = self.crop_time(start_date, end_date, var_data)

                            if resample_interval:
                                var_data = var_data.resample(resample_interval).mean()

                            if moving_avg_window:
                                moving_avg_window = int(moving_avg_window)
                                var_data = var_data.rolling(window=moving_avg_window).mean()

                            if normalize:
                                var_data = self.normalize_by_daily_total(var_data)
                            
                            var_data = var_data[~var_data.index.duplicated(keep='first')]
                            group_data[var].append(var_data)

                    for i, (var, display_name) in enumerate(zip(selected_vars, display_names)):
                        unified_index = pd.date_range(start=start_date, end=end_date, freq=resample_interval or 'T')

                        all_series = [
                            var_data.reindex(unified_index, fill_value=float('nan')).rename(f"{var}_exp_{j}")
                            for j, var_data in enumerate(group_data[var])
                        ]
                        
                        master_df = pd.concat(all_series, axis=1)
                        avg_var_data = master_df.mean(axis=1)

                        style = line_styles[i % len(line_styles)]
                        avg_var_data.plot(ax=ax, label=f"{group_name} - {display_name}", linestyle=style,color=color)

        # Plot averaged categories
        unified_index = pd.date_range(start=start_date, end=end_date, freq=resample_interval or 'T')
        for category_name in categories_to_plot:
            self.log(f"Processing category: {category_name}")
            category_names.append(category_name)
            category_data = {var: [] for var in selected_vars}

            color = self.category_colors.get(category_name, 'tab:blue')
            marker = self.category_markers.get(category_name, 'o')


            for group_name, group in grouped_experiments[category_name].items():
                group_data = {var: [] for var in selected_vars}

                for exp_data in group:
                    alias = exp_data['alias']
                    population_data = exp_data['population_data']
                    individual_data = exp_data['individual_data']

                    for var in selected_vars:
                        var_data = self.get_var_data(var, population_data, individual_data, threshold)
                        var_data = self.crop_time(start_date, end_date, var_data)

                        if resample_interval:
                            var_data = var_data.resample(resample_interval).mean()

                        if moving_avg_window:
                            moving_avg_window = int(moving_avg_window)
                            var_data = var_data.rolling(window=moving_avg_window).mean()

                        if normalize:
                            var_data = self.normalize_by_daily_total(var_data)

                        var_data = var_data[~var_data.index.duplicated(keep='first')]
                        group_data[var].append(var_data)
                
                all_series = [
                    var_data.reindex(unified_index, fill_value=float('nan')).rename(f"{var}_group_{j}")
                    for j, var_data in enumerate(group_data[var])
                ]

                master_df = pd.concat(all_series, axis=1)
                avg_group_data = master_df.mean(axis=1)  # Average over groups in category
                category_data[var].append(avg_group_data)

            for i, (var, display_name) in enumerate(zip(selected_vars, display_names)):
                final_series = pd.concat(category_data[var], axis=1).mean(axis=1)  # Final average over category
                style = line_styles[i % len(line_styles)]
                final_series.plot(ax=ax, label=f"{category_name} - {display_name}", linestyle=style, lw=1,color=color)

        ax.set_title("Time series")
        ax.set_xlabel("Days")
            # Ensure x-axis label is visible
        #ax.xaxis.set_label_coords(0.5, -0.1)
    
        # Add some padding to prevent label cutoff

        #ax.margins(x=0.2, y=0.2)
        ax.set_ylabel(f"{display_names[0]}" if len(display_names) == 1 else "Values")
        ax.legend()
        
        # Join experiment, group, and category names
        exp_part = "_and_".join(exp_names) if exp_names else ""
        group_part = "_and_".join(group_names) if group_names else ""
        category_part = "_and_".join(category_names) if category_names else ""
        exp_group_cat_part = "_and_".join(filter(None, [exp_part, group_part, category_part]))
        
        #filename = f"entire_time_series_{vars_part}_{exp_group_cat_part}_{start_date_str}_to_{end_date_str}".replace(" ", "_")
        
        # self.save_plot(fig, filename)
        # plt.show()
        # plt.close(fig)


        
        # Ensure proper spacing
        #fig.subplots_adjust(left=0.12, right=0.95, bottom=0.15, top=0.95)
        plt.tight_layout()
    
    def plot_daily_average(self, selected_vars, resample_interval, moving_avg_window, start_date_str, end_date_str, threshold, 
                        experiments_to_plot, groups_to_plot, categories_to_plot, grouped_experiments, start_hour=None, 
                        end_hour=None, normalize=False, ax=None, display_names=None, use_zeitgeber_time=False,ylim=None):
        """
        Plot the daily average of variables.

        :param use_zeitgeber_time: If True, plots the time axis in Zeitgeber Time (ZT).
        """
        try:
            start_date = pd.to_datetime(start_date_str)
            end_date = pd.to_datetime(end_date_str)

            line_styles = ['-', '--', '-.', ':']

            exp_legend_added = False
            group_legend_added = False
            category_legend_added = False

            exp_names = []
            group_names = []
            category_names = []

            # Plot individual experiments
            for exp_data in experiments_to_plot:
                alias = exp_data['alias']
                exp_names.append(alias)

                population_data = exp_data['population_data']
                individual_data = exp_data['individual_data']

                for i, (var, display_name) in enumerate(zip(selected_vars, display_names)):
                    var_data = self.get_var_data(var, population_data, individual_data, threshold)

                    # Continue if the data is empty
                    if var_data.empty:
                        self.log(f"[DEBUG] Skipped {alias} - {var}: No data.")
                        continue

                    if resample_interval:
                        var_data = var_data.resample(resample_interval).mean()

                    if moving_avg_window:
                        var_data = var_data.rolling(window=int(moving_avg_window)).mean()

                    if normalize:
                        var_data = self.normalize_by_daily_total(var_data)

                    day_avg, day_std = self.calculate_daily_avg(var_data, start_date, end_date, resample_interval, moving_avg_window)
                    
                    # Continue if the processed data is empty
                    if day_avg.empty:
                        self.log(f"[DEBUG] Skipped {alias} - {var}: No data after processing.")
                        continue

                    style = line_styles[i % len(line_styles)]

                    x_values = self.convert_to_zeitgeber_time(day_avg.index) if use_zeitgeber_time else day_avg.index

                    ax.plot(x_values, day_avg, label=f"{alias} - {display_name}", linestyle=style)

                    if not exp_legend_added:
                        ax.fill_between(x_values, day_avg - day_std, day_avg + day_std, alpha=0.3, label="Experiment: Avg ± Std (Days)")
                        exp_legend_added = True
                    else:
                        ax.fill_between(x_values, day_avg - day_std, day_avg + day_std, alpha=0.3)

            # Plot averaged groups
            for category_name, groups in grouped_experiments.items():
                for group_name in groups_to_plot:
                    if group_name in groups:
                        group_names.append(group_name)

                        group_data = {var: [] for var in selected_vars}

                        color = self.group_colors.get(group_name, 'tab:blue')
                        marker = self.group_markers.get(group_name, 'o')

                        for exp_data in groups[group_name]:
                            alias = exp_data['alias']
                            population_data = exp_data['population_data']
                            individual_data = exp_data['individual_data']

                            for var in selected_vars:
                                var_data = self.get_var_data(var, population_data, individual_data, threshold)
                                if var_data.empty:
                                    self.log(f"[DEBUG] Skipped group {group_name} - {alias} - {var}: No data.")
                                    continue

                                if resample_interval:
                                    var_data = var_data.resample(resample_interval).mean()

                                if moving_avg_window:
                                    var_data = var_data.rolling(window=int(moving_avg_window)).mean()

                                if normalize:
                                    var_data = self.normalize_by_daily_total(var_data)
                                var_data = var_data[~var_data.index.duplicated(keep='first')]

                                day_avg, day_std = self.calculate_daily_avg(var_data, start_date, end_date, resample_interval,
                                                                            moving_avg_window)
                                group_data[var].append(day_avg)

                        for i, (var, display_name) in enumerate(zip(selected_vars, display_names)):
                            if not group_data[var]:
                                continue
                            daily_avg_df = pd.concat(group_data[var], axis=1)

                            group_daily_avg = daily_avg_df.mean(axis=1)
                            group_daily_std = daily_avg_df.std(axis=1)

                            style = line_styles[i % len(line_styles)]

                            x_values = self.convert_to_zeitgeber_time(group_daily_avg.index) if use_zeitgeber_time else group_daily_avg.index

                            ax.plot(x_values, group_daily_avg, label=f"{group_name} - {display_name}", linestyle=style, color=color)

                            if not group_legend_added:
                                ax.fill_between(x_values, group_daily_avg - group_daily_std, group_daily_avg + group_daily_std, alpha=0.3,
                                                color=color, label="Group: Avg ± Std (Experiments)")
                                group_legend_added = True
                            else:
                                ax.fill_between(x_values, group_daily_avg - group_daily_std, group_daily_avg + group_daily_std, alpha=0.3,
                                                color=color)

            unified_index = pd.date_range(start=start_date, end=end_date, freq=resample_interval or 'T')
            for category_name in categories_to_plot:
                self.log(f"Processing category: {category_name}")
                category_names.append(category_name)
                category_data = {var: [] for var in selected_vars}

                color = self.category_colors.get(category_name, 'tab:blue')
                marker = self.category_markers.get(category_name, 'o')

                for group_name, group in grouped_experiments[category_name].items():
                    group_data = {var: [] for var in selected_vars}

                    for exp_data in group:
                        alias = exp_data['alias']
                        population_data = exp_data['population_data']
                        individual_data = exp_data['individual_data']

                        for var in selected_vars:
                            var_data = self.get_var_data(var, population_data, individual_data, threshold)
                            if not var_data.empty:
                                if resample_interval:
                                    var_data = var_data.resample(resample_interval).mean()

                                if moving_avg_window:
                                    moving_avg_window = int(moving_avg_window)
                                    var_data = var_data.rolling(window=moving_avg_window).mean()

                                if normalize:
                                    var_data = self.normalize_by_daily_total(var_data)

                                var_data = var_data[~var_data.index.duplicated(keep='first')]

                                day_avg, _ = self.calculate_daily_avg(var_data, start_date, end_date, resample_interval, moving_avg_window)
                                group_data[var].append(day_avg)

                    if group_data[var]:
                        group_avg_df = pd.concat(group_data[var], axis=1)
                        category_data[var].append(group_avg_df.mean(axis=1))

                for i, (var, display_name) in enumerate(zip(selected_vars, display_names)):
                    if not category_data[var]:
                        continue

                    category_avg_df = pd.concat(category_data[var], axis=1)

                    category_daily_avg = category_avg_df.mean(axis=1)
                    category_daily_std = category_avg_df.std(axis=1)

                    style = line_styles[i % len(line_styles)]
                    
                    x_values = self.convert_to_zeitgeber_time(category_daily_avg.index) if use_zeitgeber_time else category_daily_avg.index

                    ax.plot(x_values, category_daily_avg, label=f"{category_name} - {display_name}", linestyle=style, color=color)

                    if not category_legend_added:
                        ax.fill_between(x_values, category_daily_avg - category_daily_std, category_daily_avg + category_daily_std,
                                        alpha=0.3, color=color, label="Category: Avg ± Std (Groups)")
                        category_legend_added = True
                    else:
                        ax.fill_between(x_values, category_daily_avg - category_daily_std, category_daily_avg + category_daily_std,
                                        alpha=0.3, color=color)

            # if use_zeitgeber_time:
            #     # Set limits for x-axis to maintain consistency in display 
            #     ax.set_xlim(-8, 15)
            #     self.add_light_intensity_bar(ax)
            #     # Define ZT values ranging from the specified -8 to 15
            #     zt_ticks = list(range(-8, 16))
            #     zt_labels = [f"{(t)}" for t in zt_ticks]  # Correctly wrap-around to [16, ..., 23, 0, ..., 15]

            #     # Set the tick locations and labels
            #     ax.set_xticks(zt_ticks)
            #     ax.set_xticklabels(zt_labels)

                

            # Add the light intensity bar at this point
            print(ylim)
            if ylim == '':
                pass
            else:
                ax.set_ylim([0,np.float64(ylim)])
            ax.set_title("Av. daily rhythm")
            #ax.set_xlabel("Zeitgeber Time" if use_zeitgeber_time else "Hour of the Day")
            ax.set_ylabel(f"{display_names[0]}" if len(display_names) == 1 else "Values")
            #plt.xticks(rotation=45)
            plt.tight_layout()
            ax.legend(fontsize='small')
        except Exception as e:
            print(f"An error occurred: {e}")

    def plot_avg_over_days(self, selected_vars, resample_interval, moving_avg_window, start_date_str, end_date_str, threshold, experiments_to_plot, groups_to_plot, categories_to_plot, grouped_experiments, start_hour=None, end_hour=None, normalize=False, ax=None, display_names=None):
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)

        # Define line styles
        line_styles = ['-', '--', '-.', ':']

        exp_names = []
        group_names = []
        category_names = []

        # Plot individual experiments
        for exp_data in experiments_to_plot:
            alias = exp_data['alias']
            exp_names.append(alias)

            population_data = exp_data['population_data']
            individual_data = exp_data['individual_data']

            color = self.exp_colors.get(alias, 'tab:red')

            for i, (var, display_name) in enumerate(zip(selected_vars, display_names)):
                var_data = self.get_var_data(var, population_data, individual_data, threshold)

                if normalize:
                    var_data = self.normalize_by_daily_total(var_data)
                var_data = self.crop_time(start_date, end_date, var_data).between_time(f'{start_hour}:00', f'{end_hour}:00')

                daily_means = var_data.resample('D').mean()
                daily_std = var_data.resample('D').std()
                days_since_start = (daily_means.index - daily_means.index[0]).days + 1

                style = line_styles[i % len(line_styles)]
                ax.errorbar(days_since_start, daily_means, yerr=daily_std, label=f"{alias} - {display_name}", color=color, **self.errorbar_props)

        # Plot averaged groups
        for category_name, groups in grouped_experiments.items():
            for group_name in groups_to_plot:
                if group_name in groups:
                    group_names.append(group_name)

                    group_data = {var: [] for var in selected_vars}

                    color = self.group_colors.get(group_name, 'tab:blue')

                    for exp_data in groups[group_name]:
                        population_data = exp_data['population_data']
                        individual_data = exp_data['individual_data']

                        for var in selected_vars:
                            var_data = self.get_var_data(var, population_data, individual_data, threshold)

                            if normalize:
                                var_data = self.normalize_by_daily_total(var_data)
                            var_data = self.crop_time(start_date, end_date, var_data).between_time(f'{start_hour}:00', f'{end_hour}:00')

                            group_data[var].append(var_data.resample('D').mean())

                    for i, (var, display_name) in enumerate(zip(selected_vars, display_names)):
                        if not group_data[var]:
                            continue

                        daily_avg_df = pd.concat(group_data[var], axis=1)
                        group_daily_avg = daily_avg_df.mean(axis=1)
                        group_daily_std = daily_avg_df.std(axis=1)
                        days_since_start = (group_daily_avg.index - group_daily_avg.index[0]).days + 1

                        style = line_styles[i % len(line_styles)]
                        ax.errorbar(days_since_start, group_daily_avg, yerr=group_daily_std, label=f"{group_name} - {display_name}", color=color, **self.errorbar_props)

        # Plot averaged categories
        for category_name in categories_to_plot:
            self.log(f"Processing category: {category_name}")
            category_names.append(category_name)

            category_data = {var: [] for var in selected_vars}

            color = self.category_colors.get(category_name, 'tab:green')

            for group_name, group in grouped_experiments[category_name].items():
                group_data = {var: [] for var in selected_vars}

                for exp_data in group:
                    population_data = exp_data['population_data']
                    individual_data = exp_data['individual_data']

                    for var in selected_vars:
                        var_data = self.get_var_data(var, population_data, individual_data, threshold)

                        if normalize:
                            var_data = self.normalize_by_daily_total(var_data)
                        var_data = self.crop_time(start_date, end_date, var_data).between_time(f'{start_hour}:00', f'{end_hour}:00')

                        group_data[var].append(var_data.resample('D').mean())

                if group_data[var]:
                    avg_group_data = pd.concat(group_data[var], axis=1).mean(axis=1)
                    category_data[var].append(avg_group_data)

            for i, (var, display_name) in enumerate(zip(selected_vars, display_names)):
                if not category_data[var]:
                    continue

                category_avg_df = pd.concat(category_data[var], axis=1)
                category_daily_avg = category_avg_df.mean(axis=1)
                category_daily_std = category_avg_df.std(axis=1)
                days_since_start = (category_daily_avg.index - category_daily_avg.index[0]).days + 1

                style = line_styles[i % len(line_styles)]
                ax.errorbar(days_since_start, category_daily_avg, yerr=category_daily_std, label=f"{category_name} - {display_name}", color=color, **self.errorbar_props)

        ax.set_title(f"Average Over Days (from {start_hour}:00 to {end_hour}:00 each day)")
        ax.set_xlabel("Days Since Start")
        ax.set_ylabel(f"{display_names[0]}" if len(display_names) == 1 else "Values")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend()
        plt.tight_layout()
    
    def calculate_daily_avg(self, df, start_date, end_date, resample_interval, moving_avg_window):
        nb_days = (end_date - start_date).days

        all_days = []
        for day in range(nb_days):
            day_start = start_date + pd.Timedelta(days=day)
            day_end = day_start + pd.Timedelta(days=1)
            day_data = self.crop_time(day_start, day_end, df)


            all_days.append(day_data.values)

        df_all = pd.DataFrame(all_days).T
        df_all.index = day_data.index
        avg = df_all.mean(axis=1)
        std = df_all.std(axis=1)
        return avg, std

    def crop_time(self, t_i, t_f, df):
        mask = (df.index >= t_i) & (df.index < t_f)
        df = df.loc[mask]
        return df

    def get_var_data(self, var, population_data, individual_data, threshold):
        if var in population_data.columns:
            return population_data[var]
        elif var in individual_data.columns:
            var_data = individual_data[var]
            if var in ["flight_duration", "average_speed"] and threshold:
                threshold_value = float(threshold)
                var_data = var_data[var_data > threshold_value]
            return var_data
        else:
            self.log(f"Variable '{var}' not found in experiment data.")
            return pd.Series()
        

###################### Scatter bar plot variabiltity ###################
    def plot_scatter_variability(self, selected_vars, resample_interval, moving_avg_window, start_date_str, end_date_str, threshold, experiments_to_plot, groups_to_plot, categories_to_plot, grouped_experiments, start_hour=None, end_hour=None, ax=None, display_names=None, normalize=False):
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)

        # Create the plot axis if none is provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Centralized mapping for plotting styles
        plot_styles = {
            'errorbar': {
                'fmt': 'o',  # Marker style
                'markersize': 4.5,  # Marker size
                'capsize': 5,  # Cap size for error bars
            },
            'scatter': {
                's': 20,  # Marker size for scatter (equivalent to markersize^2 for 'o')
                'alpha': 0.5,  # Transparency level
                'marker':'o', # Marker style
                'edgecolors':'none',
            }
        }

        for i, (var, display_name) in enumerate(zip(selected_vars, display_names)):
            try:
                # Plot individual experiments
                for exp_data in experiments_to_plot:
                    alias = exp_data['alias']
                    population_data = exp_data['population_data']
                    individual_data = exp_data['individual_data']

                    # Fetch color for the current experiment
                    color = self.exp_colors.get(alias, 'tab:blue')

                    var_data = self.get_var_data(var, population_data, individual_data, threshold)
                    var_data = self.crop_time(start_date, end_date, var_data)

                    if normalize:
                        var_data = self.normalize_by_daily_total(var_data)
                    var_data = var_data.between_time(f'{start_hour:02d}:00', f'{end_hour:02d}:00')

                    daily_means = var_data.resample('D').mean()
                    avg_all_days = daily_means.mean()
                    std_all_days = daily_means.std()

                    ax.errorbar(alias, avg_all_days, yerr=std_all_days, label=f"{alias} - {display_name}", color=color, **plot_styles['errorbar'])
                    ax.scatter([alias] * len(daily_means), daily_means, color=color, **plot_styles['scatter'])

                # Plot averaged groups
                for category_name, groups in sorted(grouped_experiments.items()):
                    for group_name in sorted(groups_to_plot):
                        if group_name in groups:
                            group_data = {var: [] for var in selected_vars}

                            # Fetch color for the current group
                            color = self.group_colors.get(group_name, 'tab:green')

                            for exp_data in groups[group_name]:
                                population_data = exp_data['population_data']
                                individual_data = exp_data['individual_data']

                                var_data = self.get_var_data(var, population_data, individual_data, threshold)

                                if normalize:
                                    var_data = self.normalize_by_daily_total(var_data)

                                var_data = self.crop_time(start_date, end_date, var_data).between_time(f'{start_hour:02d}:00', f'{end_hour:02d}:00')

                                group_data[var].append(var_data.resample('D').mean())

                            if group_data[var]:
                                daily_avg_df = pd.concat(group_data[var], axis=1)
                                group_daily_avg = daily_avg_df.mean(axis=1)
                                group_daily_std = daily_avg_df.std(axis=1)

                                ax.errorbar(group_name, group_daily_avg.mean(), yerr=group_daily_std.mean(), color=color, **plot_styles['errorbar'])
                                ax.scatter([group_name] * len(group_daily_avg), group_daily_avg, color=color, **plot_styles['scatter'])

                # Plot categories
                for category_name in sorted(categories_to_plot):
                    category_data = {var: [] for var in selected_vars}

                    # Fetch color for the current category
                    color = self.category_colors.get(category_name, 'tab:red')

                    for group_name, group in grouped_experiments[category_name].items():
                        group_data = {var: [] for var in selected_vars}

                        for exp_data in group:
                            alias = exp_data['alias']
                            population_data = exp_data['population_data']
                            individual_data = exp_data['individual_data']

                            var_data = self.get_var_data(var, population_data, individual_data, threshold)
                            var_data = self.crop_time(start_date, end_date, var_data)

                            if normalize:
                                var_data = self.normalize_by_daily_total(var_data)
                            var_data = var_data.between_time(f'{start_hour:02d}:00', f'{end_hour:02d}:00')

                            group_data[var].append(var_data.resample('D').mean())

                        if group_data[var]:
                            group_avg_df = pd.concat(group_data[var], axis=1)
                            category_data[var].append(group_avg_df.mean(axis=1))

                    if category_data[var]:
                        category_avg_all_days = pd.concat(category_data[var], axis=1).mean(axis=1)
                        avg_all_days = category_avg_all_days.mean()
                        std_all_days = category_avg_all_days.std()

                        ax.errorbar(category_name, avg_all_days, yerr=std_all_days, color=color, **plot_styles['errorbar'])
                        ax.scatter([category_name] * len(category_avg_all_days), category_avg_all_days, color=color, **plot_styles['scatter'])

            except Exception as e:
                self.log(f"An error occurred during plotting: {e}")
                continue

        ax.set_title(f"{display_name} from {start_hour}:00 to {end_hour}:00")
        ax.legend()
        ax.set_xticklabels(ax.get_xticks(), rotation=45, ha='right')

        #ax.set_xlabel("Experiment/Group/Category")
        ax.set_ylabel(f"{display_name}")





####################### 
    def calculate_median_time_interval(self, time_interval):
        try:
            start_hour, end_hour = time_interval.split('-')
            start_hour = int(start_hour.split(':')[0])
            end_hour = int(end_hour.split(':')[0])
            if end_hour < start_hour:
                end_hour += 24
            median_hour = (start_hour + end_hour) / 2
            median_hour = median_hour % 24  # Wrap around if it exceeds 24 hours
            return median_hour
        except Exception as e:
            self.log(f"Error in calculate_median_time_interval for {time_interval}: {e}")
            return np.nan

    # def plot_heatmaps(self, effect_size_df, p_values_df, z_values_df, factor, method, start_date_str, end_date_str):
    #     try:
    #         # Creating pivot tables
    #         effect_size_heatmap_data = effect_size_df.pivot(index="days_since_start", columns="median_time_interval", values="effect_size")
    #         neg_log_p_heatmap_data = p_values_df.pivot(index="days_since_start", columns="median_time_interval", values="neg_log_p_value")
    #         z_value_heatmap_data = z_values_df.pivot(index="days_since_start", columns="median_time_interval", values="z_value")

    #         # Plotting
    #         fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 10))

    #         for ax in axes.flat:
    #             for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
    #                          ax.get_xticklabels() + ax.get_yticklabels()):
    #                 item.set_fontsize(12)
                
    #         sns.heatmap(effect_size_heatmap_data, annot=False, fmt=".2f", cmap="viridis", 
    #                     cbar_kws={'label': 'Coefficient' if method == 'glmm' else 'F Value'}, ax=axes[0])
    #         axes[0].set_title(f'{"Coefficient" if method == "glmm" else "F Value"} for Factor: {factor}', fontsize=14)
    #         axes[0].set_xlabel("Time Interval (median hour)", fontsize=12)
    #         axes[0].set_ylabel("Days Since Start", fontsize=12)

    #         p_value_mask = neg_log_p_heatmap_data.isnull()
    #         sns.heatmap(neg_log_p_heatmap_data, annot=False, fmt=".2f", cmap="viridis", 
    #                     cbar_kws={'label': '-log10(p-value)'}, ax=axes[1], mask=p_value_mask)
    #         axes[1].set_title(f'-log10(p-value) for Factor: {factor}', fontsize=14)
    #         axes[1].set_xlabel("Time Interval (median hour)", fontsize=12)
    #         axes[1].set_ylabel("Days Since Start", fontsize=12)

    #         z_value_mask = z_value_heatmap_data.isnull()
    #         sns.heatmap(z_value_heatmap_data, annot=False, fmt=".2f", cmap="viridis", 
    #                     cbar_kws={'label': 'Z Value'}, ax=axes[2], mask=z_value_mask)
    #         axes[2].set_title(f'z-value for Factor: {factor}', fontsize=14)
    #         axes[2].set_xlabel("Time Interval (median hour)", fontsize=12)
    #         axes[2].set_ylabel("Days Since Start", fontsize=12)

    #         plt.tight_layout()

    #         filename = f"heatmaps_{factor}_{method}_{start_date_str}_to_{end_date_str}"
    #         self.save_plot(fig, filename)
    #         plt.close(fig)

    #     except Exception as e:
    #         self.log(f"Error during plotting heatmaps: {e}")

    def normalize_by_daily_total(self, data):

        if isinstance(data, pd.Series):
            return self._normalize_series_by_day(data)
        else:
            raise TypeError("Data must be a pandas Series or DataFrame for normalization.")

    def _normalize_series_by_day(self, series):
        # Resample to have daily means
        daily_totals = series.resample('D').sum()

        # Ensure index compatibility in mapping
        daily_total_map = series.index.floor('D').map(daily_totals.to_dict())

        # Perform division safely
        normalized_series = series / daily_total_map



        return normalized_series
    
###################### Dimension reduction ###############
    def visualize_reduced_data(self, reduced_data, experiments, method, days, plot_days, interval_hours, data_level, normalize, explained_variance):

        experiment_to_group_category = {}
        for category_name, groups in experiments.items():
            for group_name, experiment_list in groups.items():
                for exp_data in experiment_list:
                    alias = exp_data.get('alias')
                    if alias:
                        experiment_to_group_category[alias] = (group_name, category_name)
        coeff = 1.2
        # Define figure and grid layout
        fig = plt.figure(figsize=(coeff*6, coeff*2.5))  # Adjust to widen the space for square plot area
        gs = GridSpec(nrows=1, ncols=2, width_ratios=[5, 1])  # Allocate more space for the plot

        ax = fig.add_subplot(gs[0])
        ax.set_box_aspect(1)  # This makes the data area square

        start_date = pd.to_datetime(days[0])
        category_handles = {}
        legend_entries = set()

        for idx, (label, row) in enumerate(reduced_data.iterrows()):
            if label in experiment_to_group_category:
                group_name, category_name = experiment_to_group_category[label]
                color = self.group_colors.get(group_name, 'gray')
                marker = self.category_markers.get(category_name, 'o')
            elif label in experiments:
                category_name = label
                group_name = None
                color = self.category_colors.get(category_name, 'gray')
                marker = self.category_markers.get(category_name, 'o')
            else:
                group_name = label
                category_name = next((cat_name for cat_name, groups in experiments.items() if group_name in groups), "Unknown")
                color = self.group_colors.get(group_name, 'gray')
                marker = self.category_markers.get(category_name, 'o')

            legend_label = f"{group_name}_{category_name}"
            plot_label = legend_label if legend_label not in legend_entries else "_nolegend_"

            sc = ax.scatter(
                row[0], row[1],
                color=color,
                marker=marker,
                s=40,
                alpha=1,
                label=plot_label,
                edgecolors='none'
            )

            if category_name and legend_label not in legend_entries:
                if category_name not in category_handles:
                    category_handles[category_name] = []
                category_handles[category_name].append((sc, group_name))
                legend_entries.add(legend_label)

            current_date = pd.to_datetime(days[idx])
            days_since_start = (current_date - start_date).days
            if plot_days:
                ax.annotate(days_since_start, (row[0], row[1]), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=8)

        # Prepare handles and labels
        handles, legend_labels = [], []
        for category, handlers in category_handles.items():
            for handle, label in handlers:
                handles.append(handle)
                legend_labels.append(f"{label}_{category}")

        # Set titles and labels with variance explained
        ax.set_title(f'{method.upper()} Dimensionality Reduction')
        ax.set_xlabel(f"Component 1 ({explained_variance[0]*100:.2f}% variance)")
        ax.set_ylabel(f"Component 2 ({explained_variance[1]*100:.2f}% variance)")

        # Create the legend in a separate subplot
        ax_legend = fig.add_subplot(gs[1])
        ax_legend.axis('off')
        ax_legend.legend(
            handles=handles,
            labels=legend_labels,
            title='Group/Category',
            fontsize='small',
            loc='center',
            handletextpad=0.5,
            columnspacing=1,
            borderaxespad=0.5
        )

        # Ensure the layout fits within the boundaries
        plt.tight_layout()

        filename = f"{method}_scatter_{interval_hours}mins_{data_level}_Normalize_{normalize}"
        self.save_plot(fig, filename)
        print(f"Plot saved to {filename}")

        plt.show()
        plt.close(fig)




####### GLMM plotting 

    def plot_glmm_results(self, results, fixed_effects, grouped_experiments):
        # Create a figure with 4 vertically aligned subplots
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(14, 16))

        # Extract information for GLMM plots
        minutes = [res['minute'] for res in results]
        coefficients = [res['coefficient'] for res in results]
        z_scores = [res['z_score'] for res in results]
        p_values = [res['p_value'] for res in results]

        # Plot Coefficients
        axes[0].plot(minutes, coefficients, marker='o', linestyle='-')
        axes[0].set_title('GLMM Coefficients')
        axes[0].set_ylabel('Coefficient Value')

        # Plot Z-Scores
        axes[1].plot(minutes, z_scores, marker='o', linestyle='-')
        axes[1].set_title('GLMM Z-Scores')
        axes[1].set_ylabel('Z-Score')

        # Plot P-Values in -log10 scale
        log_p_values = [-np.log10(p) if p > 0 else np.nan for p in p_values]
        axes[2].plot(minutes, log_p_values, marker='o', linestyle='-')
        axes[2].set_title('GLMM P-Values (-log10)')
        axes[2].set_ylabel('-Log10(P-Value)')
        axes[2].axhline(-np.log10(0.01), color='r', linestyle='--', label='p = 0.05')
        axes[2].legend()

        # Configure the daily average plot in the 4th subplot
        plot_args = {
            'selected_vars': ['numb_mosquitos_flying'],  # Change this to the desired variable
            'resample_interval': "1T",
            'moving_avg_window': "20",
            'start_date_str': "2023-03-01",  # Modify this to the actual start date if needed
            'end_date_str': "2023-03-10",    # Modify this to the actual end date if needed
            'threshold': None,
            'experiments_to_plot': [],
            'groups_to_plot': [],
            'categories_to_plot': list(grouped_experiments.keys()) if 'Category' in fixed_effects else [],
            'grouped_experiments': grouped_experiments,
            'start_hour': 0,
            'end_hour': 23,
            'ax': axes[3],
            'normalize': False,
            'display_names': ['Flying Mosquitoes']
        }
        
        # Add groups if "Group" is in fixed_effects
        if 'Group' in fixed_effects:
            plot_args['groups_to_plot'] = [grp for category in grouped_experiments.values() for grp in category.keys()]

        self.plot_daily_average(**plot_args)

        # Adding titles and labels for daily averages subplot
        axes[3].set_title('Daily Average of Selected Variables')
        axes[3].set_xlabel('Hour of the Day')
        axes[3].set_ylabel('Average Value')

        plt.tight_layout()
        plt.show()


    def extract_median_time(self,interval_str):
        """Calculate median time from a time interval string."""
        try:
            start_str, end_str = interval_str.split('-')
            start_time = datetime.strptime(start_str, '%H:%M:%S')
            end_time = datetime.strptime(end_str, '%H:%M:%S')
            median_time = start_time + (end_time - start_time) / 2
            return median_time.strftime('%H:%M')
        except Exception as e:
            raise ValueError(f"Error processing interval '{interval_str}': {e}")

    def validate_and_reshape_contributions(self,contributions, lda_components):
        """
        Ensure contributions array is compatible with LDA components.
        
        Parameters:
        - contributions: Array of feature contributions
        - lda_components: Number of LDA components

        Returns:
        - 2D reshaped contributions array
        """
        contributions = np.atleast_2d(contributions)
        if contributions.shape[0] != lda_components:
            raise ValueError(
                f"Contributions dimension {contributions.shape[0]} does not match expected number of LDA components {lda_components}."
            )
        return contributions

    def plot_combined_figure(self, pca_data, target, lda, feature_names, contributions,orthogonal_contributions):
        """
        Plot PCA data with LDA separation and contributions.

        Parameters:
        - pca_data: PCA-transformed data
        - target: Target variables
        - lda: Fitted LDA model
        - feature_names: List of feature names
        - contributions: Contributions of features to LDA
        """

        try:
            coeff = 1.0
            fig, axes = plt.subplots(1, 2, figsize=(5*coeff, 3*coeff), constrained_layout=True)

            # Plot 1: PCA Data with LDA Separation
            ax1 = axes[0]

            # To use the custom colors and markers logic defined in 'visualize_reduced_data':
            for i, (x, y) in enumerate(zip(pca_data[:, 0], pca_data[:, 1])):
                experiment_name = target[i]
                if experiment_name == 1:
                    color = "skyblue"
                    marker = "s"
                else:
                    color = "lightcoral"
                    marker = "o"
                # Plot each point with its specific color and marker
                scatter = ax1.scatter(x, y, color=color, marker=marker, s=20, alpha=0.8, edgecolors='none')

            # Use a legend for the plot if necessary. You can create a color/marker legend based on the colors/markers used
            handles = [mlines.Line2D([], [], color=color, marker=marker, linestyle='None', markersize=10, label=name)
                    for name, (color, marker) in zip(self.group_colors.keys(), zip(self.group_colors.values(), self.group_markers.values()))]
            ax1.legend(handles=handles, title="Experiments", loc='upper right')

            # Calculate limits based on scatter points
            xlims = ax1.get_xlim()
            ylims = ax1.get_ylim()

            # LDA line
            x_vals = np.array(ax1.get_xlim())
            mean_projection = np.mean(lda.transform(pca_data))
            lda_scaling_x, lda_scaling_y = lda.scalings_.flatten()
            y_vals = -x_vals * lda_scaling_x / lda_scaling_y + mean_projection
            ax1.plot(x_vals, y_vals, color='red', linestyle='--', label='LDA Line')

            # Orthogonal line
            slope_lda = -lda_scaling_x / lda_scaling_y
            orthogonal_slope = -1 / slope_lda
            midpoint = np.mean(pca_data, axis=0)
            y_ortho_vals = orthogonal_slope * (x_vals - midpoint[0]) + midpoint[1]
            ax1.plot(x_vals, y_ortho_vals, color='blue', linestyle='--', label='Orthogonal Line')

            ax1.set_title('PCA Data with LDA Separation')
            ax1.set_xlabel('PCA Component 1')
            ax1.set_ylabel('PCA Component 2')
            ax1.set_xlim(xlims)
            ax1.set_ylim(ylims)
            legend1 = ax1.legend(*scatter.legend_elements(), title="Classes")
            ax1.add_artist(legend1)
            ax1.set_box_aspect(1)  # This makes the data area square
            ax1.legend()



            # Plot 2: Contribution Analysis
            ax2 = axes[1]

            # Prepare data for plotting
            data = pd.DataFrame({
                'feature': feature_names,
                'loading': contributions.flatten(),
                'orthogonal_loading': orthogonal_contributions.flatten()
            })

            # Extract median times from intervals
            try:
                data['time'] = data['feature'].apply(lambda x: x.split('_')[0])
                data['median_time'] = data['time'].apply(self.extract_median_time)
                data['median_time'] = pd.to_datetime(data['median_time'], format='%H:%M')
            except Exception as e:
                raise ValueError(f"Error processing time intervals: {e}")

            data['abs_loading'] = data['loading'].abs()
            max_contributors = data.loc[data.groupby('median_time')['abs_loading'].idxmax()]

            ax2.plot(max_contributors['median_time'], max_contributors['loading'], marker='None', linestyle='-', color='red', label='LDA Contributions')

            # Plot orthogonal contributions (blue line)
            data['abs_loading_ortho'] = data['orthogonal_loading'].abs()
            max_ortho_contributors = data.loc[data.groupby('median_time')['abs_loading_ortho'].idxmax()]
            ax2.plot(max_ortho_contributors['median_time'], max_ortho_contributors['orthogonal_loading'], marker='None', linestyle='--', color='blue', label='Orthogonal Contributions')

            ax2.axhline(y=0, color='grey', linestyle='--')
            max_abs_value = max(max_contributors['loading'].abs().max(), max_ortho_contributors['orthogonal_loading'].abs().max())
            ax2.set_ylim(-max_abs_value, max_abs_value)
            ax2.set_title('Max Contributions by Time Interval')
            ax2.set_ylabel('Contribution')
            ax2.set_xlabel('Time of Day')
            ax2.legend()

            # Set x-ticks every 3 hours
            start_time = max_contributors['median_time'].min().floor('H')
            end_time = max_contributors['median_time'].max().ceil('H')
            xticks = pd.date_range(start=start_time, end=end_time, freq='3H')
            ax2.set_xticks(xticks)
            ax2.set_xticklabels(xticks.strftime('%H:%M'), rotation=45, ha='right')
            ax2.set_box_aspect(1)  # This makes the data area square


            plt.tight_layout()

            # Save and display the combined plot
            self.save_plot(fig, "Combined_PCA_LDA_and_Contributions")
            plt.show()

        except Exception as e:
            print(f"Error in plot_combined_figure: {e}")

# Note: Ensure `plot_manager` can handle saving multi-subplot figures.



