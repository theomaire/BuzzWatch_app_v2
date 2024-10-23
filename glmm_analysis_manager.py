import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

class GLMMAnalysisManager:
    def __init__(self, log_func, results_dir):
        self.log = log_func
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def segment_and_run_glmm(self, data, fixed_effects, random_effects, day_intervals, time_intervals, variable_name):
        for start_day, end_day in day_intervals:
            for start_hour, end_hour, span_midnight in time_intervals:
                segmented_data = self.segment_data(data, start_day, end_day, start_hour, end_hour, span_midnight)
                if not segmented_data.empty:
                    self.run_and_save_glmm(segmented_data, fixed_effects, random_effects, variable_name, start_day, end_day, start_hour, end_hour)

    def segment_data(self, data, start_day, end_day, start_hour, end_hour, span_midnight):
        # Filter by days
        day_segment = data[(data.index.date >= start_day) & (data.index.date <= end_day)]

        # Further filter by hours
        if span_midnight:
            return pd.concat([day_segment.between_time(f'{start_hour:02d}:00', '23:59'), day_segment.between_time('00:00', f'{end_hour:02d}:00')])
        else:
            return day_segment.between_time(f'{start_hour:02d}:00', f'{end_hour:02d}:00')

    def run_and_save_glmm(self, data, fixed_effects, random_effects, variable, start_day, end_day, start_hour, end_hour):
        # Ensure necessary data type conversions
        data = data.dropna(subset=[variable] + fixed_effects)
        data = data.reset_index(drop=True)
        for effect in fixed_effects + random_effects:
            data[effect] = data[effect].astype('category')

        # Fit GLMM model
        formula = f"{variable} ~ {' + '.join(fixed_effects)}"
        model = MixedLM.from_formula(formula, groups=data[random_effects[0]], data=data)

        try:
            fit_result = model.fit(method='lbfgs', maxiter=500, full_output=True)
            self.log(f"Successfully fitted GLMM for {start_day} to {end_day}, {start_hour} to {end_hour} hours.")
            self.save_glmm_results(fit_result, variable, start_day, end_day, start_hour, end_hour)
        except Exception as e:
            self.log(f"GLMM fitting failed for {start_day}-{end_day}, {start_hour}-{end_hour} hours: {e}")

    def save_glmm_results(self, fit_result, variable, start_day, end_day, start_hour, end_hour):
        # Construct result filename
        filename_prefix = f"glmm_{variable}_{start_day}_{end_day}_{start_hour}_{end_hour}"
        output_path = os.path.join(self.results_dir, f"{filename_prefix}_results.csv")
        try:
            summary_frame = fit_result.summary().tables[1]
            summary_frame.to_csv(output_path, index=False)
            self.log(f"GLMM results saved to {output_path}")
        except Exception as e:
            self.log(f"Error saving GLMM results: {e}")

    def visualize_glmm_overall_results(self):
        # Load results from directory
        result_files = [f for f in os.listdir(self.results_dir) if "results.csv" in f]
        all_results = pd.DataFrame()
        for file in result_files:
            file_path = os.path.join(self.results_dir, file)
            result_df = pd.read_csv(file_path)
            result_df['file'] = file
            all_results = pd.concat([all_results, result_df])

        # Example visualization using Matplotlib
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for key, grp in all_results.groupby(['file']):
            plt.plot(grp.index, grp['Coef.'], label=key)
        plt.legend()
        plt.title("GLMM Effect Size Over Intervals")
        plt.xlabel("Interval")
        plt.ylabel("Effect Size")
        plt.show()