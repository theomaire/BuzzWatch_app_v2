import tkinter as tk
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.regression.mixed_linear_model import MixedLM
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import os
from datetime import datetime
from plot_manager import PlotManager


warnings.filterwarnings("ignore", category=ConvergenceWarning)

class StatisticalAnalysisManager:
    def __init__(self, log):
        self.log = log

    def segment_data(self, data, start_hour, end_hour):
        segmented_data = data.between_time(f'{start_hour}:00', f'{end_hour}:00')
        return segmented_data

    def visualize_variability(self, data, by_category, variable, ax):
        sns.boxplot(x=by_category, y=variable, data=data, ax=ax)
        ax.set_title(f'Variability of {variable} by {by_category}')
        ax.set_ylabel(variable)
        ax.set_xlabel(by_category)

    def perform_anova(self, data, variable, fixed_effects):
        # Ensure no NA values in relevant columns
        factors = fixed_effects 
        data = data.dropna(subset=[variable] + factors)

        # Reset index to ensure smooth operation
        data = data.reset_index(drop=True)

        # Ensure factors are categorical
        for factor in factors:
            if data[factor].dtype == 'object':
                data[factor] = data[factor].astype('category')

        # Build the formula for ANOVA
        formula = f"{variable} ~ {' + '.join(factors)}"

        try:
            model = ols(formula, data).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            return anova_table
        except Exception as e:
            self.log(f"Exception during ANOVA: {e}")
            return None

    def perform_glmm(self, data, variable, fixed_effects, random_effects,output_dir, filename_prefix):
        # Ensure no NA values in relevant columns
        data = data.dropna(subset=[variable] + fixed_effects )
        
        # Reset index and set types, categorical as needed
        data = data.reset_index(drop=True)
        for effect in fixed_effects:
            data[effect] = data[effect].astype('category')
        
        formula = f"{variable} ~ {' + '.join(fixed_effects)}"
        #print(data)

        md = MixedLM.from_formula(formula, groups=data['Day'], data=data)
        mdf = md.fit(method='lbfgs', maxiter=500, full_output=True)
        self.log(f"GLMM Model fit successfully using 'lbfgs'.")
        
        # Save results after successful fit
        self.save_glmm_results(mdf, filename_prefix, output_dir)
        return self.extract_glmm_summary(mdf)




    def analyze_variability(self, data, variable, start_hour, end_hour, fixed_effects, random_effects, method='anova'):
        if method == 'anova':
            anova_table = self.perform_anova(data, variable, fixed_effects)
            return self.extract_anova_results(anova_table)
        elif method == 'glmm':
            glmm_summary = self.perform_glmm(data, variable, fixed_effects, random_effects)
            #print(glmm_summary)
            return self.extract_glmm_results(glmm_summary)

    def extract_anova_results(self, anova_table):
        if anova_table is None:
            self.log("ANOVA table is None, cannot extract results.")
            return None
        results = {
            "factor": [],
            "sum_sq": [],
            "df": [],
            "F": [],
            "PR(>F)": []
        }
        for factor, row in anova_table.iterrows():
            results["factor"].append(factor)
            results["sum_sq"].append(row["sum_sq"])
            results["df"].append(row["df"])
            results["F"].append(row["F"])
            results["PR(>F)"].append(row["PR(>F)"])
        return results



    def extract_glmm_results(self, glmm_summary):
        if glmm_summary is None:
            self.log("GLMM summary is None, cannot extract results.")
            return None
        results = {
            "coefficient": [],
            "std_error": [],
            "z_value": [],
            "p_value": [],
            "factor": glmm_summary.index.tolist()
        }
        for factor, row in glmm_summary.iterrows():
            results["coefficient"].append(row["Coef."])
            results["std_error"].append(row["Std.Err."])
            results["z_value"].append(row["z"])
            results["p_value"].append(row["P>|z|"])
        return results
    

    def run_dimensionality_reduction(self, data,plot_manager, method='pca', n_components=3):
        explained_variance = None

        if method == 'pca':
            model = PCA(n_components=n_components)
            reduced_data = model.fit_transform(data)
            explained_variance = model.explained_variance_ratio_
            self.retrieve_principal_contributors(model,plot_manager, data.columns)
        elif method == 't-sne':
            model = TSNE(n_components=n_components, metric='euclidean')
            reduced_data = model.fit_transform(data)
        elif method == 'umap':
            model = umap.UMAP(n_components=n_components, metric='euclidean')
            reduced_data = model.fit_transform(data)
        else:
            raise ValueError(f"Unknown method: {method}")

        return reduced_data, explained_variance
    


    def extract_glmm_summary(self, mdf):
        summary = mdf.summary()
        tables = summary.tables

        # Since coefficients_table is already a DataFrame, work with it directly
        coefficients_df = tables[1]
        
        # Debug: Verify the correct structure of coefficients_df
        print("Extracted Coefficients DataFrame:")
        print(coefficients_df)

        # Convert necessary columns to numeric types
        numeric_columns = ["Coef.", "Std.Err.", "z", "P>|z|", "[0.025", "0.975]"]
        for col in numeric_columns:
            if col in coefficients_df.columns:
                coefficients_df[col] = pd.to_numeric(coefficients_df[col], errors='coerce')

        print("Converted Coefficients DataFrame:")
        print(coefficients_df)
        
        return coefficients_df


    def save_glmm_results(self, mdf, filename_prefix, output_dir):
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Extract the GLMM summary data into a DataFrame
            fixed_effects_df = self.extract_glmm_summary(mdf)
            
            # Check if the extraction was successful before saving
            if fixed_effects_df is not None:
                results_path = os.path.join(output_dir, f"{filename_prefix}_glmm_results.csv")
                fixed_effects_df.to_csv(results_path, index=False)
                self.log(f"GLMM results successfully saved to {results_path}")
            else:
                self.log("Failed to save GLMM results due to extraction issue.")
        
        except Exception as e:
            self.log(f"Exception during GLMM results saving: {e}")



    def retrieve_principal_contributors(self, pca_model,plot_manager, feature_names):
        loadings = pca_model.components_

        # Setup for subplots - 3 rows (one for each PC) and 1 column
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5, 10), sharex=True, constrained_layout=True)

        # Define colors for different measurement types

        for i, (ax, pc_label) in enumerate(zip(axes, ["PC1", "PC2", "PC3"])):
            self.plot_principal_contributors(ax, loadings[i], feature_names, pc_label)
        plt.tight_layout()
        plot_manager.save_plot(fig, "PCA_reverse_inference_time_contribution")

        plt.show()
        plt.close(fig)

    def plot_principal_contributors(self, ax, pc_loadings, feature_names, pc_label):
        # Create a DataFrame for easy manipulation
        data = pd.DataFrame({
            'feature': feature_names,
            'loading': pc_loadings
        })

        # Extract time intervals and calculate median time for datetime formatting
        data['time'] = data['feature'].apply(lambda x: x.split('_')[0])
        data['median_time'] = data['time'].apply(self.extract_median_time)

        # Identify the main contributor using absolute values but retain the sign for plotting
        data['abs_loading'] = data['loading'].abs()
        max_contributors = data.loc[data.groupby('median_time')['abs_loading'].idxmax()]

        # Plot line with the actual contribution values, maintaining their sign
        ax.plot(max_contributors['median_time'], max_contributors['loading'], marker='None', linestyle='-')

            # Add a horizontal line at y=0
        ax.axhline(y=0, color='grey', linestyle='--')
            # Set y-axis limits to be symmetrical
        max_abs_value = max(abs(max_contributors['loading'].min()), max_contributors['loading'].max())
        ax.set_ylim(-max_abs_value, max_abs_value)

        ax.set_title(f'Max Contribution to {pc_label} by Time of Day')
        ax.set_ylabel('Contribution')  # Update label to indicate signed contribution

        # Adjust x-ticks for improved readability (show every 3 hours)
        tick_indices = max_contributors['median_time'][::9]  # Modify index to your actual data's frequency if needed
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_indices, rotation=45, ha='right')
        ax.set_xlabel('Time of Day')

    @staticmethod
    def extract_median_time(interval_str):
        start_str, end_str = interval_str.split('-')
        start_time = datetime.strptime(start_str, '%H:%M:%S')
        end_time = datetime.strptime(end_str, '%H:%M:%S')
        median_time = start_time + (end_time - start_time) / 2
        return median_time.strftime('%H:%M')
