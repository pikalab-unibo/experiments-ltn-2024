from scipy.stats import wilcoxon
import numpy as np

def compare_models(metrics_df):
    best_models = {}
    
    # Determine the best model for each metric using mean scores
    for metric in metrics_df.columns:
        try:
            mean_scores = metrics_df[metric].apply(np.mean)
            best_model = mean_scores.idxmax()
            best_models[metric] = best_model
        except Exception as e:
            print(f"Error with metric {metric}: {e}")
            continue
    
    # Perform Wilcoxon signed-rank test with Bonferroni correction
    results = {}
    for metric in metrics_df.columns:
        try:
            best_model = best_models[metric]
            best_scores = metrics_df.loc[best_model, metric]
            results[metric] = {}
            
            for model_name in metrics_df.index:
                if model_name == best_model:
                    continue
                
                comparison_scores = metrics_df.loc[model_name, metric]
                
                # Perform the Wilcoxon signed-rank test
                try:
                    t_stat, p_value = wilcoxon(best_scores, comparison_scores, alternative='greater')
                except ValueError:
                    print(f"Could not perform Wilcoxon test between {best_model} and {model_name} for metric {metric}")
                    continue
                
                # Apply Bonferroni correction
                corrected_p_value = p_value * (metrics_df.shape[0] - 1)
                results[metric][model_name] = corrected_p_value 

        except Exception as e:
            print(f"Error processing metric {metric}: {e}")
            continue

    # Significance level for Bonferroni correction
    significance_level = 0.05
    for metric, comparisons in results.items():
        try:
            print(f"\n{metric}:")
            best_model = best_models[metric]
            for model_name, corrected_p_value in comparisons.items():
                if corrected_p_value < significance_level:
                    print(f"  {best_model} is significantly better than {model_name} (adjusted p = {corrected_p_value:.4f})")
                else:
                    print(f"  {best_model} is NOT significantly better than {model_name} (adjusted p = {corrected_p_value:.4f})")
        except Exception as e:
            print(f"Error displaying results for metric {metric}: {e}")

def perform_t_tests(metrics_df, model_name):
    results = {}

    for metric in metrics_df.columns:
        if model_name == "Logic Tensor Network":
            continue
        
        # Get the scores for both models
        ski_mlp_scores = np.array(metrics_df.loc['Logic Tensor Network', metric])
        regular_mlp_scores = np.array(metrics_df.loc[model_name, metric])
        
        try:
            # Perform the Wilcoxon signed-rank test
            t_stat, p_value = wilcoxon(ski_mlp_scores, regular_mlp_scores, alternative='greater')
        except ValueError:
            print(f"Could not perform Wilcoxon test between Logic Tensor Network and {model_name} for metric {metric}")
            continue
        
        # Calculate the mean difference
        mean_difference = np.mean(ski_mlp_scores - regular_mlp_scores)
        
        # Apply Bonferroni correction
        corrected_p_value = p_value * (metrics_df.shape[1] - 1)
        
        results[metric] = {'p_value': corrected_p_value, 'mean_difference': mean_difference}

    # Significance level for Bonferroni correction
    significance_level = 0.05

    for metric, result in results.items():
        if model_name == "Logic Tensor Network":
            continue

        try:
            p_value = result['p_value']
            mean_difference = result['mean_difference']
            
            print(f"\n{metric}:")
            if p_value < significance_level:
                if mean_difference > 0:
                    print(f"  Logic Tensor Network is significantly better than {model_name} (adjusted p = {p_value:.4f})")
                else:
                    print(f"  {model_name} is significantly better than Logic Tensor Network (adjusted p = {p_value:.4f})")
            else:
                print(f"  There is no significant difference between Logic Tensor Network and {model_name} (adjusted p = {p_value:.4f})")
        except Exception as e:
            print(f"Error processing metric {metric}: {e}")
