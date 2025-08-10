import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

class XAIAnalyzer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        
    def compute_shapley_values(self, X_train, X_test, model_predict_fn=None):
        """
        Compute Shapley values to explain model predictions
        """
        # If a custom prediction function is provided, use it
        if model_predict_fn is not None:
            explainer = shap.Explainer(model_predict_fn, X_train)
        else:
            # Otherwise, directly use the model's predict method
            explainer = shap.Explainer(self.model.predict, X_train)
            
        shap_values = explainer(X_test)
        return shap_values
    
    def plot_modality_importance(self, shap_values, modality_mapping):
        """
        Plot the importance of different modalities
        """
        # Aggregate feature importance based on modality mapping
        modality_importance = {}
        for i, feature_name in enumerate(self.feature_names):
            # Ensure index is within range
            if i < shap_values.values.shape[1]:
                modality = modality_mapping.get(feature_name, "unknown")
                if modality not in modality_importance:
                    modality_importance[modality] = 0
                modality_importance[modality] += np.abs(shap_values.values[:, i]).mean()
        
        # Plot bar chart
        modalities = list(modality_importance.keys())
        importances = list(modality_importance.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(modalities, importances, color='skyblue')
        plt.title("Modality Importance based on SHAP Values")
        plt.xlabel("Modality")
        plt.ylabel("Average |SHAP Value|")
        plt.xticks(rotation=45, ha='right')
        
        # Add numerical labels on bars
        for bar, importance in zip(bars, importances):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f'{importance:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
    def generate_explanation_report(self, shap_values, sample_idx, top_k=5):
        """
        Generate an explanation report for a specific sample
        """
        # Check if index is valid
        if sample_idx >= len(shap_values.values):
            return "Sample index out of range"
            
        # Get SHAP values for the sample
        sample_shap = shap_values.values[sample_idx]
        
        # Sort and get the most important features
        feature_importance = []
        for i, feature_name in enumerate(self.feature_names):
            if i < len(sample_shap):  # Ensure index is valid
                feature_importance.append((feature_name, sample_shap[i]))
                
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Generate report
        base_value = shap_values.base_values
        if isinstance(base_value, np.ndarray):
            base_value = base_value[sample_idx] if sample_idx < len(base_value) else base_value[0]
        
        prediction = base_value + sample_shap.sum()
        
        report = f"Explanation for Sample {sample_idx}:\n"
        report += f"Base Value: {base_value:.4f}\n"
        report += f"Predicted Probability: {prediction:.4f}\n\n"
        report += "Top Contributing Features:\n"
        for i, (feature, shap_val) in enumerate(feature_importance[:top_k]):
            report += f"{i+1}. {feature}: {shap_val:.4f}\n"
            
        return report
        
    def plot_feature_importance_waterfall(self, shap_values, sample_idx):
        """
        Plot a feature importance waterfall chart for a specific sample
        """
        # Check if index is valid
        if sample_idx >= len(shap_values.values):
            print("Sample index out of range")
            return
            
        # Get SHAP values for the sample
        sample_shap = shap_values.values[sample_idx]
        base_value = shap_values.base_values
        if isinstance(base_value, np.ndarray):
            base_value = base_value[sample_idx] if sample_idx < len(base_value) else base_value[0]
        
        # Create feature importance list
        feature_importance = []
        for i, feature_name in enumerate(self.feature_names):
            if i < len(sample_shap):  # Ensure index is valid
                feature_importance.append((feature_name, sample_shap[i]))
                
        # Sort by absolute value
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Extract feature names and SHAP values
        features = [x[0] for x in feature_importance]
        shap_vals = [x[1] for x in feature_importance]
        
        # Create waterfall chart
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate cumulative values
        cumulative = np.concatenate([[base_value], base_value + np.cumsum(shap_vals)])
        
        # Plot bars
        colors = ['green' if x > 0 else 'red' for x in shap_vals]
        y_pos = np.arange(len(features))
        
        ax.barh(y_pos, shap_vals, color=colors, left=cumulative[:-1])
        
        # Add baseline and prediction values
        ax.axvline(base_value, color='black', linestyle='--', linewidth=1, label=f'Base Value: {base_value:.3f}')
        ax.axvline(cumulative[-1], color='blue', linestyle='-', linewidth=2, label=f'Prediction: {cumulative[-1]:.3f}')
        
        # Set labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('SHAP Value')
        ax.set_title(f'Feature Contribution Waterfall Plot for Sample {sample_idx}')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def compute_permutation_importance(model, X_test, y_test, feature_names, n_repeats=10):
    """
    Compute feature importance using permutation importance
    """
    # Calculate permutation importance
    perm_importance = permutation_importance(
        model, X_test, y_test, 
        n_repeats=n_repeats,
        random_state=42
    )
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=False)
    
    return importance_df


def plot_permutation_importance(importance_df, top_k=20):
    """
    Plot permutation importance
    """
    # Select top K important features
    top_features = importance_df.head(top_k)
    
    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(top_features))
    
    plt.barh(y_pos, top_features['importance'], xerr=top_features['std'], 
             align='center', alpha=0.7, color='lightcoral')
    plt.yticks(y_pos, top_features['feature'])
    plt.xlabel('Permutation Importance')
    plt.title(f'Top {top_k} Feature Permutation Importance')
    plt.gca().invert_yaxis()  # Most important features at the top
    
    plt.tight_layout()
    plt.show()
