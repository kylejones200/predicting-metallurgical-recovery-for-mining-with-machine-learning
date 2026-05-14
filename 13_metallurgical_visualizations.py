import signalplot
import sys
import os

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

"""
Visualization generation for Blog 13: Metallurgical Recovery Prediction
Creates minimalist-style visualizations for recovery prediction models.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings



from pathlib import Path
warnings.filterwarnings('ignore')

def apply_minimalist_style_manual(ax):
    """Apply minimalist style components manually to axis."""
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
def generate_metallurgical_data(n_samples=800):
    """
    Generate synthetic metallurgical recovery data.
    
    Features: Cu grade, Fe%, S%, grind_size_um, ph, collector_dose, frother_dose
    Target: Recovery (%)
    """
    np.random.seed(42)
    
    # Generate features
    cu_grade = np.random.uniform(0.3, 2.5, n_samples)
    fe_pct = np.random.uniform(8, 35, n_samples)
    s_pct = np.random.uniform(1, 15, n_samples)
    grind_size = np.random.uniform(50, 200, n_samples)
    ph = np.random.uniform(8.5, 11.5, n_samples)
    collector = np.random.uniform(10, 80, n_samples)
    frother = np.random.uniform(5, 40, n_samples)
    
    # Recovery model (non-linear relationships)
    recovery = (
        75 +  # Base recovery
        8 * np.log(cu_grade + 0.1) +  # Higher grade slightly easier
        -0.15 * fe_pct +  # Iron penalty
        0.3 * s_pct +  # Sulfur helps flotation
        -0.08 * (grind_size - 100) +  # Finer grind better
        5 * (ph - 9.5)**2 * -0.5 +  # Optimal pH around 9.5
        0.1 * collector +  # More collector helps
        0.05 * frother  # Frother effect
    )
    
    # Add interaction: high Fe + coarse grind = worse recovery
    recovery -= 0.002 * fe_pct * grind_size
    
    # Add noise
    recovery += np.random.randn(n_samples) * 2.5
    
    # Clip to realistic range
    recovery = np.clip(recovery, 60, 95)
    
    X = np.column_stack([cu_grade, fe_pct, s_pct, grind_size, ph, collector, frother])
    
    return X, recovery

def create_main_recovery_prediction_plot(plot: bool = False):
    """
    Create predicted vs actual recovery plot.
    """
    logger.info("Generating main recovery prediction visualization...")
    
    # Generate data
    X, y = generate_metallurgical_data(n_samples=800)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Train model
    logger.info("  Training Gradient Boosting model...")
    model = GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Create figure
    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plots
        ax.scatter(y_train, y_pred_train, alpha=0.5, s=30, 
                  edgecolors='black', linewidth=0.5, c='#0074D9',
                  label=f'Training (n={len(y_train)})')
    
        ax.scatter(y_test, y_pred_test, alpha=0.7, s=50, 
                  edgecolors='black', linewidth=0.8, c='#FF851B',
                  label=f'Test (n={len(y_test)})')
    
    # Perfect prediction line
        min_val = min(y.min(), y_pred_train.min(), y_pred_test.min())
        max_val = max(y.max(), y_pred_train.max(), y_pred_test.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'k--', linewidth=2, alpha=0.7, label='Perfect Prediction')
    
    # ±5% error bands
        ax.fill_between([min_val, max_val], 
                        [min_val * 0.95, max_val * 0.95],
                        [min_val * 1.05, max_val * 1.05],
                        alpha=0.1, color='gray', label='±5% Error Band')
    
    # Apply minimalist style
        apply_minimalist_style_manual(ax)
    
        ax.set_xlabel('Actual Recovery (%)', fontsize=11)
        ax.set_ylabel('Predicted Recovery (%)', fontsize=11)
        ax.set_title('Metallurgical Recovery Prediction', 
                     fontsize=13, fontweight='bold', loc='left', pad=20)
    
        ax.legend(loc='lower right', frameon=False, fontsize=9)
    
    # Add metrics box
        metrics_text = (f'Training:\n  R² = {train_r2:.3f}\n  MAE = {train_mae:.2f}%\n\n'
                       f'Test:\n  R² = {test_r2:.3f}\n  MAE = {test_mae:.2f}%')
    
        ax.text(0.05, 0.95, metrics_text,
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1))
    
        ax.set_aspect('equal')
        ax.set_xlim(min_val - 2, max_val + 2)
        ax.set_ylim(min_val - 2, max_val + 2)
    
        plt.tight_layout()
        plt.savefig('/Users/k.jones/Desktop/blogs/blog_posts/13_metallurgical_recovery_main.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"✓ Main recovery prediction visualization saved")
    logger.info(f"  Test R²: {test_r2:.3f}, MAE: {test_mae:.2f}%")

def create_feature_importance_plot(plot: bool = False):
    """
    Create feature importance bar chart.
    """
    logger.info("Generating feature importance visualization...")
    
    # Generate data
    X, y = generate_metallurgical_data(n_samples=800)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Train model
    logger.info("  Training model for feature importance...")
    model = GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # Get feature importance
    feature_names = ['Cu Grade\n(%)', 'Fe Content\n(%)', 'S Content\n(%)', 
                    'Grind Size\n(μm)', 'pH', 'Collector\nDose (g/t)', 'Frother\nDose (g/t)']
    importances = model.feature_importances_
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    
    # Create figure
    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
    
        colors = ['#FF4136' if i < 2 else '#2ECC40' if i < 4 else '#0074D9' 
                 for i in range(len(feature_names))]
    
        bars = ax.barh(range(len(feature_names)), importances[indices], 
                       color=[colors[i] for i in indices],
                       edgecolor='black', linewidth=1.5)
    
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
    
    # Add value labels
        for i, (bar, val) in enumerate(zip(bars, importances[indices])):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{val:.3f}',
                   ha='left', va='center', fontsize=9, 
                   fontweight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='none', alpha=0.8))
    
    # Apply minimalist style
        apply_minimalist_style_manual(ax)
    
        ax.set_xlabel('Feature Importance (Gain)', fontsize=11)
        ax.set_title('Feature Importance for Recovery Prediction', 
                     fontsize=13, fontweight='bold', loc='left', pad=20)
        ax.set_xlim(0, max(importances) * 1.25)
    
    # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='black', edgecolor='black', label='Ore Properties'),
            Patch(facecolor='black', edgecolor='black', label='Process Parameters'),
            Patch(facecolor='black', edgecolor='black', label='Reagent Dosage')
        ]
        ax.legend(handles=legend_elements, loc='lower right', frameon=False, fontsize=9)
    
        plt.tight_layout()
        plt.savefig('/Users/k.jones/Desktop/blogs/blog_posts/13_metallurgical_feature_importance.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info("✓ Feature importance visualization saved")

def main():
    """Generate all visualizations for Blog 13."""
    signalplot.apply(font_family='serif')
    logger.info("Blog 13: Metallurgical Recovery - Visualizations")
    logger.info()
    
    create_main_recovery_prediction_plot()
    create_feature_importance_plot()
    
    logger.info()
    logger.info("All visualizations generated successfully!")
    logger.info()
    logger.info("Files created:")
    logger.info("  - 13_metallurgical_recovery_main.png")
    logger.info("  - 13_metallurgical_feature_importance.png")

if __name__ == "__main__":
    main()

