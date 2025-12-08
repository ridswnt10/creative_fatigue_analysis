"""
Evaluation and visualization functions for Creative Fatigue Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
import warnings
warnings.filterwarnings('ignore')


def compute_ctr_by_exposure(
    df: pd.DataFrame,
    exposure_col: str = 'exposure_count',
    click_col: str = 'click',
    max_exposure: Optional[int] = None,
    min_samples: int = 100
) -> pd.DataFrame:
    """
    Compute CTR for each exposure count bucket.
    
    This is the core function for Research Question 1.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with exposure counts and clicks
    exposure_col : str
        Exposure count column
    click_col : str
        Click label column
    max_exposure : int, optional
        Maximum exposure count to analyze
    min_samples : int
        Minimum samples required per exposure bucket
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with exposure_count, ctr, clicks, impressions, confidence intervals
    """
    if max_exposure:
        df = df[df[exposure_col] <= max_exposure].copy()
    
    # Group by exposure count
    exposure_stats = df.groupby(exposure_col)[click_col].agg([
        ('clicks', 'sum'),
        ('impressions', 'count'),
        ('ctr', 'mean')
    ]).reset_index()
    
    # Filter by minimum samples
    exposure_stats = exposure_stats[exposure_stats['impressions'] >= min_samples]
    
    # Calculate confidence intervals (95% CI using normal approximation)
    n = exposure_stats['impressions']
    p = exposure_stats['ctr']
    se = np.sqrt(p * (1 - p) / n)
    z = 1.96  # 95% confidence
    exposure_stats['ctr_lower'] = np.maximum(0, p - z * se)
    exposure_stats['ctr_upper'] = np.minimum(1, p + z * se)
    
    return exposure_stats


def test_ctr_decline(
    df: pd.DataFrame,
    exposure_col: str = 'exposure_count',
    click_col: str = 'click',
    exposure_1: int = 1,
    exposure_2: int = 5
) -> Dict:
    """
    Test if CTR significantly declines between two exposure counts.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with exposure counts and clicks
    exposure_col : str
        Exposure count column
    click_col : str
        Click label column
    exposure_1 : int
        First exposure count to compare
    exposure_2 : int
        Second exposure count to compare
        
    Returns:
    --------
    dict
        Dictionary with test results
    """
    group1 = df[df[exposure_col] == exposure_1][click_col]
    group2 = df[df[exposure_col] == exposure_2][click_col]
    
    if len(group1) == 0 or len(group2) == 0:
        return {'error': 'Insufficient data'}
    
    # Create 2x2 contingency table for chi-square test
    # Rows: exposure groups, Columns: click (0/1)
    clicks1 = group1.sum()
    non_clicks1 = len(group1) - clicks1
    clicks2 = group2.sum()
    non_clicks2 = len(group2) - clicks2
    
    contingency = pd.DataFrame({
        'No Click': [non_clicks1, non_clicks2],
        'Click': [clicks1, clicks2]
    }, index=[f'Exposure {exposure_1}', f'Exposure {exposure_2}'])
    
    # Chi-square test for proportions
    chi2, p_value = stats.chi2_contingency(contingency)[:2]
    
    # Effect size (difference in proportions)
    ctr1 = group1.mean()
    ctr2 = group2.mean()
    effect_size = ctr1 - ctr2
    relative_decline = (ctr1 - ctr2) / ctr1 if ctr1 > 0 else 0
    
    return {
        'exposure_1': exposure_1,
        'exposure_2': exposure_2,
        'ctr_1': ctr1,
        'ctr_2': ctr2,
        'effect_size': effect_size,
        'relative_decline': relative_decline,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def find_fatigue_threshold(
    df: pd.DataFrame,
    exposure_col: str = 'exposure_count',
    click_col: str = 'click',
    decline_threshold: float = 0.20
) -> Dict:
    """
    Find exposure count where CTR drops by a certain percentage.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with exposure counts and clicks
    exposure_col : str
        Exposure count column
    click_col : str
        Click label column
    decline_threshold : float
        Percentage decline to detect (e.g., 0.20 for 20%)
        
    Returns:
    --------
    dict
        Dictionary with threshold information
    """
    ctr_by_exposure = compute_ctr_by_exposure(df, exposure_col, click_col)
    
    if len(ctr_by_exposure) == 0:
        return {'error': 'Insufficient data'}
    
    baseline_ctr = ctr_by_exposure[ctr_by_exposure[exposure_col] == 1]['ctr'].values
    if len(baseline_ctr) == 0:
        baseline_ctr = ctr_by_exposure.iloc[0]['ctr']
    else:
        baseline_ctr = baseline_ctr[0]
    
    # Find where CTR drops by threshold
    for _, row in ctr_by_exposure.iterrows():
        exposure = row[exposure_col]
        ctr = row['ctr']
        decline = (baseline_ctr - ctr) / baseline_ctr if baseline_ctr > 0 else 0
        
        if decline >= decline_threshold:
            return {
                'threshold_exposure': int(exposure),
                'baseline_ctr': baseline_ctr,
                'threshold_ctr': ctr,
                'decline_percentage': decline * 100
            }
    
    return {
        'threshold_exposure': None,
        'baseline_ctr': baseline_ctr,
        'message': f'CTR never dropped by {decline_threshold*100}%'
    }


def compare_decay_by_category(
    df: pd.DataFrame,
    category_col: str,
    exposure_col: str = 'exposure_count',
    click_col: str = 'click',
    max_exposure: Optional[int] = None,
    max_categories: Optional[int] = 10
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compare decay rates across categories.
    
    This is the core function for Research Question 2.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with categories, exposure counts, and clicks
    category_col : str
        Category column to group by
    exposure_col : str
        Exposure count column
    click_col : str
        Click label column
    max_exposure : int, optional
        Maximum exposure count to analyze
    max_categories : int, optional
        Maximum number of categories to process (prevents memory issues)
        
    Returns:
    --------
    Tuple of (decay_comparison_df, statistical_test_results)
    """
    # Filter by max_exposure early to reduce memory
    if max_exposure:
        df = df[df[exposure_col] <= max_exposure].copy()
    
    # Limit categories to prevent memory issues
    unique_categories = df[category_col].unique()
    if max_categories and len(unique_categories) > max_categories:
        # Keep top categories by sample size
        category_sizes = df.groupby(category_col).size().sort_values(ascending=False)
        top_categories = category_sizes.head(max_categories).index.tolist()
        df = df[df[category_col].isin(top_categories)].copy()
        unique_categories = top_categories
        print(f"Processing top {max_categories} categories by size (out of {len(category_sizes)} total)")
    
    # Compute CTR by exposure for each category
    category_decay_data = []
    total_categories = len(unique_categories)
    
    for idx, category in enumerate(unique_categories, 1):
        try:
            # Use query for faster filtering
            category_df = df.query(f"{category_col} == @category")
            
            # Skip if too small
            if len(category_df) < 100:
                print(f"  [{idx}/{total_categories}] {category}: Skipped (too few samples: {len(category_df)})")
                continue
            
            ctr_by_exposure = compute_ctr_by_exposure(
                category_df, exposure_col, click_col, min_samples=50
            )
            
            if len(ctr_by_exposure) < 3:
                print(f"  [{idx}/{total_categories}] {category}: Insufficient exposure levels ({len(ctr_by_exposure)} < 3)")
                continue
            
            # Check if CTR is actually decaying or increasing
            first_ctr = ctr_by_exposure.iloc[0]['ctr']
            last_ctr = ctr_by_exposure.iloc[-1]['ctr']
            ctr_trend = 'increasing' if last_ctr > first_ctr else 'decreasing' if last_ctr < first_ctr else 'flat'
            
            # Fit exponential decay
            from .models import DecayModel
            decay_model = DecayModel(decay_function='exponential')
            exposure_counts = ctr_by_exposure[exposure_col].values
            ctr_values = ctr_by_exposure['ctr'].values
            _, params = decay_model.fit_decay_curve(exposure_counts, ctr_values)
            
            if 'error' not in params:
                # Validate decay rate is reasonable
                decay_rate = params['b']
                r_squared = params.get('r_squared', 0)
                
                # If decay rate is extremely small (< 1e-6), it might indicate no decay or fitting issues
                if decay_rate < 1e-6:
                    print(f"  [{idx}/{total_categories}] {category}: Very small decay rate ({decay_rate:.2e}), CTR trend: {ctr_trend}")
                    if ctr_trend == 'increasing':
                        print(f"    Warning: CTR is increasing, decay model may not be appropriate")
                
                category_decay_data.append({
                    'category': category,
                    'base_ctr': params['a'],
                    'decay_rate': params['b'],
                    'r_squared': r_squared,
                    'n_samples': len(category_df),
                    'ctr_trend': ctr_trend,
                    'first_ctr': first_ctr,
                    'last_ctr': last_ctr
                })
            else:
                print(f"  [{idx}/{total_categories}] {category}: Decay fitting failed - {params.get('error', 'unknown error')}")
                
        except MemoryError:
            print(f"  [{idx}/{total_categories}] {category}: Memory error - skipping")
            continue
        except Exception as e:
            print(f"  [{idx}/{total_categories}] {category}: Error - {str(e)}")
            continue
    
    if len(category_decay_data) == 0:
        return pd.DataFrame(), {'error': 'No valid categories processed'}
    
    decay_comparison = pd.DataFrame(category_decay_data)
    
    # Statistical test: Kruskal-Wallis (simplified - no redundant computation)
    if len(decay_comparison) >= 2:
        try:
            # Use decay rates directly from the comparison dataframe
            groups = [decay_comparison[decay_comparison['category'] == cat]['decay_rate'].values
                     for cat in decay_comparison['category'].unique()]
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) >= 2:
                h_stat, p_value = stats.kruskal(*groups)
                test_results = {
                    'test': 'Kruskal-Wallis',
                    'h_statistic': h_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            else:
                test_results = {'error': 'Insufficient groups for testing'}
        except Exception as e:
            test_results = {'error': f'Statistical test failed: {str(e)}'}
    else:
        test_results = {'error': 'Insufficient categories for comparison'}
    
    return decay_comparison, test_results


def evaluate_models(
    models: Dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Evaluate multiple models and compare performance.
    
    This is the core function for Research Question 3.
    
    Parameters:
    -----------
    models : dict
        Dictionary of model_name: model_object
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    model_names : list, optional
        Names of models to evaluate
        
    Returns:
    --------
    pd.DataFrame
        Comparison of model performance metrics
    """
    if model_names is None:
        model_names = list(models.keys())
    
    results = []
    
    for name in model_names:
        if name not in models:
            continue
        
        model = models[name]
        
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            elif hasattr(model, 'predict'):
                y_pred_proba = model.predict(X_test)
            else:
                continue
            
            # Ensure probabilities are in correct format
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                y_pred_proba = y_pred_proba[:, 1]
            
            # Calculate metrics
            auc = roc_auc_score(y_test, y_pred_proba)
            logloss = log_loss(y_test, y_pred_proba)
            brier = brier_score_loss(y_test, y_pred_proba)
            
            results.append({
                'model': name,
                'auc_roc': auc,
                'log_loss': logloss,
                'brier_score': brier
            })
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            continue
    
    return pd.DataFrame(results)


def plot_decay_curves(
    ctr_by_exposure: pd.DataFrame,
    fitted_curves: Optional[Dict] = None,
    title: str = "CTR Decay with Exposure",
    save_path: Optional[str] = None
):
    """
    Plot CTR decay curves with confidence intervals.
    
    Parameters:
    -----------
    ctr_by_exposure : pd.DataFrame
        Output from compute_ctr_by_exposure
    fitted_curves : dict, optional
        Dictionary of fitted curve data
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    # Plot empirical data with confidence intervals
    plt.plot(
        ctr_by_exposure['exposure_count'],
        ctr_by_exposure['ctr'],
        'o-', label='Empirical CTR', linewidth=2, markersize=6
    )
    
    plt.fill_between(
        ctr_by_exposure['exposure_count'],
        ctr_by_exposure['ctr_lower'],
        ctr_by_exposure['ctr_upper'],
        alpha=0.3, label='95% CI'
    )
    
    # Plot fitted curves if provided
    if fitted_curves:
        for name, curve_data in fitted_curves.items():
            plt.plot(
                curve_data['exposure'],
                curve_data['ctr'],
                '--', label=name, linewidth=2
            )
    
    plt.xlabel('Exposure Count', fontsize=12)
    plt.ylabel('Click-Through Rate (CTR)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_decay_by_category(
    decay_comparison: pd.DataFrame,
    title: str = "Decay Rates by Category",
    save_path: Optional[str] = None
):
    """
    Plot decay rates comparison across categories.
    
    Parameters:
    -----------
    decay_comparison : pd.DataFrame
        Output from compare_decay_by_category
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot decay rates
    decay_comparison_sorted = decay_comparison.sort_values('decay_rate')
    
    # Filter out extremely small decay rates for better visualization
    # If decay rates are < 1e-6, they're essentially zero (no decay)
    display_decay = decay_comparison_sorted.copy()
    display_decay['decay_rate_display'] = display_decay['decay_rate'].apply(
        lambda x: max(x, 1e-6) if x > 0 else 1e-6  # Floor at 1e-6 for display
    )
    
    ax1.barh(
        decay_comparison_sorted['category'],
        decay_comparison_sorted['decay_rate'],
        alpha=0.7
    )
    ax1.set_xlabel('Decay Rate', fontsize=12)
    ax1.set_title('Decay Rate by Category', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add note if decay rates are very small
    if decay_comparison_sorted['decay_rate'].max() < 0.01:
        ax1.text(0.95, 0.05, 
                'Note: Decay rates are very small.\nCTR may not be decaying significantly.',
                transform=ax1.transAxes,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)
    
    # Plot base CTR
    decay_comparison_sorted = decay_comparison.sort_values('base_ctr')
    ax2.barh(
        decay_comparison_sorted['category'],
        decay_comparison_sorted['base_ctr'],
        alpha=0.7, color='orange'
    )
    ax2.set_xlabel('Base CTR', fontsize=12)
    ax2.set_title('Base CTR by Category', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()

