"""
Creative Fatigue Analysis Package

A package for analyzing creative fatigue in digital advertising,
studying how CTR changes with repeated ad exposure.
"""

__version__ = "0.1.0"

# Data loading utilities
from .data_loader import (
    load_config,
    load_criteo_data,
    create_stratified_sample,
    temporal_train_test_split,
    inspect_data_structure,
)

# Feature engineering
from .feature_engineering import (
    compute_exposure_counts,
    compute_exposure_recency,
    compute_exposure_intensity,
    compute_temporal_features,
    compute_campaign_features,
    compute_user_features,
    create_all_features,
)

# Models
from .models import (
    BaselineModel,
    TimeAwareModel,
    DecayModel,
)

# Evaluation
from .evaluation import (
    compute_ctr_by_exposure,
    test_ctr_decline,
    find_fatigue_threshold,
    compare_decay_by_category,
    evaluate_models,
    plot_decay_curves,
    plot_decay_by_category,
)

# Utilities
from .utils import (
    ensure_dir,
    cyclical_encode,
)

# Bias Analysis
from .bias_analysis import (
    diagnose_selection_bias,
    compute_propensity_scores,
    apply_inverse_probability_weighting,
    create_matched_sample,
    compute_within_user_ctr_change,
    stratified_exposure_analysis,
)

