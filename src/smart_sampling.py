"""
Smart Sampling for Creative Fatigue Analysis

This module provides sampling strategies that prioritize multi-exposure users
to enable proper fatigue analysis while controlling for selection bias.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import json
from datetime import datetime
from pathlib import Path


def identify_multi_exposure_users(
    df: pd.DataFrame,
    user_col: str = 'uid',
    campaign_col: str = 'campaign',
    min_exposures: int = 2
) -> Tuple[pd.Index, dict]:
    """
    Identify users with multiple exposures to the same campaign.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    user_col : str
        User ID column
    campaign_col : str
        Campaign ID column
    min_exposures : int
        Minimum exposures to qualify as multi-exposure
        
    Returns:
    --------
    Tuple of (multi_exposure_user_ids, statistics_dict)
    """
    # Count exposures per user-campaign pair
    user_campaign_counts = df.groupby([user_col, campaign_col]).size().reset_index(name='exposures')
    
    # Find users with at least one campaign with multiple exposures
    multi_exp_pairs = user_campaign_counts[user_campaign_counts['exposures'] >= min_exposures]
    multi_exp_users = multi_exp_pairs[user_col].unique()
    
    # Get max exposures per user
    user_max_exp = user_campaign_counts.groupby(user_col)['exposures'].max()
    
    stats = {
        'total_users': df[user_col].nunique(),
        'multi_exposure_users': len(multi_exp_users),
        'pct_multi_exposure': len(multi_exp_users) / df[user_col].nunique() * 100,
        'max_exposures_in_data': int(user_campaign_counts['exposures'].max()),
        'exposure_distribution': user_max_exp.value_counts().sort_index().head(10).to_dict()
    }
    
    return pd.Index(multi_exp_users), stats


def create_fatigue_optimized_sample(
    df: pd.DataFrame,
    user_col: str = 'uid',
    campaign_col: str = 'campaign',
    click_col: str = 'click',
    target_sample_size: int = 200000,
    multi_exp_ratio: float = 0.5,
    min_exposures: int = 2,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, dict]:
    """
    Create a sample optimized for fatigue analysis.
    
    This sampling strategy:
    1. Identifies multi-exposure users
    2. Samples heavily from multi-exposure users (default 50%)
    3. Samples remaining from single-exposure users
    4. Preserves click distribution within each group
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    user_col : str
        User ID column
    campaign_col : str
        Campaign ID column  
    click_col : str
        Click label column
    target_sample_size : int
        Desired total sample size
    multi_exp_ratio : float
        Proportion of sample from multi-exposure users (0.5 = 50%)
    min_exposures : int
        Minimum exposures to qualify as multi-exposure
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    Tuple of (sampled_df, metadata_dict)
    """
    np.random.seed(random_seed)
    
    print(f"Creating fatigue-optimized sample...")
    print(f"  Target size: {target_sample_size:,}")
    print(f"  Multi-exposure ratio: {multi_exp_ratio:.0%}")
    
    # Identify multi-exposure users
    multi_exp_users, user_stats = identify_multi_exposure_users(
        df, user_col, campaign_col, min_exposures
    )
    
    print(f"\n  Multi-exposure users in full data: {len(multi_exp_users):,} ({user_stats['pct_multi_exposure']:.1f}%)")
    
    # Split data
    multi_exp_df = df[df[user_col].isin(multi_exp_users)]
    single_exp_df = df[~df[user_col].isin(multi_exp_users)]
    
    print(f"  Multi-exposure records: {len(multi_exp_df):,}")
    print(f"  Single-exposure records: {len(single_exp_df):,}")
    
    # Calculate sample sizes
    multi_exp_target = int(target_sample_size * multi_exp_ratio)
    single_exp_target = target_sample_size - multi_exp_target
    
    # Sample from multi-exposure users (take all if less than target)
    if len(multi_exp_df) <= multi_exp_target:
        multi_exp_sample = multi_exp_df.copy()
        print(f"\n  Taking ALL {len(multi_exp_df):,} multi-exposure records")
    else:
        # Sample by user to keep user-campaign pairs together
        sample_ratio = multi_exp_target / len(multi_exp_df)
        sampled_users = np.random.choice(
            multi_exp_users, 
            size=int(len(multi_exp_users) * sample_ratio),
            replace=False
        )
        multi_exp_sample = multi_exp_df[multi_exp_df[user_col].isin(sampled_users)]
        print(f"\n  Sampled {len(multi_exp_sample):,} multi-exposure records")
    
    # Adjust single-exposure target based on actual multi-exposure sample
    single_exp_target = target_sample_size - len(multi_exp_sample)
    
    # Stratified sample from single-exposure users (preserve click distribution)
    if len(single_exp_df) <= single_exp_target:
        single_exp_sample = single_exp_df.copy()
    else:
        sample_ratio = single_exp_target / len(single_exp_df)
        single_exp_sample = single_exp_df.groupby(click_col, group_keys=False).apply(
            lambda x: x.sample(frac=sample_ratio, random_state=random_seed)
        )
    
    print(f"  Sampled {len(single_exp_sample):,} single-exposure records")
    
    # Combine samples
    sample = pd.concat([multi_exp_sample, single_exp_sample], ignore_index=True)
    
    # Compute sample statistics
    sample_multi_exp_users, sample_stats = identify_multi_exposure_users(
        sample, user_col, campaign_col, min_exposures
    )
    
    metadata = {
        'sample_date': datetime.now().isoformat(),
        'sampling_strategy': 'fatigue_optimized',
        'original_size': len(df),
        'sample_size': len(sample),
        'target_multi_exp_ratio': multi_exp_ratio,
        'actual_multi_exp_ratio': len(multi_exp_sample) / len(sample),
        'original_multi_exp_users': len(multi_exp_users),
        'sample_multi_exp_users': len(sample_multi_exp_users),
        'original_multi_exp_pct': user_stats['pct_multi_exposure'],
        'sample_multi_exp_pct': sample_stats['pct_multi_exposure'],
        'original_click_rate': df[click_col].mean(),
        'sample_click_rate': sample[click_col].mean(),
        'random_seed': random_seed,
        'exposure_distribution': sample_stats['exposure_distribution']
    }
    
    print(f"\n  Final sample: {len(sample):,} records")
    print(f"  Multi-exposure users: {len(sample_multi_exp_users):,} ({sample_stats['pct_multi_exposure']:.1f}%)")
    print(f"  Click rate: {sample[click_col].mean():.4f}")
    
    return sample, metadata


def save_optimized_sample(
    sample: pd.DataFrame,
    metadata: dict,
    output_dir: str = "data/samples",
    sample_name: str = "criteo_fatigue_optimized"
):
    """Save the optimized sample and its metadata."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save sample
    sample_file = output_path / f"{sample_name}.csv"
    sample.to_csv(sample_file, index=False)
    print(f"Sample saved to {sample_file}")
    
    # Save metadata
    metadata_file = output_path / f"{sample_name}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata saved to {metadata_file}")
    
    return str(sample_file), str(metadata_file)


