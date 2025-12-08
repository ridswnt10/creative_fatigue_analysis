"""
Feature engineering for Creative Fatigue Analysis
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from .utils import cyclical_encode


def compute_exposure_counts(
    df: pd.DataFrame,
    user_col: str = 'user_id',
    campaign_col: str = 'campaign_id',
    time_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    Compute exposure counts for each user-campaign pair.
    
    This is the core function for tracking how many times a user
    has seen a particular campaign.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with user, campaign, and timestamp columns
    user_col : str
        Name of user ID column
    campaign_col : str
        Name of campaign ID column
    time_col : str
        Name of timestamp column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with exposure_count column added
    """
    df = df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Sort by timestamp to ensure chronological order
    df_sorted = df.sort_values([user_col, campaign_col, time_col]).reset_index(drop=True)
    
    # Compute cumulative exposure count for each user-campaign pair
    df_sorted['exposure_count'] = (
        df_sorted.groupby([user_col, campaign_col])
        .cumcount() + 1
    )
    
    return df_sorted


def compute_time_windowed_exposures(
    df: pd.DataFrame,
    user_col: str = 'user_id',
    campaign_col: str = 'campaign_id',
    time_col: str = 'timestamp',
    windows_hours: List[int] = [24, 168, 720]  # 24h, 7d, 30d
) -> pd.DataFrame:
    """
    Compute exposure counts within time windows using optimized binary search.
    
    Optimized from O(n²) to O(n log n) using binary search for window boundaries.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    user_col : str
        Name of user ID column
    campaign_col : str
        Name of campaign ID column
    time_col : str
        Name of timestamp column
    windows_hours : List[int]
        List of time windows in hours
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with exposure_count_24h, exposure_count_7d, etc. columns
    """
    df = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    df_sorted = df.sort_values([user_col, campaign_col, time_col]).reset_index(drop=True)
    
    # Initialize window columns
    for window_hours in windows_hours:
        window_name = f'exposure_count_{window_hours}h'
        df_sorted[window_name] = 1  # At minimum, count self
    
    # Optimized approach: use binary search for window boundaries
    for window_hours in windows_hours:
        window_name = f'exposure_count_{window_hours}h'
        window_td = pd.Timedelta(hours=window_hours)
        
        def count_in_window_optimized(group):
            """Count exposures within time window using binary search (O(n log n))."""
            times = group[time_col].values
            n = len(times)
            
            if n == 0:
                return pd.Series([], dtype=int, index=group.index)
            if n == 1:
                return pd.Series([1], index=group.index)
            
            counts = np.ones(n, dtype=int)
            
            try:
                # Use timedelta64 for safer datetime arithmetic
                times_dt64 = times.astype('datetime64[ns]')
                window_ns = int(window_td.total_seconds() * 1e9)
                window_dt64 = np.timedelta64(window_ns, 'ns')
                
                for i in range(n):
                    current_time = times_dt64[i]
                    window_start = current_time - window_dt64
                    
                    # Binary search for the start of the window
                    # Find the leftmost index where times >= window_start
                    left_idx = np.searchsorted(times_dt64[:i+1], window_start, side='left')
                    
                    # Count from left_idx to i (inclusive)
                    counts[i] = i + 1 - left_idx
                    
            except (ValueError, OverflowError, TypeError) as e:
                # Fallback to simpler approach if binary search fails
                # This can happen with very large timestamps or edge cases
                for i in range(n):
                    current_time = times[i]
                    window_start = current_time - window_td
                    mask = (times <= current_time) & (times >= window_start)
                    counts[i] = mask.sum()
            
            return pd.Series(counts, index=group.index)
        
        # Apply to each user-campaign group
        try:
            df_sorted[window_name] = df_sorted.groupby(
                [user_col, campaign_col], group_keys=False
            ).apply(count_in_window_optimized)
        except Exception as e:
            print(f"Warning: Error computing {window_name}, using fallback method: {e}")
            # Fallback: simple vectorized approach (slower but more robust)
            def count_simple(group):
                times = group[time_col].values
                n = len(times)
                counts = np.ones(n, dtype=int)
                for i in range(n):
                    window_start = times[i] - window_td
                    counts[i] = np.sum((times[:i+1] >= window_start) & (times[:i+1] <= times[i]))
                return pd.Series(counts, index=group.index)
            
            df_sorted[window_name] = df_sorted.groupby(
                [user_col, campaign_col], group_keys=False
            ).apply(count_simple)
    
    return df_sorted


def compute_exposure_recency(
    df: pd.DataFrame,
    user_col: str = 'user_id',
    campaign_col: str = 'campaign_id',
    time_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    Compute recency features for exposures.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    user_col : str
        Name of user ID column
    campaign_col : str
        Name of campaign ID column
    time_col : str
        Name of timestamp column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with recency features added
    """
    df = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    df_sorted = df.sort_values([user_col, campaign_col, time_col]).reset_index(drop=True)
    
    # Time since first exposure
    first_exposure = df_sorted.groupby([user_col, campaign_col])[time_col].transform('min')
    df_sorted['hours_since_first_exposure'] = (
        (df_sorted[time_col] - first_exposure).dt.total_seconds() / 3600
    )
    
    # Time since last exposure (previous exposure for this user-campaign pair)
    df_sorted['hours_since_last_exposure'] = (
        df_sorted.groupby([user_col, campaign_col])[time_col]
        .diff()
        .dt.total_seconds() / 3600
    )
    df_sorted['hours_since_last_exposure'] = df_sorted['hours_since_last_exposure'].fillna(0)
    
    # Average time between exposures
    user_campaign_groups = df_sorted.groupby([user_col, campaign_col])
    avg_intervals = user_campaign_groups['hours_since_last_exposure'].transform('mean')
    df_sorted['avg_hours_between_exposures'] = avg_intervals
    
    return df_sorted


def compute_exposure_intensity(
    df: pd.DataFrame,
    user_col: str = 'user_id',
    campaign_col: str = 'campaign_id',
    time_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    Compute exposure intensity features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    user_col : str
        Name of user ID column
    campaign_col : str
        Name of campaign ID column
    time_col : str
        Name of timestamp column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with intensity features added
    """
    df = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Ensure we have exposure_count
    if 'exposure_count' not in df.columns:
        df = compute_exposure_counts(df, user_col, campaign_col, time_col)
    
    # Ensure we have hours_since_first_exposure
    if 'hours_since_first_exposure' not in df.columns:
        df = compute_exposure_recency(df, user_col, campaign_col, time_col)
    
    # Exposures per day
    df['days_since_first'] = df['hours_since_first_exposure'] / 24
    df['exposures_per_day'] = np.where(
        df['days_since_first'] > 0,
        df['exposure_count'] / df['days_since_first'],
        0
    )
    
    # Recent exposure rate (7d / 30d)
    if 'exposure_count_168h' in df.columns and 'exposure_count_720h' in df.columns:
        df['recent_exposure_rate'] = np.where(
            df['exposure_count_720h'] > 0,
            df['exposure_count_168h'] / df['exposure_count_720h'],
            0
        )
    else:
        df['recent_exposure_rate'] = 0
    
    return df


def compute_temporal_features(
    df: pd.DataFrame,
    time_col: str = 'timestamp',
    use_cyclical: bool = True
) -> pd.DataFrame:
    """
    Extract temporal features from timestamp.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    time_col : str
        Name of timestamp column
    use_cyclical : bool
        Whether to use cyclical encoding for hour/day
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with temporal features added
    """
    df = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Extract basic temporal features
    df['hour_of_day'] = df[time_col].dt.hour
    df['day_of_week'] = df[time_col].dt.dayofweek
    df['month'] = df[time_col].dt.month
    df['day_of_month'] = df[time_col].dt.day
    
    if use_cyclical:
        # Cyclical encoding for hour
        df['hour_sin'], df['hour_cos'] = cyclical_encode(df['hour_of_day'], 23)
        
        # Cyclical encoding for day of week
        df['dow_sin'], df['dow_cos'] = cyclical_encode(df['day_of_week'], 6)
        
        # Cyclical encoding for month
        df['month_sin'], df['month_cos'] = cyclical_encode(df['month'], 12)
    
    return df


def compute_campaign_features(
    df: pd.DataFrame,
    campaign_col: str = 'campaign_id',
    time_col: str = 'timestamp',
    click_col: str = 'click'
) -> pd.DataFrame:
    """
    Compute campaign-level aggregated features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    campaign_col : str
        Name of campaign ID column
    time_col : str
        Name of timestamp column
    click_col : str
        Name of click label column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with campaign features added
    """
    df = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Campaign-level aggregations
    campaign_stats = df.groupby(campaign_col).agg({
        click_col: ['count', 'sum', 'mean'],
        time_col: ['min', 'max']
    }).reset_index()
    
    campaign_stats.columns = [
        campaign_col,
        'campaign_total_impressions',
        'campaign_total_clicks',
        'campaign_overall_ctr',
        'campaign_first_impression',
        'campaign_last_impression'
    ]
    
    # Campaign age
    campaign_stats['campaign_age_days'] = (
        (campaign_stats['campaign_last_impression'] - 
         campaign_stats['campaign_first_impression'])
        .dt.total_seconds() / (3600 * 24)
    )
    
    # Merge back to original dataframe
    df = df.merge(campaign_stats, on=campaign_col, how='left')
    
    # Average user exposures per campaign
    if 'exposure_count' in df.columns:
        avg_exposures = df.groupby(campaign_col)['exposure_count'].mean().reset_index()
        avg_exposures.columns = [campaign_col, 'campaign_avg_user_exposures']
        df = df.merge(avg_exposures, on=campaign_col, how='left')
    else:
        df['campaign_avg_user_exposures'] = 0
    
    return df


def compute_user_features(
    df: pd.DataFrame,
    user_col: str = 'user_id',
    campaign_col: str = 'campaign_id',
    click_col: str = 'click'
) -> pd.DataFrame:
    """
    Compute user-level aggregated features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    user_col : str
        Name of user ID column
    campaign_col : str
        Name of campaign ID column
    click_col : str
        Name of click label column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with user features added
    """
    df = df.copy()
    
    # User overall CTR
    user_stats = df.groupby(user_col).agg({
        click_col: ['sum', 'count', 'mean']
    }).reset_index()
    user_stats.columns = [
        user_col,
        'user_total_clicks',
        'user_total_impressions',
        'user_overall_ctr'
    ]
    df = df.merge(user_stats, on=user_col, how='left')
    
    # User-campaign specific CTR
    user_campaign_stats = df.groupby([user_col, campaign_col]).agg({
        click_col: ['sum', 'count', 'mean']
    }).reset_index()
    user_campaign_stats.columns = [
        user_col,
        campaign_col,
        'user_campaign_clicks',
        'user_campaign_impressions',
        'user_campaign_ctr'
    ]
    df = df.merge(user_campaign_stats, on=[user_col, campaign_col], how='left')
    
    # Number of distinct campaigns user has seen
    user_campaigns = df.groupby(user_col)[campaign_col].nunique().reset_index()
    user_campaigns.columns = [user_col, 'user_total_campaigns_seen']
    df = df.merge(user_campaigns, on=user_col, how='left')
    
    # Average exposure count across all campaigns
    if 'exposure_count' in df.columns:
        user_avg_exposures = df.groupby(user_col)['exposure_count'].mean().reset_index()
        user_avg_exposures.columns = [user_col, 'user_avg_exposure_count']
        df = df.merge(user_avg_exposures, on=user_col, how='left')
    else:
        df['user_avg_exposure_count'] = 0
    
    return df


def create_all_features(
    df: pd.DataFrame,
    user_col: str = 'user_id',
    campaign_col: str = 'campaign_id',
    time_col: str = 'timestamp',
    click_col: str = 'click',
    windows_hours: List[int] = [24, 168, 720],
    parallel: bool = False,  # Disabled by default to avoid memory issues
    n_jobs: int = 4
) -> pd.DataFrame:
    """
    Create all features in the correct order with optional parallelization.
    
    Dependency graph:
    1. exposure_counts (must be first)
    2. [time_windowed, recency, temporal, campaign, user] can run in parallel
    3. intensity (depends on recency)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    user_col : str
        User ID column name
    campaign_col : str
        Campaign ID column name
    time_col : str
        Timestamp column name
    click_col : str
        Click label column name
    windows_hours : List[int]
        Time windows for windowed exposures
    parallel : bool
        Whether to use parallel processing for independent operations
    n_jobs : int
        Number of parallel jobs (if parallel=True)
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with all features added
    """
    # Step 1: Must run first (dependency for everything else)
    print("Computing exposure counts...")
    df = compute_exposure_counts(df, user_col, campaign_col, time_col)
    
    if parallel and n_jobs > 1:
        # Step 2: Run dependent operations sequentially
        print("Computing time-windowed exposures...")
        df = compute_time_windowed_exposures(
            df, user_col, campaign_col, time_col, windows_hours
        )
        
        print("Computing exposure recency...")
        df = compute_exposure_recency(df, user_col, campaign_col, time_col)
        
        # Step 3: Run independent operations in parallel
        print(f"Computing independent features in parallel ({n_jobs} workers)...")
        
        def compute_temporal():
            return compute_temporal_features(df.copy(), time_col)
        
        def compute_campaign():
            return compute_campaign_features(df.copy(), campaign_col, time_col, click_col)
        
        def compute_user():
            return compute_user_features(df.copy(), user_col, campaign_col, click_col)
        
        # Execute independent operations in parallel
        results = {}
        try:
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                futures = {
                    executor.submit(compute_temporal): 'temporal',
                    executor.submit(compute_campaign): 'campaign',
                    executor.submit(compute_user): 'user'
                }
                
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        result_df = future.result()
                        # Extract only new columns
                        new_cols = [c for c in result_df.columns if c not in df.columns]
                        if new_cols:
                            results[name] = result_df[new_cols].copy()
                            print(f"  ✓ {name} completed ({len(new_cols)} new columns)")
                        else:
                            print(f"  ⚠ {name} completed (no new columns)")
                    except Exception as e:
                        print(f"  ✗ {name} failed: {e}")
                        import traceback
                        traceback.print_exc()
                        raise
        
            # Merge results back into df (ensure index alignment)
            for name in ['temporal', 'campaign', 'user']:
                if name in results and len(results[name].columns) > 0:
                    # Reset index to ensure alignment
                    df = df.reset_index(drop=True)
                    results[name] = results[name].reset_index(drop=True)
                    df = pd.concat([df, results[name]], axis=1)
                    
        except MemoryError:
            print("  ⚠ Memory error in parallel processing, falling back to sequential")
            # Fall back to sequential
            df = compute_temporal_features(df, time_col)
            df = compute_campaign_features(df, campaign_col, time_col, click_col)
            df = compute_user_features(df, user_col, campaign_col, click_col)
        
        # Step 4: Compute intensity (depends on recency)
        print("Computing exposure intensity...")
        df = compute_exposure_intensity(df, user_col, campaign_col, time_col)
        
    else:
        # Sequential execution (original approach)
        print("Computing time-windowed exposures...")
        df = compute_time_windowed_exposures(
            df, user_col, campaign_col, time_col, windows_hours
        )
        
        print("Computing exposure recency...")
        df = compute_exposure_recency(df, user_col, campaign_col, time_col)
        
        print("Computing exposure intensity...")
        df = compute_exposure_intensity(df, user_col, campaign_col, time_col)
        
        print("Computing temporal features...")
        df = compute_temporal_features(df, time_col)
        
        print("Computing campaign features...")
        df = compute_campaign_features(df, campaign_col, time_col, click_col)
        
        print("Computing user features...")
        df = compute_user_features(df, user_col, campaign_col, click_col)
    
    print(f"Feature engineering complete. Final shape: {df.shape}")
    return df

