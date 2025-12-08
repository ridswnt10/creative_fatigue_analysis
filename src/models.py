"""
Model implementations for Creative Fatigue Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from scipy.optimize import curve_fit

# Optional imports for advanced models
# Use importlib to safely import optional dependencies
import importlib
import sys

XGBOOST_AVAILABLE = False
xgb = None
try:
    # Try to import xgboost using importlib for better error handling
    xgb_module = importlib.import_module('xgboost')
    xgb = xgb_module
    XGBOOST_AVAILABLE = True
except Exception:
    # Silently fail - will use fallback models
    XGBOOST_AVAILABLE = False
    xgb = None

LIGHTGBM_AVAILABLE = False
lgb = None
try:
    # Try to import lightgbm using importlib for better error handling
    lgb_module = importlib.import_module('lightgbm')
    lgb = lgb_module
    LIGHTGBM_AVAILABLE = True
except Exception:
    # Silently fail - will use fallback models
    LIGHTGBM_AVAILABLE = False
    lgb = None


class BaselineModel:
    """
    Baseline static model that predicts CTR without exposure/time features.
    """
    
    def __init__(self, model_type: str = 'xgboost', **kwargs):
        """
        Parameters:
        -----------
        model_type : str
            'xgboost', 'lightgbm', or 'logistic'
        **kwargs
            Additional parameters for the model
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        
        if model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not available. Install it with: pip install xgboost")
            self.model = xgb.XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=kwargs.get('random_state', 42),
                eval_metric='logloss'
            )
        elif model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM is not available. Install it with: pip install lightgbm")
            self.model = lgb.LGBMClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=kwargs.get('random_state', 42),
                verbose=-1
            )
        elif model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=kwargs.get('max_iter', 1000),
                random_state=kwargs.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, exclude_features: Optional[list] = None):
        """
        Train the model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target labels
        exclude_features : list, optional
            Features to exclude (e.g., exposure/time features)
        """
        X_train = X.copy()
        
        if exclude_features:
            X_train = X_train.drop(columns=[f for f in exclude_features if f in X_train.columns])
        
        self.feature_names = list(X_train.columns)
        
        # Remove any remaining non-numeric columns
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train = X_train[numeric_cols]
        self.feature_names = [f for f in self.feature_names if f in numeric_cols]
        
        self.model.fit(X_train, y)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict click probabilities."""
        X_pred = X.copy()
        
        # Use only features that were used in training
        if self.feature_names:
            available_features = [f for f in self.feature_names if f in X_pred.columns]
            X_pred = X_pred[available_features]
            
            # Ensure numeric only
            numeric_cols = X_pred.select_dtypes(include=[np.number]).columns
            X_pred = X_pred[numeric_cols]
        
        return self.model.predict_proba(X_pred)[:, 1]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if available."""
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            # For logistic regression, use absolute coefficients
            return dict(zip(self.feature_names, np.abs(self.model.coef_[0])))
        else:
            return {}


class TimeAwareModel(BaselineModel):
    """
    Time-aware model that includes exposure and temporal features.
    """
    
    def fit(self, X: pd.DataFrame, y: pd.Series, exclude_features: Optional[list] = None):
        """
        Train the model with all features including exposure/time.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix (should include exposure features)
        y : pd.Series
            Target labels
        exclude_features : list, optional
            Features to exclude (typically None for time-aware model)
        """
        super().fit(X, y, exclude_features=exclude_features)


class DecayModel:
    """
    Explicit decay model that fits decay curves and predicts CTR.
    """
    
    def __init__(self, decay_function: str = 'exponential'):
        """
        Parameters:
        -----------
        decay_function : str
            'exponential', 'power_law', or 'logistic'
        """
        self.decay_function = decay_function
        self.base_ctr_model = None
        self.decay_rate_model = None
        self.campaign_decay_params = {}
    
    @staticmethod
    def exponential_decay(x: np.ndarray, a: float, b: float) -> np.ndarray:
        """Exponential decay: CTR(x) = a * exp(-b*x)"""
        return a * np.exp(-b * x)
    
    @staticmethod
    def power_law_decay(x: np.ndarray, a: float, b: float) -> np.ndarray:
        """Power law decay: CTR(x) = a * x^(-b)"""
        return a * np.power(x, -b)
    
    @staticmethod
    def logistic_decay(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """Logistic decay: CTR(x) = a / (1 + exp(b*(x-c)))"""
        return a / (1 + np.exp(b * (x - c)))
    
    def fit_decay_curve(
        self,
        exposure_counts: np.ndarray,
        ctr_values: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Fit decay curve to exposure-CTR data.
        
        Parameters:
        -----------
        exposure_counts : np.ndarray
            Exposure counts
        ctr_values : np.ndarray
            CTR values at each exposure count
            
        Returns:
        --------
        Tuple of (fitted_values, parameters_dict)
        """
        # Remove zeros and invalid values
        mask = (exposure_counts > 0) & (ctr_values >= 0) & (ctr_values <= 1)
        x = exposure_counts[mask]
        y = ctr_values[mask]
        
        if len(x) < 3:
            return np.zeros_like(exposure_counts), {'a': 0, 'b': 0, 'error': 'insufficient_data'}
        
        try:
            if self.decay_function == 'exponential':
                # Initial guess: a = max CTR, b = small positive value
                p0 = [np.max(y), 0.1]
                popt, pcov = curve_fit(
                    self.exponential_decay,
                    x, y,
                    p0=p0,
                    bounds=([0, 0], [1, 10]),
                    maxfev=1000
                )
                fitted = self.exponential_decay(exposure_counts, *popt)
                params = {'a': popt[0], 'b': popt[1]}
                
            elif self.decay_function == 'power_law':
                p0 = [np.max(y), 0.1]
                popt, pcov = curve_fit(
                    self.power_law_decay,
                    x, y,
                    p0=p0,
                    bounds=([0, 0], [1, 10]),
                    maxfev=1000
                )
                fitted = self.power_law_decay(exposure_counts, *popt)
                params = {'a': popt[0], 'b': popt[1]}
                
            elif self.decay_function == 'logistic':
                p0 = [np.max(y), 0.1, np.median(x)]
                popt, pcov = curve_fit(
                    self.logistic_decay,
                    x, y,
                    p0=p0,
                    bounds=([0, 0, 0], [1, 10, np.max(x)]),
                    maxfev=1000
                )
                fitted = self.logistic_decay(exposure_counts, *popt)
                params = {'a': popt[0], 'b': popt[1], 'c': popt[2]}
            else:
                raise ValueError(f"Unknown decay_function: {self.decay_function}")
            
            # Calculate R² using the fitted values for the filtered data
            if self.decay_function == 'exponential':
                fitted_y = self.exponential_decay(x, params['a'], params['b'])
            elif self.decay_function == 'power_law':
                fitted_y = self.power_law_decay(x, params['a'], params['b'])
            elif self.decay_function == 'logistic':
                fitted_y = self.logistic_decay(x, params['a'], params['b'], params.get('c', 5))
            else:
                fitted_y = y
            
            ss_res = np.sum((y - fitted_y)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            params['r_squared'] = r_squared
            
            return fitted, params
            
        except Exception as e:
            return np.zeros_like(exposure_counts), {'error': str(e)}
    
    def _get_decay_func(self):
        """Get the decay function based on type."""
        if self.decay_function == 'exponential':
            return self.exponential_decay
        elif self.decay_function == 'power_law':
            return self.power_law_decay
        elif self.decay_function == 'logistic':
            return self.logistic_decay
    
    def fit_campaign_decay(
        self,
        df: pd.DataFrame,
        campaign_col: str = 'campaign_id',
        exposure_col: str = 'exposure_count',
        click_col: str = 'click'
    ):
        """
        Fit decay curves for each campaign.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with exposure counts and clicks
        campaign_col : str
            Campaign ID column
        exposure_col : str
            Exposure count column
        click_col : str
            Click label column
        """
        for campaign_id in df[campaign_col].unique():
            campaign_data = df[df[campaign_col] == campaign_id]
            
            # Compute CTR by exposure count
            ctr_by_exposure = campaign_data.groupby(exposure_col)[click_col].agg(['mean', 'count'])
            ctr_by_exposure = ctr_by_exposure[ctr_by_exposure['count'] >= 10]  # Minimum samples
            
            if len(ctr_by_exposure) < 3:
                continue
            
            exposure_counts = ctr_by_exposure.index.values
            ctr_values = ctr_by_exposure['mean'].values
            
            _, params = self.fit_decay_curve(exposure_counts, ctr_values)
            self.campaign_decay_params[campaign_id] = params
    
    def fit_meta_models(
        self,
        df: pd.DataFrame,
        campaign_features: pd.DataFrame,
        campaign_col: str = 'campaign_id'
    ):
        """
        Fit meta-models to predict decay parameters from campaign features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with exposure and clicks
        campaign_features : pd.DataFrame
            Features for each campaign
        campaign_col : str
            Campaign ID column
        """
        # First fit decay curves for each campaign
        self.fit_campaign_decay(df, campaign_col)
        
        # Extract decay parameters
        decay_data = []
        for campaign_id, params in self.campaign_decay_params.items():
            if 'error' not in params:
                row = {campaign_col: campaign_id, 'base_ctr': params['a'], 'decay_rate': params['b']}
                decay_data.append(row)
        
        if len(decay_data) == 0:
            print("Warning: No valid decay parameters found")
            return
        
        decay_df = pd.DataFrame(decay_data)
        
        # Merge with campaign features
        feature_df = campaign_features.merge(decay_df, on=campaign_col, how='inner')
        
        # Fit models to predict base CTR and decay rate
        X = feature_df.drop(columns=[campaign_col, 'base_ctr', 'decay_rate'])
        X = X.select_dtypes(include=[np.number])
        
        # Base CTR model
        y_base = feature_df['base_ctr']
        if XGBOOST_AVAILABLE and xgb is not None:
            try:
                self.base_ctr_model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
            except Exception:
                # Fallback if XGBoost fails at runtime
                from sklearn.ensemble import RandomForestRegressor
                self.base_ctr_model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
        else:
            # Fallback to sklearn RandomForestRegressor if XGBoost not available
            from sklearn.ensemble import RandomForestRegressor
            self.base_ctr_model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
        self.base_ctr_model.fit(X, y_base)
        
        # Decay rate model
        y_decay = feature_df['decay_rate']
        if XGBOOST_AVAILABLE and xgb is not None:
            try:
                self.decay_rate_model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
            except Exception:
                # Fallback if XGBoost fails at runtime
                from sklearn.ensemble import RandomForestRegressor
                self.decay_rate_model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
        else:
            # Fallback to sklearn RandomForestRegressor if XGBoost not available
            from sklearn.ensemble import RandomForestRegressor
            self.decay_rate_model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
        self.decay_rate_model.fit(X, y_decay)
    
    def predict(
        self,
        df: pd.DataFrame,
        campaign_features: Optional[pd.DataFrame] = None,
        campaign_col: str = 'campaign_id',
        exposure_col: str = 'exposure_count'
    ) -> np.ndarray:
        """
        Predict CTR using decay model.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data to predict on
        campaign_features : pd.DataFrame, optional
            Campaign features for meta-model prediction
        campaign_col : str
            Campaign ID column
        exposure_col : str
            Exposure count column
            
        Returns:
        --------
        np.ndarray
            Predicted CTR values
        """
        predictions = np.zeros(len(df))
        
        # Use enumerate to get position index, not DataFrame index
        for pos_idx, (_, row) in enumerate(df.iterrows()):
            campaign_id = row[campaign_col]
            exposure = row[exposure_col]
            
            # Get decay parameters
            if campaign_id in self.campaign_decay_params:
                params = self.campaign_decay_params[campaign_id]
            elif campaign_features is not None and self.base_ctr_model is not None:
                # Predict parameters from features
                campaign_feat = campaign_features[campaign_features[campaign_col] == campaign_id]
                if len(campaign_feat) > 0:
                    X = campaign_feat.drop(columns=[campaign_col])
                    X = X.select_dtypes(include=[np.number])
                    a = self.base_ctr_model.predict(X)[0]
                    b = self.decay_rate_model.predict(X)[0]
                    params = {'a': max(0, min(1, a)), 'b': max(0, b)}
                else:
                    params = {'a': 0.01, 'b': 0.1}  # Default
            else:
                params = {'a': 0.01, 'b': 0.1}  # Default
            
            if 'error' not in params and exposure > 0:
                if self.decay_function == 'exponential':
                    predictions[pos_idx] = self.exponential_decay(exposure, params['a'], params['b'])
                elif self.decay_function == 'power_law':
                    predictions[pos_idx] = self.power_law_decay(exposure, params['a'], params['b'])
                elif self.decay_function == 'logistic':
                    c = params.get('c', 5)
                    predictions[pos_idx] = self.logistic_decay(exposure, params['a'], params['b'], c)
            else:
                predictions[pos_idx] = 0.01  # Default low CTR
        
        return predictions

