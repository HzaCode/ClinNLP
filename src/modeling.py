# src/modeling.py
import pandas as pd
import numpy as np
import time
import joblib
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RandomizedSearchCV, StratifiedGroupKFold)
import lightgbm as lgb
from sklearn.metrics import (classification_report, roc_curve, auc, confusion_matrix,
                             precision_recall_curve, average_precision_score, brier_score_loss)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Import configuration and check optional dependencies
from .config import (RANDOM_STATE, TEST_SIZE, CV_FOLDS, N_ITER_SEARCH,
                     MODEL_DIR, SHAP_AVAILABLE) # Import flags too

# Check and update flags based on actual imports
try:
    import shap
    # SHAP_AVAILABLE = True # Already assumed True in config
except ImportError:
    print("Warning: shap not found. SHAP explanations will be skipped. Install with: pip install shap")
    SHAP_AVAILABLE = False # Update flag

def train_evaluate_advanced_model(feature_df):
    """Tunes LightGBM using RandomizedSearchCV, evaluates, and extracts SHAP values."""
    print("\n--- Training and Evaluating Advanced Model (Tuned LightGBM + SHAP) ---")
    if feature_df.empty or 'target' not in feature_df.columns or feature_df['target'].nunique() < 2:
         print("Insufficient data or variance in target variable for model training. Skipping.")
         # Return structure expected by run_pipeline.py on failure
         return None, {}, None, None, None, None, None

    # Separate features (X), target (y), and groups (patient_id for CV)
    y = feature_df['target']
    groups = feature_df['patient_id'] # Keep for GroupKFold
    X = feature_df.drop(['note_id', 'patient_id', 'target'], axis=1)

    # Identify feature types
    # Ensure 'cancer_type' is treated as category if it exists and was converted
    categorical_features_names = X.select_dtypes(include=['category', 'object']).columns.tolist()
    numerical_features_names = X.select_dtypes(include=np.number).columns.tolist()

    # --- Train/Test Split ---
    try:
        X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
            X, y, groups, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        print(f"Train/Val data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        target_dist_train = y_train.value_counts(normalize=True)
        target_dist_test = y_test.value_counts(normalize=True)
        print(f"Target distribution in Train/Val data:\n{target_dist_train}")
        print(f"Target distribution in Test data:\n{target_dist_test}")
        if len(target_dist_train) < 2 or len(target_dist_test) < 2:
            print("Training or Test data has only one class after split. Skipping model fitting.")
            return None, {}, None, None, None, y_test, None # Return y_test for potential analysis
    except ValueError as e:
         print(f"Error during train/test split (potentially too few samples per class/group): {e}. Skipping training.")
         return None, {}, None, None, None, None, None

    # --- Preprocessing ---
    numerical_transformer = StandardScaler()
    # Use passthrough for categorical features; LGBM handles them internally if type='category'
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features_names)],
        remainder='passthrough', # Keeps categorical features as they are
        verbose_feature_names_out=False) # Keep original names simpler
    preprocessor.set_output(transform="pandas") # Keep as pandas DataFrame

    # --- LightGBM Classifier ---
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1
    print(f"Calculated scale_pos_weight for training: {scale_pos_weight:.2f}")

    lgbm = lgb.LGBMClassifier(
        random_state=RANDOM_STATE,
        objective='binary',
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1
    )

    # --- Model Pipeline ---
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', lgbm)])

    # --- Hyperparameter Tuning (Randomized Search with Group K-Fold) ---
    print(f"\nStarting Randomized Search CV (Folds={CV_FOLDS}, Iterations={N_ITER_SEARCH})...")
    param_distributions = {
        'classifier__n_estimators': [100, 200, 300, 500],
        'classifier__learning_rate': [0.01, 0.02, 0.05, 0.1],
        'classifier__num_leaves': [15, 31, 61],
        'classifier__max_depth': [-1, 5, 10, 15],
        'classifier__reg_alpha': [0, 0.01, 0.1, 1.0],
        'classifier__reg_lambda': [0, 0.01, 0.1, 1.0],
        'classifier__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
    }
    cv = StratifiedGroupKFold(n_splits=CV_FOLDS)
    scoring = 'average_precision' # Good for imbalanced data

    search = RandomizedSearchCV(
        model_pipeline, param_distributions=param_distributions, n_iter=N_ITER_SEARCH,
        scoring=scoring, cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=1, refit=True
    )

    start_time_tuning = time.time()
    # --- Fit with Group Information and Categorical Feature Handling ---
    fit_params = {}
    # Get correct categorical feature names *after* potential preprocessing changes if any were made
    # Since we use 'passthrough', the names should remain the same.
    # Check if X_train still contains the categorical columns after potential split issues
    categorical_features_in_train = [f for f in categorical_features_names if f in X_train.columns]

    if categorical_features_in_train:
        # Need to tell LightGBM which features are categorical *within the pipeline step*
        fit_params['classifier__categorical_feature'] = categorical_features_in_train
        print(f"Passing categorical features to LGBM during search: {categorical_features_in_train}")
        # Ensure X_train dtypes are correct before fitting
        for cat_feat in categorical_features_in_train:
             if X_train[cat_feat].dtype.name != 'category':
                  print(f"  Warning: Feature '{cat_feat}' is not category dtype. Converting.")
                  X_train[cat_feat] = X_train[cat_feat].astype('category')
                  X_test[cat_feat] = X_test[cat_feat].astype('category') # Convert in test set too


    try:
        search.fit(X_train, y_train, groups=groups_train, **fit_params) # Pass groups!
    except Exception as e:
        print(f"\nError during RandomizedSearchCV fitting: {e}")
        # Try fitting without explicit categorical features if conversion/passing failed
        try:
             print("Retrying fit without explicitly passing categorical features...")
             search.fit(X_train, y_train, groups=groups_train)
        except Exception as e2:
             print(f"Second fit attempt failed: {e2}")
             print("Skipping model training.")
             return None, {}, None, None, None, y_test, None # Return y_test for analysis

    print(f"Hyperparameter tuning finished in {time.time() - start_time_tuning:.2f} seconds.")
    print(f"Best Score ({scoring}): {search.best_score_:.4f}")
    print("Best Parameters:")
    print(search.best_params_)

    # --- Evaluate Best Model on Test Set ---
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    print("\n--- Evaluation on Hold-Out Test Set ---")
    print("\nClassification Report (Test Set):")
    report = classification_report(y_test, y_pred)
    print(report)
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0) # Handle cases where matrix is not 2x2
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    # Calculate metrics
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)

    print(f"\nROC AUC Score: {roc_auc:.4f}")
    print(f"Average Precision (AUPRC): {avg_precision:.4f}")
    print(f"Brier Score: {brier:.4f}")

    metrics = {
        'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr,
        'avg_precision': avg_precision, 'precision': precision, 'recall': recall,
        'brier_score': brier, 'confusion_matrix': cm,
        'classification_report': report, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }

    # --- SHAP Explanations ---
    shap_values = None
    explainer = None
    X_test_processed_df = None # Initialize
    feature_names_processed = None # Initialize

    if SHAP_AVAILABLE:
        print("\nCalculating SHAP values for test set...")
        start_time_shap = time.time()
        try:
            best_lgbm = best_model.named_steps['classifier']
            preprocessor = best_model.named_steps['preprocessor']

            # Transform the test data *using the fitted preprocessor* from the pipeline
            X_test_processed = preprocessor.transform(X_test)
            # Ensure it's a DataFrame for SHAP TreeExplainer compatibility and feature names
            if not isinstance(X_test_processed, pd.DataFrame):
                 # Try to get feature names after transformation
                 try:
                      feature_names_processed = preprocessor.get_feature_names_out()
                 except Exception:
                     # Fallback if names can't be retrieved automatically
                     num_numeric_features = len(numerical_features_names)
                     numeric_feature_names = preprocessor.transformers_[0][2] # Get numeric names used
                     # Get remainder names (passthrough)
                     remainder_indices = [i for i, transformer in enumerate(preprocessor.transformers_) if transformer[0] == 'remainder']
                     if remainder_indices:
                         remainder_cols = X_test.columns[preprocessor.transformers_[remainder_indices[0]][2]].tolist()
                     else: # Should not happen with remainder='passthrough' unless no remainder cols exist
                         remainder_cols = [c for c in X_test.columns if c not in numeric_feature_names]
                     feature_names_processed = numeric_feature_names + remainder_cols

                 X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names_processed, index=X_test.index)
            else:
                 X_test_processed_df = X_test_processed # Already a DataFrame
                 feature_names_processed = X_test_processed_df.columns.tolist()


            print(f"  Shape of data passed to SHAP: {X_test_processed_df.shape}")
            print(f"  Feature names for SHAP: {feature_names_processed}")


            explainer = shap.TreeExplainer(best_lgbm)
            shap_values_tuple = explainer.shap_values(X_test_processed_df) # Pass DataFrame

            # For binary classification with LGBM, shap_values returns list [shap_class0, shap_class1]
            if isinstance(shap_values_tuple, list) and len(shap_values_tuple) == 2:
                 shap_values = shap_values_tuple[1] # Use values for the positive class (class 1)
                 print(f"  SHAP values calculated for positive class (shape: {shap_values.shape})")
            else: # Fallback if the output format is different
                 shap_values = shap_values_tuple
                 print(f"  SHAP values calculated (format might differ), shape: {np.shape(shap_values)}")


            print(f"SHAP calculation finished in {time.time() - start_time_shap:.2f} seconds.")

        except Exception as e:
            print(f"Error during SHAP value calculation: {e}")
            print("SHAP explanations will be skipped.")
            shap_values = None
            explainer = None # Ensure explainer is None if failed

    else:
        print("SHAP library not available. Skipping SHAP analysis.")


    return best_model, metrics, shap_values, explainer, X_test_processed_df, y_test, feature_names_processed

