# run_pipeline.py
import time
import pandas as pd
import random
import warnings
import os
import sys # Keep sys import for potential debugging

# --- Check if src is in path (useful for debugging if run from wrong dir) ---
current_script_path = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_script_path, 'src')
if src_path not in sys.path and current_script_path not in sys.path:
     # If run_pipeline.py is in root, current_script_path IS the root.
     # We expect 'src' to be findable relative to the root.
     print(f"Attempting to add script directory to sys.path: {current_script_path}")
     sys.path.insert(0, current_script_path)
# --------------------------------------------------------------------------


# Import pipeline components from src
try:
    from src.config import (PATIENT_DATA_PATH, NOTES_DATA_PATH, AE_EVENTS_PATH,
                            AE_LABELS, PLOT_DIR, RANDOM_STATE) # Add other needed configs
    from src.data_processing import load_patient_data, load_notes_data, determine_ae_outcomes
    from src.nlp_extraction import extract_entities_advanced_nlp
    from src.feature_engineering import create_advanced_features
    from src.modeling import train_evaluate_advanced_model
    from src.visualization import (plot_ae_frequency, plot_ae_severity, plot_ae_by_drug,
                                   plot_patient_timeline, plot_roc_curve, plot_precision_recall_curve,
                                   plot_calibration_curve, plot_shap_summary)
    from src.utils import save_model, save_results # Optional utilities
except ImportError as e:
     print(f"FATAL ERROR: Cannot import necessary modules from 'src'.")
     print(f"Ensure 'run_pipeline.py' is in the project root directory")
     print(f"and the 'src' directory exists and contains the required .py files.")
     print(f"ImportError: {e}")
     sys.exit(1) # Exit if core modules can't be loaded


# Ignore specific warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# Filter specific LightGBM warning about categorical features if needed
warnings.filterwarnings("ignore", message="Using categorical_feature in Dataset.")


def main():
    """Runs the entire AE prediction pipeline."""
    print("--- Starting Enhanced AE Monitoring Analysis Pipeline ---")
    pipeline_start_time = time.time()

    # --- Step 1: Load Data & Determine Outcomes ---
    print("\n=== Step 1: Loading Data & Determining Outcomes ===")
    try:
        # Paths are used directly from the imported config
        patients_df = load_patient_data(PATIENT_DATA_PATH)
        notes_df = load_notes_data(NOTES_DATA_PATH)
        notes_with_outcomes_df = determine_ae_outcomes(notes_df, AE_EVENTS_PATH)
        print(f"Target distribution ('severe_ae_in_period') in full dataset:\n{notes_with_outcomes_df['severe_ae_in_period'].value_counts(normalize=True, dropna=False)}")
    except FileNotFoundError:
        print("\nFATAL ERROR: One or more input files not found. Please check paths in src/config.py.")
        return # Exit gracefully
    except ValueError as ve:
        print(f"\nFATAL ERROR during data loading (missing columns or invalid data): {ve}")
        return
    except Exception as e:
        print(f"\nFATAL ERROR during data loading or outcome determination: {e}")
        return

    # --- Step 2: Extract Entities using Advanced NLP ---
    print("\n=== Step 2: Extracting Entities (NLP) ===")
    nlp_results_df = pd.DataFrame() # Initialize
    if not notes_with_outcomes_df.empty:
        try:
            nlp_results_df = extract_entities_advanced_nlp(notes_with_outcomes_df) # Pass relevant subset
            if not nlp_results_df.empty:
                print(f"Extracted {len(nlp_results_df)} affirmative entities.")
                print("\nSample Normalized NLP Extraction Output (AEs with Severity):")
                sample = nlp_results_df[nlp_results_df['entity_type'].isin(AE_LABELS) & nlp_results_df['severity_grade'].notna()]
                print(sample.head())
            else:
                print("Warning: NLP extraction yielded no affirmative entities.")
        except Exception as e:
            print(f"ERROR during NLP extraction: {e}. Proceeding without NLP results.")
            nlp_results_df = pd.DataFrame() # Ensure it's empty on error
    else:
        print("No notes data available for NLP processing.")


    # --- Step 3: Advanced Feature Engineering ---
    print("\n=== Step 3: Engineering Features ===")
    feature_df = pd.DataFrame() # Initialize
    if not notes_with_outcomes_df.empty:
        try:
            feature_df = create_advanced_features(
                nlp_results_df, # Pass potentially empty df
                notes_with_outcomes_df,
                patients_df
            )
            if not feature_df.empty:
                 print(f"Created {feature_df.shape[1]-3} features for {feature_df.shape[0]} instances.") # Exclude id, patient_id, target
                 print(f"Target distribution in final feature set:\n{feature_df['target'].value_counts(normalize=True, dropna=False)}")
            else:
                 print("Feature engineering resulted in an empty DataFrame.")
        except Exception as e:
            print(f"ERROR during Feature Engineering: {e}. Cannot proceed to modeling.")
            feature_df = pd.DataFrame() # Ensure empty on error
    else:
        print("Skipping feature engineering due to missing notes data.")


    # --- Step 4: Train and Evaluate Advanced Model ---
    print("\n=== Step 4: Training and Evaluating Model ===")
    model = None
    metrics = {}
    shap_values = None
    explainer = None
    X_test_processed = None
    y_test_final = None
    feature_names_final = None

    can_train = False
    if not feature_df.empty and 'target' in feature_df.columns and feature_df['target'].nunique() > 1:
        print("Feature set seems valid for training.")
        can_train = True
    elif feature_df.empty:
        print("Feature DataFrame is empty. Skipping model training.")
    elif 'target' not in feature_df.columns:
         print("Target column missing in feature DataFrame. Skipping model training.")
    else: # Target has only one class
        print("Target variable has only one class. Cannot train classification model.")

    if can_train:
        try:
            model, metrics, shap_values, explainer, X_test_processed, y_test_final, feature_names_final = train_evaluate_advanced_model(
                feature_df
            )
            if model:
                 print("\nModel training and evaluation completed.")
                 # Save the best model and results
                 save_model(model, model_name_prefix="lgbm_ae_predictor")
                 save_results(metrics, results_name_prefix="lgbm_eval_metrics")
            else:
                 print("\nModel training failed or was skipped during the process.")

        except Exception as e:
            print(f"ERROR during Model Training/Evaluation: {e}")


    # --- Step 5: Visualization ---
    print("\n=== Step 5: Generating Visualizations ===")
    os.makedirs(PLOT_DIR, exist_ok=True) # Ensure plot dir exists

    # NLP Data Visualizations
    if not nlp_results_df.empty:
        print("\nGenerating NLP data visualizations...")
        try: plot_ae_frequency(nlp_results_df)
        except Exception as e: print(f"Error plotting AE frequency: {e}")
        try: plot_ae_severity(nlp_results_df)
        except Exception as e: print(f"Error plotting AE severity: {e}")
        try: plot_ae_by_drug(nlp_results_df)
        except Exception as e: print(f"Error plotting AE by drug: {e}")
        try:
             patients_with_aes = nlp_results_df[
                 nlp_results_df['entity_type'].isin(AE_LABELS) & nlp_results_df['severity_grade'].notna()
             ]['patient_id'].unique()
             if len(patients_with_aes) > 0:
                  random.seed(RANDOM_STATE)
                  example_patient_id = random.choice(patients_with_aes)
                  print(f"Attempting timeline plot for example Patient ID: {example_patient_id}")
                  plot_patient_timeline(nlp_results_df, patient_id_to_plot=example_patient_id)
             else:
                  print("\nNo patients with detected AEs+Severity found for timeline plot example.")
        except Exception as e: print(f"Error plotting patient timeline: {e}")
    else:
        print("\nSkipping NLP-based visualizations as NLP results are empty or NLP step failed.")

    # Model Performance Visualizations
    if model and metrics and y_test_final is not None: # Check y_test_final exists
        print("\nGenerating model performance plots...")
        y_pred_proba_final = None # Initialize
        try:
            # Recalculate probabilities if X_test_processed is available
            if X_test_processed is not None:
                y_pred_proba_final = model.predict_proba(X_test_processed)[:, 1]
            else:
                print("Warning: X_test_processed not available, cannot generate probabilities for calibration plot.")

            try: plot_roc_curve(metrics)
            except Exception as e: print(f"Error plotting ROC curve: {e}")
            try: plot_precision_recall_curve(metrics)
            except Exception as e: print(f"Error plotting PR curve: {e}")
            try:
                 if y_pred_proba_final is not None:
                      plot_calibration_curve(y_test_final, y_pred_proba_final, model)
                 else:
                      print("Skipping calibration plot as predicted probabilities are unavailable.")
            except Exception as e: print(f"Error plotting calibration curve: {e}")

        except Exception as e:
            print(f"An error occurred during model visualization setup: {e}")
    else:
         print("\nNo trained model, metrics, or test data available for performance plotting.")

    # SHAP Plot
    if shap_values is not None and X_test_processed is not None:
        print("\nGenerating SHAP summary plot...")
        try: plot_shap_summary(shap_values, X_test_processed, feature_names=feature_names_final)
        except Exception as e: print(f"Error plotting SHAP summary: {e}")
    else:
         print("\nSHAP values or processed data not available, skipping SHAP plot.")

    # --- Final Comments ---
    pipeline_end_time = time.time()
    print(f"\n--- Enhanced Analysis Pipeline Finished in {pipeline_end_time - pipeline_start_time:.2f} seconds ---")
    print("\n** Next Steps & Considerations **")
    # (Add relevant comments here)


if __name__ == "__main__":
    main()
