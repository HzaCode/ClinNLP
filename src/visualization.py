# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import random

# Import configuration and check optional dependencies
from .config import (AE_LABELS, DRUG_LABELS, TOP_N_AE_FREQ,
                     TOP_N_DRUGS_COOCCUR, TOP_N_AES_COOCCUR,
                     SHAP_MAX_DISPLAY, CALIBRATION_N_BINS,
                     PLOT_DIR, SHAP_AVAILABLE)

# Plot styling
sns.set_theme(style="whitegrid")

# Check SHAP availability for relevant plots
try:
    import shap
    from sklearn.calibration import CalibrationDisplay
    # SHAP_AVAILABLE already checked in modeling.py, assume flag is correct
except ImportError:
    # Update flag if import fails here too, though modeling.py check is primary
    SHAP_AVAILABLE = False


# --- NLP Data Visualizations ---

def plot_ae_frequency(nlp_results_df, save_plot=True):
    """Plots overall frequency of detected AEs (using normalized names)."""
    print("\nPlotting Figure 1: Normalized AE Frequency")
    if nlp_results_df.empty:
        print("No NLP results to plot AE frequency.")
        return
    # Use the 'entity_text' which is normalized, filter for AE types
    ae_counts = nlp_results_df[nlp_results_df['entity_type'].isin(AE_LABELS)]['entity_text'].value_counts()
    if ae_counts.empty:
        print("No affirmative AEs found to plot frequency.")
        return

    plt.figure(figsize=(12, 7))
    ae_counts.head(TOP_N_AE_FREQ).plot(kind='bar', color='skyblue')
    plt.title(f'Figure 1: Top {min(TOP_N_AE_FREQ, len(ae_counts))} Frequent Normalized AEs (Affirmative Mentions)')
    plt.ylabel('Number of Mentions')
    plt.xlabel('Normalized Adverse Event')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_plot:
        filepath = os.path.join(PLOT_DIR, 'fig1_ae_frequency.png')
        plt.savefig(filepath, dpi=300)
        print(f"Plot saved to {filepath}")
    plt.show()

def plot_ae_severity(nlp_results_df, save_plot=True):
    """Plots distribution of AE severity grades (for affirmative AEs)."""
    print("\nPlotting Figure 2: AE Severity Distribution (Affirmative Mentions)")
    if nlp_results_df.empty:
        print("No NLP results to plot AE severity.")
        return
    # Filter for AE types before checking severity
    severity_counts = nlp_results_df[
        nlp_results_df['entity_type'].isin(AE_LABELS) & nlp_results_df['severity_grade'].notna()
    ].copy() # Ensure it's a copy
    if severity_counts.empty:
        print("No severity grades found for affirmative AEs to plot.")
        return
    # Ensure grades are numeric for plotting
    severity_counts['severity_grade'] = pd.to_numeric(severity_counts['severity_grade'], errors='coerce')
    severity_counts.dropna(subset=['severity_grade'], inplace=True)
    if severity_counts.empty:
        print("No numeric severity grades found after conversion.")
        return

    grade_counts = severity_counts['severity_grade'].astype(float).value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    grade_counts.plot(kind='bar', color='lightcoral')
    plt.title('Figure 2: Distribution of AE Severity Grades (Affirmative Mentions)')
    plt.xlabel('Severity Grade')
    plt.ylabel('Number of Mentions')
    plt.xticks(rotation=0)
    plt.tight_layout()
    if save_plot:
        filepath = os.path.join(PLOT_DIR, 'fig2_ae_severity_distribution.png')
        plt.savefig(filepath, dpi=300)
        print(f"Plot saved to {filepath}")
    plt.show()

def plot_ae_by_drug(nlp_results_df, save_plot=True):
    """Plots AEs co-occurring with specific drugs in notes (uses normalized AE names)."""
    print("\nPlotting Figure 3: Normalized AEs Co-occurring with Drugs")
    if nlp_results_df.empty:
        print("No NLP results to plot AE-drug co-occurrence.")
        return

    # Filter for DRUG entities (use DRUG_LABELS from config)
    drug_mentions = nlp_results_df[nlp_results_df['entity_type'].isin(DRUG_LABELS)][['note_id', 'entity_text']].drop_duplicates()
    # Filter for AE entities and use normalized names
    ae_mentions = nlp_results_df[nlp_results_df['entity_type'].isin(AE_LABELS)][['note_id', 'entity_text']].drop_duplicates()

    if drug_mentions.empty or ae_mentions.empty:
        print("No drug or AE mentions found for co-occurrence plot.")
        return

    # Merge requires same dtype for note_id if necessary
    drug_mentions['note_id'] = drug_mentions['note_id'].astype(str)
    ae_mentions['note_id'] = ae_mentions['note_id'].astype(str)
    merged_mentions = pd.merge(drug_mentions, ae_mentions, on='note_id', suffixes=('_drug', '_ae'))

    if merged_mentions.empty:
        print("No drug-AE co-occurrences found in the same notes.")
        return

    # Create the co-occurrence matrix
    cooccurrence_counts = merged_mentions.groupby(['entity_text_drug', 'entity_text_ae']).size().unstack(fill_value=0)

    # Determine top drugs and AEs based on overall frequency
    top_drugs = nlp_results_df[nlp_results_df['entity_type'].isin(DRUG_LABELS)]['entity_text'].value_counts().head(TOP_N_DRUGS_COOCCUR).index
    top_aes = nlp_results_df[nlp_results_df['entity_type'].isin(AE_LABELS)]['entity_text'].value_counts().head(TOP_N_AES_COOCCUR).index

    # Filter the matrix for top drugs and AEs present in the matrix
    common_drugs = cooccurrence_counts.index.intersection(top_drugs)
    common_aes = cooccurrence_counts.columns.intersection(top_aes)

    if common_drugs.empty or common_aes.empty:
        print("No co-occurrences found between the selected top drugs and top AEs.")
        return

    cooccurrence_filtered = cooccurrence_counts.loc[common_drugs, common_aes]

    plt.figure(figsize=(max(10, len(common_aes)*0.8), max(6, len(common_drugs) * 0.6)))
    sns.heatmap(cooccurrence_filtered, cmap="viridis", annot=True, fmt="d", linewidths=.5, cbar=True)
    plt.title(f'Figure 3: Drug-AE Co-occurrence in Notes (Top {len(common_drugs)} Drugs, Top {len(common_aes)} Norm. AEs)')
    plt.xlabel('Normalized Adverse Event')
    plt.ylabel('Drug Mentioned in Same Note')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_plot:
        filepath = os.path.join(PLOT_DIR, 'fig3_drug_ae_cooccurrence.png')
        plt.savefig(filepath, dpi=300)
        print(f"Plot saved to {filepath}")
    plt.show()


def plot_patient_timeline(nlp_results_df, patient_id_to_plot, save_plot=True):
    """Plots a timeline of AEs and severity for a single patient (uses normalized names)."""
    print(f"\nPlotting Figure 4: Timeline for Patient {patient_id_to_plot}")
    if nlp_results_df.empty:
        print("No NLP results available for timeline plot.")
        return

    patient_data = nlp_results_df[nlp_results_df['patient_id'] == str(patient_id_to_plot)].sort_values('timestamp')

    # Filter for AE entities with a valid severity grade
    patient_aes = patient_data[
        patient_data['entity_type'].isin(AE_LABELS) &
        patient_data['severity_grade'].notna()
    ].copy()

    if patient_aes.empty:
        print(f"No AEs with severity found for Patient {patient_id_to_plot}.")
        return

    patient_aes['severity_grade'] = pd.to_numeric(patient_aes['severity_grade'], errors='coerce')
    patient_aes.dropna(subset=['severity_grade'], inplace=True)
    patient_aes['timestamp'] = pd.to_datetime(patient_aes['timestamp']) # Ensure datetime

    if patient_aes.empty:
        print(f"No numeric severity grades found for Patient {patient_id_to_plot} after conversion.")
        return

    plt.figure(figsize=(14, max(6, patient_aes['entity_text'].nunique() * 0.5)))
    scatter = sns.scatterplot(
        data=patient_aes,
        x='timestamp',
        y='entity_text', # Normalized AE name
        size='severity_grade',
        hue='severity_grade',
        sizes=(50, 350),
        palette='viridis_r', # Higher severity = brighter/yellower
        legend='full'
    )
    plt.title(f'Figure 4: Normalized AE Timeline for Patient {patient_id_to_plot}')
    plt.xlabel('Date')
    plt.ylabel('Normalized Adverse Event')
    # Improve legend placement
    handles, labels = scatter.get_legend_handles_labels()
    # Separate hue and size legends if they overlap too much
    # Find the index separating hue and size legends (often size comes after hue)
    size_legend_start_index = -1
    for i, label in enumerate(labels):
        try:
           # Check if the label can be converted to a number (likely severity grade)
           float(label)
        except ValueError:
            # If not a number, it might be the start of the size legend title
            if "severity_grade" in label.lower(): # Check for title
                 size_legend_start_index = i
                 break # Assume first non-numeric is the start

    if size_legend_start_index != -1:
         plt.legend(handles[1:size_legend_start_index], labels[1:size_legend_start_index], title='Severity Grade (Color)', bbox_to_anchor=(1.02, 1), loc='upper left')
    else: # Fallback if separation logic fails
         plt.legend(title='Severity Grade', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
    if save_plot:
        filepath = os.path.join(PLOT_DIR, f'fig4_timeline_patient_{patient_id_to_plot}.png')
        plt.savefig(filepath, dpi=300)
        print(f"Plot saved to {filepath}")
    plt.show()


# --- Model Performance Visualizations ---

def plot_roc_curve(metrics, save_plot=True):
    """Plots the ROC curve."""
    if not metrics or 'fpr' not in metrics or 'tpr' not in metrics:
        print("ROC data missing, skipping ROC plot.")
        return
    print("\nPlotting Figure 5: ROC Curve")
    fpr = metrics['fpr']
    tpr = metrics['tpr']
    roc_auc = metrics.get('roc_auc', None)
    auc_label = f'AUC = {roc_auc:.3f}' if roc_auc is not None else 'AUC not available'

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve ({auc_label})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Figure 5: Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.5)
    plt.tight_layout()
    if save_plot:
        filepath = os.path.join(PLOT_DIR, 'fig5_roc_curve.png')
        plt.savefig(filepath, dpi=300)
        print(f"Plot saved to {filepath}")
    plt.show()

def plot_precision_recall_curve(metrics, save_plot=True):
    """Plots the Precision-Recall curve."""
    if not metrics or 'precision' not in metrics or 'recall' not in metrics:
        print("Precision-Recall data missing, skipping PR plot.")
        return
    print("\nPlotting Figure 6: Precision-Recall Curve")
    recall = metrics['recall']
    precision = metrics['precision']
    avg_precision = metrics.get('avg_precision', None)
    ap_label = f'AUPRC = {avg_precision:.3f}' if avg_precision is not None else 'AUPRC not available'

    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, where='post', color='blue', alpha=0.7, label=ap_label)
    # Optional: Add baseline (positive class prevalence)
    # baseline = y_test.mean() # Requires y_test to be passed or available
    # plt.axhline(baseline, linestyle='--', color='grey', label=f'Baseline ({baseline:.2f})')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Figure 6: Precision-Recall Curve')
    plt.legend(loc="upper right") # Usually better location for PR curves
    plt.grid(alpha=0.5)
    plt.tight_layout()
    if save_plot:
        filepath = os.path.join(PLOT_DIR, 'fig6_precision_recall_curve.png')
        plt.savefig(filepath, dpi=300)
        print(f"Plot saved to {filepath}")
    plt.show()

def plot_calibration_curve(y_test, y_pred_proba, model, save_plot=True):
    """Plots the calibration curve."""
    if y_test is None or y_pred_proba is None or model is None:
         print("Missing data/model for calibration plot. Skipping.")
         return
    # Ensure CalibrationDisplay is available
    try:
        from sklearn.calibration import CalibrationDisplay
    except ImportError:
        print("Scikit-learn CalibrationDisplay not found (likely version issue). Skipping calibration plot.")
        return

    print("\nPlotting Figure 8: Calibration Curve")
    try:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        model_name = model.named_steps['classifier'].__class__.__name__ if hasattr(model, 'named_steps') else "Model"
        display = CalibrationDisplay.from_predictions(
            y_test,
            y_pred_proba,
            n_bins=CALIBRATION_N_BINS,
            name=model_name,
            ax=ax,
            strategy='uniform' # Or 'quantile'
        )
        ax.set_title('Figure 8: Calibration Curve (Reliability Diagram)')
        ax.grid(alpha=0.5)
        plt.tight_layout()
        if save_plot:
             filepath = os.path.join(PLOT_DIR, 'fig8_calibration_curve.png')
             plt.savefig(filepath, dpi=300)
             print(f"Plot saved to {filepath}")
        plt.show()
    except Exception as e:
        print(f"Error plotting calibration curve: {e}")


# --- Model Interpretation Visualization ---

def plot_shap_summary(shap_values, processed_data, feature_names=None, save_plot=True):
    """Plots SHAP summary plot (feature importance). Requires SHAP values and processed data."""
    if not SHAP_AVAILABLE:
        print("\nSHAP library not available. Skipping SHAP summary plot.")
        return
    if shap_values is None or processed_data is None:
        print("\nSHAP values or processed data missing. Skipping SHAP summary plot.")
        return

    print(f"\nPlotting Figure 7: SHAP Summary Plot (Top {SHAP_MAX_DISPLAY})")
    try:
        # Ensure processed_data is DataFrame with correct feature names if possible
        if not isinstance(processed_data, pd.DataFrame) and feature_names:
             processed_data_df = pd.DataFrame(processed_data, columns=feature_names)
        elif isinstance(processed_data, pd.DataFrame):
             processed_data_df = processed_data
             if feature_names is None:
                  feature_names = processed_data_df.columns.tolist() # Get names from DataFrame
        else:
             processed_data_df = processed_data # Use as is if no names available

        plt.figure() # Let SHAP manage figure creation/size
        shap.summary_plot(
            shap_values,
            processed_data_df, # Pass DataFrame or numpy array
            feature_names=feature_names, # Pass names if available
            max_display=SHAP_MAX_DISPLAY,
            show=False # Control show() outside
        )
        plt.title(f'Figure 7: SHAP Feature Importance (Impact on model output)')
        plt.tight_layout()
        if save_plot:
            filepath = os.path.join(PLOT_DIR, 'fig7_shap_summary.png')
            # Increase bbox_inches to prevent cutoff if tight_layout isn't enough
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        plt.show()

    except Exception as e:
        print(f"Error plotting SHAP summary: {e}")
        # Try plotting without feature names as a fallback
        try:
             print("  Attempting SHAP plot with default feature indices...")
             plt.figure()
             shap.summary_plot(shap_values, processed_data, max_display=SHAP_MAX_DISPLAY, show=False)
             plt.title(f'Figure 7: SHAP Feature Importance (Indices)')
             plt.tight_layout()
             if save_plot:
                  filepath = os.path.join(PLOT_DIR, 'fig7_shap_summary_indices.png')
                  plt.savefig(filepath, dpi=300, bbox_inches='tight')
                  print(f"Plot saved to {filepath}")
             plt.show()
        except Exception as e2:
            print(f"  Second attempt at SHAP plot failed: {e2}")

