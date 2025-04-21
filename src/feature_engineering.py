# src/feature_engineering.py
import pandas as pd
import numpy as np
from datetime import timedelta
import time

# Import configuration
from .config import LOOKBACK_DAYS, RECENT_WINDOW_DAYS, DRUGS, AE_LABELS

def create_advanced_features(nlp_results_df, notes_with_outcomes_df, patients_df):
    """Creates features using normalized, affirmative NLP results and temporal dynamics."""
    print("\nEngineering advanced features...")
    start_time = time.time()

    # --- Data Preparation ---
    if nlp_results_df.empty:
        print("Warning: NLP results are empty. Returning features based only on baseline and notes.")
        # Simplified feature creation if no NLP data
        feature_list = []
        patients_df_indexed = patients_df.set_index('patient_id')
        notes_with_outcomes_df = notes_with_outcomes_df.sort_values(['patient_id', 'timestamp'])
        notes_with_outcomes_df['prev_timestamp'] = notes_with_outcomes_df.groupby('patient_id')['timestamp'].shift(1)

        for index, note in notes_with_outcomes_df.iterrows():
            patient_id = note['patient_id']
            current_time = note['timestamp']
            prev_note_time = note['prev_timestamp']
            features = {'note_id': note['note_id'], 'patient_id': patient_id, 'target': note['severe_ae_in_period']}

            try:
                patient_info = patients_df_indexed.loc[patient_id]
                features['age'] = patient_info['age']
                features['baseline_risk_score'] = patient_info.get('baseline_risk_score', -1)
                features['cancer_type'] = patient_info['cancer_type']
                regimen = patient_info['treatment_regimen']
                features['n_drugs_regimen'] = len(regimen) if isinstance(regimen, list) else 0
                for drug in DRUGS: features[f'has_{drug}'] = 1 if isinstance(regimen, list) and drug in regimen else 0
            except KeyError: continue

            if pd.notna(prev_note_time): features['days_since_last_note'] = (current_time - prev_note_time).days
            else: features['days_since_last_note'] = -1

            # Add default NLP-derived features as 0 or -1
            nlp_feature_names = [ 'n_ae_mentions_lw', 'n_severe_ae_lw', 'max_severity_lw',
                                'n_ae_mentions_rw', 'n_severe_ae_rw', 'max_severity_rw',
                                'ratio_severe_ae_rw', 'days_since_last_ae', 'days_since_last_severe_ae',
                                'severity_increased_recently', 'count_NauseaVomiting_rw', 'count_grade4plus_ae_lw'] # Add others as needed
            for feat in nlp_feature_names:
                if 'days_since' in feat: features[feat] = -1
                else: features[feat] = 0

            feature_list.append(features)

        feature_df = pd.DataFrame(feature_list)
        feature_df.fillna(0, inplace=True) # Fill any remaining NaNs
        print(f"Feature engineering finished (limited due to empty NLP results) in {time.time() - start_time:.2f} seconds.")
        return feature_df


    # --- Full Feature Engineering with NLP Data ---
    # Ensure DataFrames are sorted correctly and types are right
    nlp_results_df = nlp_results_df.sort_values(['patient_id', 'timestamp'])
    notes_with_outcomes_df = notes_with_outcomes_df.sort_values(['patient_id', 'timestamp'])
    patients_df['patient_id'] = patients_df['patient_id'].astype(str)
    nlp_results_df['patient_id'] = nlp_results_df['patient_id'].astype(str)
    notes_with_outcomes_df['patient_id'] = notes_with_outcomes_df['patient_id'].astype(str)
    notes_with_outcomes_df['note_id'] = notes_with_outcomes_df['note_id'].astype(str)

    patients_df_indexed = patients_df.set_index('patient_id') # Index patients for faster lookup

    feature_list = []

    # Pre-calculate NLP aggregations
    nlp_results_df['is_ae'] = nlp_results_df['entity_type'].isin(AE_LABELS).astype(int)
    # Ensure severity_grade is numeric before comparison
    nlp_results_df['severity_grade'] = pd.to_numeric(nlp_results_df['severity_grade'], errors='coerce')
    nlp_results_df['is_severe'] = ((nlp_results_df['severity_grade'] >= 3) & nlp_results_df['is_ae']).fillna(0).astype(int) # Handle NaN grades

    # Timestamps of last events per patient (ensure timestamp is datetime)
    nlp_results_df['timestamp'] = pd.to_datetime(nlp_results_df['timestamp'])
    last_event_time = nlp_results_df.groupby('patient_id')['timestamp'].last().to_dict()
    last_ae_time = nlp_results_df[nlp_results_df['is_ae'] == 1].groupby('patient_id')['timestamp'].last().to_dict()
    last_severe_ae_time = nlp_results_df[nlp_results_df['is_severe'] == 1].groupby('patient_id')['timestamp'].last().to_dict()

    # Efficiently get previous note timestamps per patient
    notes_with_outcomes_df['timestamp'] = pd.to_datetime(notes_with_outcomes_df['timestamp'])
    notes_with_outcomes_df['prev_timestamp'] = notes_with_outcomes_df.groupby('patient_id')['timestamp'].shift(1)

    # Group NLP results by patient ID for faster filtering within the loop
    nlp_grouped_by_patient = nlp_results_df.groupby('patient_id')
    patient_nlp_map = {pid: group for pid, group in nlp_grouped_by_patient}

    total_notes_to_feature = len(notes_with_outcomes_df)
    processed_feature_count = 0

    # Iterate through notes that have an outcome defined
    for index, note in notes_with_outcomes_df.iterrows():
        current_time = note['timestamp']
        patient_id = note['patient_id']
        note_id = note['note_id']
        target = note['severe_ae_in_period']
        prev_note_time = note['prev_timestamp'] # Timestamp of the previous note

        features = {'note_id': note_id, 'patient_id': patient_id, 'target': target}

        # --- Baseline Patient Features ---
        try:
            patient_info = patients_df_indexed.loc[patient_id]
            features['age'] = patient_info['age']
            features['baseline_risk_score'] = patient_info.get('baseline_risk_score', -1)
            features['cancer_type'] = patient_info['cancer_type']
            regimen = patient_info['treatment_regimen']
            features['n_drugs_regimen'] = len(regimen) if isinstance(regimen, list) else 0
            for drug in DRUGS: # Use the global DRUGS list from config
                 features[f'has_{drug}'] = 1 if isinstance(regimen, list) and drug in regimen else 0
        except KeyError:
             # print(f"Warning: Patient ID {patient_id} not found in patient baseline data. Skipping features for note {note_id}.")
             # Handle missing baseline data - maybe add placeholder features? Or skip note?
             # For now, add placeholders
             features['age'] = -1
             features['baseline_risk_score'] = -1
             features['cancer_type'] = 'Unknown'
             features['n_drugs_regimen'] = 0
             for drug in DRUGS: features[f'has_{drug}'] = 0
             # Decide if you want to continue or skip this note
             # continue # Option to skip note if baseline missing

        # Get NLP results for this patient efficiently
        patient_nlp_history = patient_nlp_map.get(patient_id)
        if patient_nlp_history is None or patient_nlp_history.empty:
             relevant_nlp_full = pd.DataFrame()
             relevant_nlp_recent = pd.DataFrame()
        else:
            lookback_start_time = current_time - timedelta(days=LOOKBACK_DAYS)
            # Ensure we only look *before* the current note's timestamp
            relevant_nlp_full = patient_nlp_history[
                (patient_nlp_history['timestamp'] < current_time) &
                (patient_nlp_history['timestamp'] >= lookback_start_time)
            ]

            recent_start_time = current_time - timedelta(days=RECENT_WINDOW_DAYS)
            relevant_nlp_recent = relevant_nlp_full[
                relevant_nlp_full['timestamp'] >= recent_start_time
            ]

        # --- NLP Aggregate Features ---
        features['n_ae_mentions_lw'] = relevant_nlp_full['is_ae'].sum() if not relevant_nlp_full.empty else 0
        features['n_severe_ae_lw'] = relevant_nlp_full['is_severe'].sum() if not relevant_nlp_full.empty else 0
        max_sev_lw = relevant_nlp_full['severity_grade'].max() if not relevant_nlp_full.empty else np.nan
        features['max_severity_lw'] = max_sev_lw if pd.notna(max_sev_lw) else 0

        features['n_ae_mentions_rw'] = relevant_nlp_recent['is_ae'].sum() if not relevant_nlp_recent.empty else 0
        features['n_severe_ae_rw'] = relevant_nlp_recent['is_severe'].sum() if not relevant_nlp_recent.empty else 0
        max_sev_rw = relevant_nlp_recent['severity_grade'].max() if not relevant_nlp_recent.empty else np.nan
        features['max_severity_rw'] = max_sev_rw if pd.notna(max_sev_rw) else 0

        features['ratio_severe_ae_rw'] = (features['n_severe_ae_rw'] / features['n_ae_mentions_rw']) if features['n_ae_mentions_rw'] > 0 else 0

        # --- Temporal Features ---
        if pd.notna(prev_note_time):
            features['days_since_last_note'] = (current_time - prev_note_time).days
        else:
            features['days_since_last_note'] = -1 # Indicator for first note

        last_ae_ts = last_ae_time.get(patient_id)
        # Ensure last AE was strictly before current note
        if last_ae_ts and pd.to_datetime(last_ae_ts) < current_time:
             features['days_since_last_ae'] = (current_time - pd.to_datetime(last_ae_ts)).days
        else:
             features['days_since_last_ae'] = -1

        last_severe_ae_ts = last_severe_ae_time.get(patient_id)
        if last_severe_ae_ts and pd.to_datetime(last_severe_ae_ts) < current_time:
            features['days_since_last_severe_ae'] = (current_time - pd.to_datetime(last_severe_ae_ts)).days
        else:
            features['days_since_last_severe_ae'] = -1

        # --- Trend Features ---
        if not relevant_nlp_full.empty and not relevant_nlp_recent.empty:
             # Calculate max severity in the period *before* the recent window starts
             max_sev_before_recent = relevant_nlp_full[relevant_nlp_full['timestamp'] < recent_start_time]['severity_grade'].max()
             max_sev_before_recent = max_sev_before_recent if pd.notna(max_sev_before_recent) else 0
             features['severity_increased_recently'] = 1 if features['max_severity_rw'] > max_sev_before_recent else 0
        else:
             features['severity_increased_recently'] = 0 # Default if not enough history

        # --- Counts of Specific Normalized AEs / Severities ---
        if not relevant_nlp_recent.empty:
             # Example: Count of normalized 'Nausea/Vomiting' in recent window
             features['count_NauseaVomiting_rw'] = relevant_nlp_recent[
                 (relevant_nlp_recent['is_ae'] == 1) &
                 (relevant_nlp_recent['entity_text'] == 'Nausea/Vomiting') # Use normalized name from config?
             ].shape[0]
             # Example: Count of any Grade 4+ AE in lookback window
             features['count_grade4plus_ae_lw'] = relevant_nlp_full[
                 (relevant_nlp_full['is_ae'] == 1) &
                 (relevant_nlp_full['severity_grade'] >= 4) # Assumes severity grade is numeric
             ].shape[0]
        else:
             features['count_NauseaVomiting_rw'] = 0
             features['count_grade4plus_ae_lw'] = 0

        feature_list.append(features)
        processed_feature_count += 1
        if processed_feature_count % 5000 == 0 or processed_feature_count == total_notes_to_feature:
             elapsed = time.time() - start_time
             print(f"  Engineered features for {processed_feature_count}/{total_notes_to_feature} notes ({elapsed:.2f}s)")

    if not feature_list:
        print("Warning: No features were generated.")
        # Return an empty dataframe with expected columns if needed downstream
        # Define expected_cols based on features created above
        # return pd.DataFrame(columns=['note_id', 'patient_id', 'target', 'age', ...]) # List all columns
        return pd.DataFrame()


    feature_df = pd.DataFrame(feature_list)
    # Fill NaNs strategically
    feature_df.fillna({'days_since_last_note': -1, 'days_since_last_ae': -1, 'days_since_last_severe_ae': -1}, inplace=True)
    # Fill remaining NaNs that might arise from calculations (e.g., ratios, max severity if no AEs) with 0
    feature_df.fillna(0, inplace=True)

    # Convert cancer_type to category for potential use by LGBM
    if 'cancer_type' in feature_df.columns:
        feature_df['cancer_type'] = feature_df['cancer_type'].astype('category')

    end_time = time.time()
    print(f"Feature engineering finished in {end_time - start_time:.2f} seconds.")
    return feature_df

