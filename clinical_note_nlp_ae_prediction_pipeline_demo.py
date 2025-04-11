# --- 0. Import Libraries ---
import pandas as pd
import numpy as np
import random
import re
from datetime import datetime, timedelta
import warnings
import time
import joblib # For saving models potentially

# NLP - Using spaCy with Transformer support & Negation Detection
import spacy
import spacy_transformers # Required for en_core_web_trf
from spacy.matcher import Matcher
from spacy.tokens import Span
try:
    from negspacy.negation import Negex
    NEGSPACY_AVAILABLE = True
except ImportError:
    print("Warning: negspacy not found. Negation detection will be skipped. Install with: pip install negspacy")
    NEGSPACY_AVAILABLE = False

# Machine Learning & Evaluation
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RandomizedSearchCV, StratifiedGroupKFold) # Added GroupKFold and RandomizedSearch
from sklearn.ensemble import RandomForestClassifier # Keep as an alternative/comparison
import lightgbm as lgb # Using LightGBM
from sklearn.metrics import (classification_report, roc_curve, auc, confusion_matrix,
                             precision_recall_curve, average_precision_score, brier_score_loss) # Added PR curve metrics, Brier score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibrationDisplay # For calibration plot

# Explainable AI
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("Warning: shap not found. SHAP explanations will be skipped. Install with: pip install shap")
    SHAP_AVAILABLE = False

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Ignore specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# --- 1. Configuration & Parameters ---
PATIENT_DATA_PATH = r'path/to/your/patient_baseline_data.csv' # UPDATE
NOTES_DATA_PATH = r'path/to/your/clinical_notes.csv'       # UPDATE
AE_EVENTS_PATH = r'path/to/your/recorded_ae_events.csv'   # UPDATE

# Feature Engineering Parameters
LOOKBACK_DAYS = 90
RECENT_WINDOW_DAYS = 14
OUTCOME_WINDOW_DAYS = 30

# NLP Model
NLP_MODEL_NAME = "en_core_web_trf" # Advanced (Requires GPU for reasonable speed!)
# NLP_MODEL_NAME = "en_core_web_lg" # Large non-transformer alternative

# ML Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5 # Folds for cross-validation during tuning
N_ITER_SEARCH = 25 # Number of parameter settings sampled by RandomizedSearchCV

# Definitions
DRUGS = ["Cisplatin", "Paclitaxel", "Pembrolizumab", "Letrozole", "Trastuzumab", "Oxaliplatin"] # Example
AES = ["Nausea", "Vomiting", "Fatigue", "Anemia", "Neutropenia", "Thrombocytopenia", "Diarrhea", "Rash", "Neuropathy", "Pneumonitis", "Mucositis"] # Example

# Basic AE Normalization Map (Expand significantly for real use)
AE_NORMALIZATION_MAP = {
    "n/v": "Nausea/Vomiting", # Example grouping
    "nausea": "Nausea/Vomiting",
    "vomiting": "Nausea/Vomiting",
    "emesis": "Nausea/Vomiting",
    "fatigue": "Fatigue",
    "tiredness": "Fatigue",
    "low white count": "Neutropenia",
    "neutropenia": "Neutropenia",
    "low platelets": "Thrombocytopenia",
    "thrombocytopenia": "Thrombocytopenia",
    "diarrhea": "Diarrhea",
    "loose stools": "Diarrhea",
    "rash": "Rash",
    "skin rash": "Rash",
    "neuropathy": "Neuropathy",
    "tingling": "Neuropathy",
    "numbness": "Neuropathy",
    "pneumonitis": "Pneumonitis",
    "mucositis": "Mucositis",
    "mouth sores": "Mucositis",
    "anemia": "Anemia",
    "low hemoglobin": "Anemia",
    "low hgb": "Anemia"
    # Add many more mappings based on data exploration
}
# Severity remains unchanged
SEVERITY_TERMS = { "grade 1": 1, "g1": 1, "mild": 1, "grade 2": 2, "g2": 2, "moderate": 2, "grade 3": 3, "g3": 3, "severe": 3, "grade 4": 4, "g4": 4, "life-threatening": 4, "grade 5": 5, "g5": 5, "death": 5 }
SEVERITY_PATTERNS = list(SEVERITY_TERMS.keys())

# Standardized AE Labels (used by Negex and potentially the model)
AE_LABELS = ["AE", "ADVERSE_EVENT", "SYMPTOM"] # Adjust based on your spaCy model's output

# --- 2. Data Loading and Preprocessing Functions ---
# Assume load_patient_data, load_notes_data, determine_ae_outcomes exist and work robustly
# (Implementations omitted for brevity - use robust versions from previous steps)
def load_patient_data(filepath):
    """Loads baseline patient data. [Implementation from previous version]"""
    print(f"Loading patient data from: {filepath}")
    # ... (Add robust loading, cleaning, type conversion logic here) ...
    # Example structure:
    try:
        patients_df = pd.read_csv(filepath)
        # --- Add necessary cleaning/validation ---
        required_cols = ['patient_id', 'age', 'cancer_type', 'treatment_regimen']
        assert all(col in patients_df.columns for col in required_cols), "Missing required columns"
        # ... more cleaning ...
        if 'treatment_regimen' in patients_df.columns and isinstance(patients_df['treatment_regimen'].iloc[0], str):
             patients_df['treatment_regimen'] = patients_df['treatment_regimen'].apply(lambda x: x.split(';') if pd.notna(x) else [])
        if 'baseline_risk_score' not in patients_df.columns:
             patients_df['baseline_risk_score'] = (patients_df['age'] / 80).clip(0.1, 1.0) # Example proxy
        print(f"Loaded {len(patients_df)} patients.")
        return patients_df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        raise
    except Exception as e:
        print(f"Error processing patient data: {e}")
        raise

def load_notes_data(filepath):
    """Loads clinical notes. [Implementation from previous version]"""
    print(f"Loading clinical notes from: {filepath}")
    # ... (Add robust loading, cleaning, date parsing logic here) ...
    try:
        notes_df = pd.read_csv(filepath)
        # --- Add necessary cleaning/validation ---
        required_cols = ['note_id', 'patient_id', 'timestamp', 'note_text']
        assert all(col in notes_df.columns for col in required_cols), "Missing required columns"
        notes_df['timestamp'] = pd.to_datetime(notes_df['timestamp'], errors='coerce')
        notes_df.dropna(subset=['timestamp', 'note_text'], inplace=True)
        notes_df['note_text'] = notes_df['note_text'].astype(str)
        print(f"Loaded {len(notes_df)} notes.")
        return notes_df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        raise
    except Exception as e:
        print(f"Error processing notes data: {e}")
        raise

def determine_ae_outcomes(notes_df, ae_events_filepath, outcome_window_days):
    """Determines target variable from recorded AE events. [Implementation from previous version]"""
    print(f"Determining outcomes using AE events from: {ae_events_filepath}")
    # ... (Add robust loading of AE events, date parsing, grade cleaning, and logic
    #      to map AE events within OUTCOME_WINDOW_DAYS to preceding notes) ...
    try:
        ae_events_df = pd.read_csv(ae_events_filepath)
        # --- Add necessary cleaning/validation ---
        required_cols = ['patient_id', 'ae_timestamp', 'ae_grade']
        assert all(col in ae_events_df.columns for col in required_cols), "Missing required AE event columns"
        ae_events_df['ae_timestamp'] = pd.to_datetime(ae_events_df['ae_timestamp'], errors='coerce')
        # Handle potential non-numeric grades before converting
        ae_events_df['ae_grade'] = pd.to_numeric(ae_events_df['ae_grade'], errors='coerce')
        ae_events_df.dropna(subset=['patient_id', 'ae_timestamp', 'ae_grade'], inplace=True)
        ae_events_df['ae_grade'] = ae_events_df['ae_grade'].astype('Int64') # Keep as nullable Int

        target_map = {}
        notes_df = notes_df.sort_values(['patient_id', 'timestamp'])
        ae_events_df = ae_events_df.sort_values(['patient_id', 'ae_timestamp'])
        ae_events_grouped = ae_events_df.groupby('patient_id')
        patient_ae_map = {pid: group for pid, group in ae_events_grouped}

        print(f"Mapping AEs within {outcome_window_days} days post-note...")
        num_notes_processed = 0
        start_time_outcome = time.time()

        for index, note in notes_df.iterrows():
            patient_id = note['patient_id']
            note_timestamp = note['timestamp']
            note_id = note['note_id']
            window_start = note_timestamp + timedelta(microseconds=1) # Start just after the note
            window_end = note_timestamp + timedelta(days=outcome_window_days)
            target = 0 # Default: No severe AE in period
            patient_aes = patient_ae_map.get(patient_id)

            if patient_aes is not None:
                # Efficiently find AEs within the window for this patient
                aes_in_window = patient_aes[
                    (patient_aes['ae_timestamp'] >= window_start) &
                    (patient_aes['ae_timestamp'] <= window_end) &
                    (patient_aes['ae_grade'].notna()) # Ensure grade is valid
                ]
                if not aes_in_window.empty:
                    # Check if *any* AE in the window is Grade 3 or higher
                    if (aes_in_window['ae_grade'] >= 3).any():
                        target = 1 # Severe AE occurred

            target_map[note_id] = target
            num_notes_processed += 1
            if num_notes_processed % 5000 == 0:
                 elapsed = time.time() - start_time_outcome
                 print(f"  ...processed outcomes for {num_notes_processed}/{len(notes_df)} notes ({elapsed:.2f}s)")


        notes_df['severe_ae_in_period'] = notes_df['note_id'].map(target_map)
        # We keep all notes initially, features will be generated based on NLP results later
        # notes_with_outcomes = notes_df.dropna(subset=['severe_ae_in_period']) # Don't drop notes yet
        notes_with_outcomes = notes_df
        notes_with_outcomes['severe_ae_in_period'] = notes_with_outcomes['severe_ae_in_period'].astype(int) # Ensure integer type
        print(f"Finished outcome determination for {len(notes_with_outcomes)} notes in {time.time() - start_time_outcome:.2f}s.")
        return notes_with_outcomes
    except FileNotFoundError:
        print(f"Error: AE events file not found at {ae_events_filepath}")
        raise
    except Exception as e:
        print(f"Error processing AE events or determining outcomes: {e}")
        raise


# --- 3. Advanced ClinNLP Extraction (with Negation & Normalization) ---

print(f"\nLoading spaCy NLP model: {NLP_MODEL_NAME}...")
if NLP_MODEL_NAME.endswith("_trf"):
    try:
        if spacy.require_gpu(): print("GPU available, spaCy Transformer model will use it.")
        else: print("Warning: GPU not detected. Transformer model will run on CPU (SLOW!).")
    except Exception as e: print(f"GPU check failed: {e}. Running on CPU.")

try:
    nlp = spacy.load(NLP_MODEL_NAME)
except OSError:
    print(f"Model {NLP_MODEL_NAME} not found. Downloading...")
    spacy.cli.download(NLP_MODEL_NAME)
    nlp = spacy.load(NLP_MODEL_NAME)

# Add Matcher for severity terms
matcher = Matcher(nlp.vocab)
for term in SEVERITY_PATTERNS:
    pattern = [{"LOWER": word} for word in term.split()]
    matcher.add(f"SEVERITY_{term.upper()}", [pattern])

# Add Negation Detection pipe (if available)
if NEGSPACY_AVAILABLE:
    print("Adding negSpacy negation detection pipe...")
    # Configure negex for relevant entity types (use AE_LABELS)
    negex = Negex(nlp, name="negex", ent_types=AE_LABELS)
    if "negex" not in nlp.pipe_names: # Avoid adding multiple times if re-running cells
      nlp.add_pipe("negex", last=True)
    else:
      print("negex pipe already exists.")
else:
    print("Skipping negation detection (negspacy not installed).")

# --- Link AE Severity (remains basic, focus on negation/normalization first) ---
def link_ae_severity_improved(doc, ae_ents, severity_matches):
    """Links AE entities with nearby severity terms (simple proximity)."""
    linked_data = {}
    # Convert matcher results to Spans
    severity_spans = [Span(doc, start, end, label=doc.vocab.strings[match_id]) for match_id, start, end in severity_matches]
    sorted_severity_spans = sorted(severity_spans, key=lambda s: s.start) # Sort for efficiency?

    for ae in ae_ents:
        # Simple proximity check (e.g., severity term immediately follows or closely precedes)
        linked_severity_grade = None
        linked_severity_term = None
        min_dist_after = 5  # Max distance for severity *after* AE
        min_dist_before = 3 # Max distance for severity *before* AE

        for sev_span in sorted_severity_spans:
            # Check if severity follows AE closely
            if sev_span.start >= ae.end and (sev_span.start - ae.end) < min_dist_after:
                 # Optional: Check for intervening punctuation like '('
                 # intervening_text = doc[ae.end:sev_span.start].text.strip()
                 # if not intervening_text or intervening_text in ['(','-']:
                     sev_text = sev_span.text.lower()
                     if sev_text in SEVERITY_TERMS:
                         linked_severity_grade = SEVERITY_TERMS[sev_text]
                         linked_severity_term = sev_text
                         min_dist_after = sev_span.start - ae.end # Found closer one
                         break # Often take the first close one found after

            # Check if severity precedes AE closely (less common but possible)
            elif ae.start >= sev_span.end and (ae.start - sev_span.end) < min_dist_before:
                 sev_text = sev_span.text.lower()
                 if sev_text in SEVERITY_TERMS:
                      # Only update if no 'after' severity was found, or maybe based on distance?
                      # Simple approach: take it if no 'after' grade found yet
                      if linked_severity_grade is None:
                           linked_severity_grade = SEVERITY_TERMS[sev_text]
                           linked_severity_term = sev_text
                           # Don't necessarily break, an 'after' term might be better

        linked_data[ae] = {'severity_grade': linked_severity_grade, 'severity_term': linked_severity_term}
    return linked_data


def extract_entities_advanced_nlp(notes_df):
    """Extracts entities using spaCy, applies normalization, and skips negated AEs."""
    print(f"\nExtracting entities using {NLP_MODEL_NAME} with Negation & Normalization...")
    extracted_results = []
    start_time = time.time()

    texts = notes_df['note_text'].tolist()
    note_ids = notes_df['note_id'].tolist()
    patient_ids = notes_df['patient_id'].tolist()
    timestamps = notes_df['timestamp'].tolist()

    total_notes = len(texts)
    processed_count = 0
    affirmative_ae_count = 0
    negated_ae_count = 0

    # Using nlp.pipe for efficient processing
    # Adjust batch_size based on memory/GPU. Smaller batch size for less memory.
    batch_size = 16 if NLP_MODEL_NAME.endswith("_trf") else 64
    print(f"Processing {total_notes} notes in batches of {batch_size}...")
    docs = nlp.pipe(texts, batch_size=batch_size)

    for i, doc in enumerate(docs):
        note_id = note_ids[i]
        patient_id = patient_ids[i]
        timestamp = timestamps[i]

        # Get entities & filter for relevant types
        all_entities = doc.ents
        ae_entities = [ent for ent in all_entities if ent.label_ in AE_LABELS]
        drug_entities = [ent for ent in all_entities if ent.label_ in ["DRUG", "CHEMICAL"]]

        # Match severity terms
        severity_matches = matcher(doc)
        ae_severity_links = link_ae_severity_improved(doc, ae_entities, severity_matches)

        # Store results, applying normalization and checking negation
        relevant_entities = ae_entities + drug_entities
        for ent in relevant_entities:
            entity_type = ent.label_
            entity_text_original = ent.text.lower()
            normalized_entity_text = entity_text_original # Default
            severity_grade = None
            severity_term = None
            is_negated = False

            # Check Negation only for AE types
            if entity_type in AE_LABELS:
                # Check if negSpacy ran and found negation
                if NEGSPACY_AVAILABLE and hasattr(ent._, 'negex'):
                    is_negated = ent._.negex
                else:
                    is_negated = False # Assume not negated if negspacy unavailable/failed

                if is_negated:
                    negated_ae_count += 1
                    continue # *** SKIP NEGATED AE MENTIONS ***

                # Apply normalization only to affirmative AEs
                affirmative_ae_count += 1
                normalized_entity_text = AE_NORMALIZATION_MAP.get(entity_text_original, entity_text_original)

                # Get linked severity if it's an AE
                if ent in ae_severity_links:
                    severity_grade = ae_severity_links[ent]['severity_grade']
                    severity_term = ae_severity_links[ent]['severity_term']

            # For drugs, just normalize text (optional, map to generic names?)
            elif entity_type in ["DRUG", "CHEMICAL"]:
                 # Simple normalization could be added here if needed
                 normalized_entity_text = entity_text_original

            extracted_results.append({
                'note_id': note_id,
                'patient_id': patient_id,
                'timestamp': timestamp,
                'entity_type': entity_type,
                'entity_text': normalized_entity_text, # Use normalized text
                'entity_text_original': entity_text_original, # Keep original for reference
                'severity_grade': severity_grade,
                'severity_term': severity_term,
                # 'is_negated': is_negated # Could store this if not skipping
            })

        processed_count += 1
        if processed_count % 200 == 0:
            elapsed = time.time() - start_time
            print(f"  Processed {processed_count}/{total_notes} notes ({elapsed:.2f} seconds)")

    end_time = time.time()
    print(f"NLP extraction finished in {end_time - start_time:.2f} seconds.")
    print(f"  Found {affirmative_ae_count} affirmative AE mentions.")
    if NEGSPACY_AVAILABLE:
        print(f"  Skipped {negated_ae_count} negated AE mentions.")

    return pd.DataFrame(extracted_results)


# --- 4. Enhanced Feature Engineering (Using Normalized, Affirmed Entities) ---

def create_advanced_features(nlp_results_df, notes_with_outcomes_df, patients_df, lookback_days=90, recent_window_days=14):
    """Creates features using normalized, affirmative NLP results and temporal dynamics."""
    print("\nEngineering advanced features...")
    start_time = time.time()

    # Ensure DataFrames are sorted correctly
    nlp_results_df = nlp_results_df.sort_values(['patient_id', 'timestamp'])
    notes_with_outcomes_df = notes_with_outcomes_df.sort_values(['patient_id', 'timestamp'])
    patients_df_indexed = patients_df.set_index('patient_id') # Index patients for faster lookup

    feature_list = []

    # Pre-calculate some NLP aggregations per patient for efficiency
    nlp_results_df['is_ae'] = nlp_results_df['entity_type'].isin(AE_LABELS).astype(int)
    nlp_results_df['is_severe'] = ((nlp_results_df['severity_grade'] >= 3) & nlp_results_df['is_ae']).astype(int)
    # Timestamps of last events per patient
    last_event_time = nlp_results_df.groupby('patient_id')['timestamp'].last().to_dict()
    last_ae_time = nlp_results_df[nlp_results_df['is_ae'] == 1].groupby('patient_id')['timestamp'].last().to_dict()
    last_severe_ae_time = nlp_results_df[nlp_results_df['is_severe'] == 1].groupby('patient_id')['timestamp'].last().to_dict()

    # Efficiently get previous note timestamps per patient
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
        prev_note_time = note['prev_timestamp'] # Timestamp of the previous note for this patient

        features = {'note_id': note_id, 'patient_id': patient_id, 'target': target} # Keep patient_id for grouping later

        # --- Baseline Patient Features ---
        try:
            patient_info = patients_df_indexed.loc[patient_id]
            features['age'] = patient_info['age']
            features['baseline_risk_score'] = patient_info.get('baseline_risk_score', -1) # Handle missing baseline score
            features['cancer_type'] = patient_info['cancer_type']
            regimen = patient_info['treatment_regimen']
            features['n_drugs_regimen'] = len(regimen) if isinstance(regimen, list) else 0
            for drug in DRUGS: # Use the global DRUGS list
                 features[f'has_{drug}'] = 1 if isinstance(regimen, list) and drug in regimen else 0
        except KeyError:
             # print(f"Warning: Patient ID {patient_id} not found in patient baseline data. Skipping note {note_id}.")
             continue # Skip this note if baseline data is missing

        # Get NLP results for this patient efficiently
        patient_nlp_history = patient_nlp_map.get(patient_id)
        if patient_nlp_history is None or patient_nlp_history.empty:
             # Handle cases with no prior NLP info for this patient (assign defaults)
             relevant_nlp_full = pd.DataFrame()
             relevant_nlp_recent = pd.DataFrame()
        else:
            # Filter NLP results for the full lookback window up to the current note time
            lookback_start_time = current_time - timedelta(days=lookback_days)
            relevant_nlp_full = patient_nlp_history[
                (patient_nlp_history['timestamp'] < current_time) & # Only use history *before* current note
                (patient_nlp_history['timestamp'] >= lookback_start_time)
            ]

            # Filter for the recent window
            recent_start_time = current_time - timedelta(days=recent_window_days)
            relevant_nlp_recent = relevant_nlp_full[
                relevant_nlp_full['timestamp'] >= recent_start_time
            ]

        # --- NLP Aggregate Features (using pre-calculated flags) ---
        # Use .get(patient_id, default_value) for precalculated times to handle missing patients
        features['n_ae_mentions_lw'] = relevant_nlp_full['is_ae'].sum() if not relevant_nlp_full.empty else 0
        features['n_severe_ae_lw'] = relevant_nlp_full['is_severe'].sum() if not relevant_nlp_full.empty else 0
        max_sev_lw = relevant_nlp_full['severity_grade'].max() if not relevant_nlp_full.empty else np.nan
        features['max_severity_lw'] = max_sev_lw if pd.notna(max_sev_lw) else 0

        features['n_ae_mentions_rw'] = relevant_nlp_recent['is_ae'].sum() if not relevant_nlp_recent.empty else 0
        features['n_severe_ae_rw'] = relevant_nlp_recent['is_severe'].sum() if not relevant_nlp_recent.empty else 0
        max_sev_rw = relevant_nlp_recent['severity_grade'].max() if not relevant_nlp_recent.empty else np.nan
        features['max_severity_rw'] = max_sev_rw if pd.notna(max_sev_rw) else 0

        # Ratio: Recent severe AE mentions / Total recent AE mentions (handle division by zero)
        features['ratio_severe_ae_rw'] = (features['n_severe_ae_rw'] / features['n_ae_mentions_rw']) if features['n_ae_mentions_rw'] > 0 else 0

        # --- Temporal Features ---
        if pd.notna(prev_note_time):
            features['days_since_last_note'] = (current_time - prev_note_time).days
        else:
            features['days_since_last_note'] = -1 # Indicator for first note

        last_ae_ts = last_ae_time.get(patient_id)
        if last_ae_ts and last_ae_ts < current_time: # Ensure last AE was strictly before current note
             features['days_since_last_ae'] = (current_time - last_ae_ts).days
        else:
             features['days_since_last_ae'] = -1

        last_severe_ae_ts = last_severe_ae_time.get(patient_id)
        if last_severe_ae_ts and last_severe_ae_ts < current_time:
            features['days_since_last_severe_ae'] = (current_time - last_severe_ae_ts).days
        else:
            features['days_since_last_severe_ae'] = -1

        # --- Trend Features ---
        # Flag if max severity increased recently compared to the period before the recent window
        if not relevant_nlp_full.empty:
             max_sev_before_recent = relevant_nlp_full[relevant_nlp_full['timestamp'] < recent_start_time]['severity_grade'].max()
             max_sev_before_recent = max_sev_before_recent if pd.notna(max_sev_before_recent) else 0
             features['severity_increased_recently'] = 1 if features['max_severity_rw'] > max_sev_before_recent else 0
        else:
             features['severity_increased_recently'] = 0

        # --- Counts of Specific Normalized AEs / Severities ---
        # Example: Count of normalized 'Nausea/Vomiting' in recent window
        if not relevant_nlp_recent.empty:
             features['count_NauseaVomiting_rw'] = relevant_nlp_recent[
                 (relevant_nlp_recent['is_ae'] == 1) &
                 (relevant_nlp_recent['entity_text'] == 'Nausea/Vomiting') # Use normalized name
             ].shape[0]
             # Example: Count of any Grade 4+ AE in lookback window
             features['count_grade4plus_ae_lw'] = relevant_nlp_full[
                 (relevant_nlp_full['is_ae'] == 1) &
                 (relevant_nlp_full['severity_grade'] >= 4)
             ].shape[0]
        else:
             features['count_NauseaVomiting_rw'] = 0
             features['count_grade4plus_ae_lw'] = 0


        # --- Placeholder for Text Embedding Features ---
        # Add later if implementing embedding extraction in step 3

        feature_list.append(features)
        processed_feature_count += 1
        if processed_feature_count % 5000 == 0:
             elapsed = time.time() - start_time
             print(f"  Engineered features for {processed_feature_count}/{total_notes_to_feature} notes ({elapsed:.2f}s)")


    feature_df = pd.DataFrame(feature_list)
    # Fill any potential NaNs introduced (e.g., from calculations)
    # Use -1 as indicator for missing temporal features etc.
    feature_df.fillna({'days_since_last_note': -1, 'days_since_last_ae': -1, 'days_since_last_severe_ae': -1}, inplace=True)
    feature_df.fillna(0, inplace=True) # Fill remaining NaNs (e.g., ratios) with 0

    end_time = time.time()
    print(f"Feature engineering finished in {end_time - start_time:.2f} seconds.")
    return feature_df


# --- 5. Machine Learning Pipeline (Tuned LightGBM + SHAP + Calibration) ---

def train_evaluate_advanced_model(feature_df, random_state=RANDOM_STATE, test_size=TEST_SIZE, cv_folds=CV_FOLDS, n_iter_search=N_ITER_SEARCH):
    """Tunes LightGBM using RandomizedSearchCV, evaluates, and extracts SHAP values."""
    print("\n--- Training and Evaluating Advanced Model (Tuned LightGBM + SHAP) ---")
    if feature_df.empty or 'target' not in feature_df.columns or feature_df['target'].nunique() < 2:
         print("Insufficient data or variance in target variable for model training. Skipping.")
         return None, {}, None, None, None # Model, metrics, shap_values, X_test_processed, feature_names

    # Separate features (X), target (y), and groups (patient_id for CV)
    y = feature_df['target']
    groups = feature_df['patient_id'] # Keep for GroupKFold
    X = feature_df.drop(['note_id', 'patient_id', 'target'], axis=1)

    # Identify feature types
    categorical_features_names = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'cancer_type' in X.columns:
         X['cancer_type'] = X['cancer_type'].astype('category')
         categorical_features_names = X.select_dtypes(include=['category']).columns.tolist() # Update

    numerical_features_names = X.select_dtypes(include=np.number).columns.tolist()

    # Split data (Train/Validation + Test) - Stratified by target, groups used later in CV
    try:
        X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
            X, y, groups, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Train/Val data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        target_dist_train = y_train.value_counts(normalize=True)
        target_dist_test = y_test.value_counts(normalize=True)
        print(f"Target distribution in Train/Val data:\n{target_dist_train}")
        print(f"Target distribution in Test data:\n{target_dist_test}")
        if len(target_dist_train) < 2:
            print("Training data has only one class. Skipping model fitting.")
            return None, {}, None, None, None
    except ValueError as e:
         print(f"Error during train/test split (potentially too few samples per class/group): {e}. Skipping training.")
         return None, {}, None, None, None

    # --- Preprocessing ---
    # Scale numerical, pass through categorical (LGBM handles them if type='category')
    numerical_transformer = StandardScaler()
    # Note: OneHotEncoder might be needed if using models other than LGBM/XGBoost/CatBoost
    # categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features_names)],
            #('cat', categorical_transformer, categorical_features_names)], # Use only if OHE needed
        remainder='passthrough', # KEEP categorical features for LGBM internal handling
        verbose_feature_names_out=False) # Keep original names easier

    # --- LightGBM Classifier (Base) ---
    # Calculate scale_pos_weight for imbalance
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1
    print(f"Calculated scale_pos_weight for training: {scale_pos_weight:.2f}")

    lgbm = lgb.LGBMClassifier(
        random_state=random_state,
        objective='binary',
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1
    )

    # --- Model Pipeline ---
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', lgbm)])

    # --- Hyperparameter Tuning (Randomized Search with Group K-Fold) ---
    print(f"\nStarting Randomized Search CV (Folds={cv_folds}, Iterations={n_iter_search})...")
    # Define parameter distributions to sample from (adjust ranges based on experience/data)
    param_distributions = {
        'classifier__n_estimators': [100, 200, 300, 400, 500],
        'classifier__learning_rate': [0.01, 0.02, 0.05, 0.1],
        'classifier__num_leaves': [15, 31, 61, 91],
        'classifier__max_depth': [-1, 5, 10, 15],
        'classifier__reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
        'classifier__reg_lambda': [0, 0.01, 0.1, 0.5, 1.0],
        'classifier__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0], # Feature subsampling
        'classifier__subsample': [0.6, 0.7, 0.8, 0.9, 1.0], # Row subsampling
    }

    # Use StratifiedGroupKFold to respect patient boundaries during CV
    cv = StratifiedGroupKFold(n_splits=cv_folds)

    # Scoring metric for tuning (use AUPRC for imbalanced data)
    scoring = 'average_precision'

    search = RandomizedSearchCV(
        model_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter_search,
        scoring=scoring,
        cv=cv, # Use group k-fold
        random_state=random_state,
        n_jobs=-1, # Use all available cores
        verbose=1, # Show progress
        refit=True # Refit the best model on the whole train/val set
    )

    start_time_tuning = time.time()
    # --- Pass categorical feature info to LGBM within the pipeline ---
    fit_params = {}
    if categorical_features_names:
        # Ensure pipeline step name ('classifier') matches
        fit_params['classifier__categorical_feature'] = [f for f in categorical_features_names if f in X_train.columns]
        print(f"Passing categorical features to LGBM during search: {fit_params['classifier__categorical_feature']}")

    try:
        search.fit(X_train, y_train, groups=groups_train, **fit_params) # Pass groups!
    except Exception as e:
        print(f"\nError during RandomizedSearchCV fitting: {e}")
        print("This might be due to small group sizes or issues with categorical features.")
        print("Consider reducing CV folds or checking feature preprocessing.")
        return None, {}, None, None, None

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
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")


    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    print(f"\nROC AUC Score: {roc_auc:.4f}")

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    print(f"Average Precision (AUPRC): {avg_precision:.4f}")

    # Brier Score (Calibration measure)
    brier = brier_score_loss(y_test, y_pred_proba)
    print(f"Brier Score: {brier:.4f}")


    metrics = {
        'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr,
        'avg_precision': avg_precision, 'precision': precision, 'recall': recall,
        'brier_score': brier, 'confusion_matrix': cm
    }

    # --- SHAP Explanations ---
    shap_values = None
    X_test_processed = None
    feature_names_processed = None
    if SHAP_AVAILABLE:
        print("\nCalculating SHAP values for test set...")
        start_time_shap = time.time()
        try:
            # Need to get the trained classifier and preprocessor from the best pipeline
            best_lgbm = best_model.named_steps['classifier']
            preprocessor = best_model.named_steps['preprocessor']

            # Transform the test data using the fitted preprocessor
            X_test_processed = preprocessor.transform(X_test)

            # Get feature names *after* preprocessing (crucial!)
            try:
                 feature_names_processed = preprocessor.get_feature_names_out()
                 # Ensure feature names are strings (sometimes needed for SHAP plots)
                 feature_names_processed = [str(f) for f in feature_names_processed]
                 print(f"  Got {len(feature_names_processed)} processed feature names.")
            except Exception as e_fn:
                 print(f"  Warning: Could not automatically get processed feature names: {e_fn}. Using indices.")
                 feature_names_processed = [f"feature_{i}" for i in range(X_test_processed.shape[1])]


            # Ensure X_test_processed is a DataFrame if SHAP needs it (depends on explainer type)
            # TreeExplainer usually works with numpy arrays, but DataFrames are safer for plotting
            if not isinstance(X_test_processed, pd.DataFrame):
                 X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names_processed)
            else:
                 X_test_processed_df = X_test_processed


            # Use TreeExplainer for tree-based models like LightGBM
            explainer = shap.TreeExplainer(best_lgbm)

            # Calculate SHAP values for the positive class (index 1 for binary classification)
            shap_values_tuple = explainer.shap_values(X_test_processed_df) # Will be a list/tuple for multi-class if not binary

            # For binary classification, shap_values returns a list [shap_values_class0, shap_values_class1]
            # We usually care about the positive class (class 1)
            if isinstance(shap_values_tuple, list) and len(shap_values_tuple) == 2:
                 shap_values = shap_values_tuple[1]
                 print(f"  SHAP values calculated for positive class (shape: {shap_values.shape})")
            else:
                 # Handle unexpected SHAP output format
                 shap_values = shap_values_tuple # Use the raw output if unsure
                 print(f"  SHAP values calculated (format might differ, shape: {np.shape(shap_values)})")


            print(f"SHAP calculation finished in {time.time() - start_time_shap:.2f} seconds.")

        except Exception as e:
            print(f"Error during SHAP value calculation: {e}")
            print("SHAP explanations will be skipped.")
            shap_values = None # Ensure it's None if failed
            X_test_processed = None
            feature_names_processed = None

    else:
        print("SHAP library not available. Skipping SHAP analysis.")


    return best_model, metrics, shap_values, X_test_processed_df, feature_names_processed


# --- 6. Enhanced Visualization Functions ---
# Include previous plots + SHAP + Calibration

# Keep plot_ae_frequency, plot_ae_severity, plot_ae_by_drug, plot_patient_timeline (implementations omitted)
def plot_ae_frequency(nlp_results_df, top_n=15):
    """Plots overall frequency of detected AEs (using normalized names)."""
    print("\nPlotting Figure 1: Normalized AE Frequency")
    # Use the 'entity_text' which is normalized, filter for AE types
    ae_counts = nlp_results_df[nlp_results_df['entity_type'].isin(AE_LABELS)]['entity_text'].value_counts()
    if ae_counts.empty: print("No affirmative AEs found to plot frequency."); return
    plt.figure(figsize=(10, 6))
    ae_counts.head(top_n).plot(kind='bar')
    plt.title(f'Figure 1: Top {top_n} Frequent Normalized AEs (Affirmative Mentions)')
    plt.ylabel('Mentions')
    plt.xlabel('Normalized Adverse Event')
    plt.xticks(rotation=45, ha='right'); plt.tight_layout(); plt.show()

def plot_ae_severity(nlp_results_df):
    """Plots distribution of AE severity grades (for affirmative AEs)."""
    print("\nPlotting Figure 2: AE Severity Distribution (Affirmative Mentions)")
    # Filter for AE types before checking severity
    severity_counts = nlp_results_df[nlp_results_df['entity_type'].isin(AE_LABELS)].dropna(subset=['severity_grade'])
    if severity_counts.empty: print("No severity grades found for affirmative AEs to plot."); return
    # Ensure grades are numeric for plotting
    severity_counts['severity_grade'] = pd.to_numeric(severity_counts['severity_grade'], errors='coerce')
    severity_counts.dropna(subset=['severity_grade'], inplace=True)
    if severity_counts.empty: print("No numeric severity grades found after conversion."); return

    grade_counts = severity_counts['severity_grade'].astype(float).value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    grade_counts.plot(kind='bar', color='skyblue')
    plt.title('Figure 2: AE Severity Grades (Affirmative Mentions)')
    plt.xlabel('Severity Grade')
    plt.ylabel('Mentions')
    plt.xticks(rotation=0); plt.tight_layout(); plt.show()

def plot_ae_by_drug(nlp_results_df, top_n_drugs=5, top_n_aes=10):
    """Plots AEs co-occurring with specific drugs in notes (uses normalized AE names)."""
    print("\nPlotting Figure 3: Normalized AEs Co-occurring with Drugs")
    # Filter for DRUG/CHEMICAL entities
    drug_mentions = nlp_results_df[nlp_results_df['entity_type'].isin(['DRUG','CHEMICAL'])][['note_id', 'entity_text']].drop_duplicates()
    # Filter for AE entities and use normalized names
    ae_mentions = nlp_results_df[nlp_results_df['entity_type'].isin(AE_LABELS)][['note_id', 'entity_text']].drop_duplicates()

    if drug_mentions.empty or ae_mentions.empty:
        print("No drug or AE mentions found for co-occurrence plot.")
        return

    merged_mentions = pd.merge(drug_mentions, ae_mentions, on='note_id', suffixes=('_drug', '_ae'))
    if merged_mentions.empty: print("No drug-AE co-occurrences found in the same notes."); return

    # Create the co-occurrence matrix
    cooccurrence_counts = merged_mentions.groupby(['entity_text_drug', 'entity_text_ae']).size().unstack(fill_value=0)

    # Determine top drugs and AEs based on overall frequency in the NLP results
    top_drugs = nlp_results_df[nlp_results_df['entity_type'].isin(['DRUG','CHEMICAL'])]['entity_text'].value_counts().head(top_n_drugs).index
    top_aes = nlp_results_df[nlp_results_df['entity_type'].isin(AE_LABELS)]['entity_text'].value_counts().head(top_n_aes).index

    # Filter the matrix for top drugs and AEs present in the matrix indices/columns
    common_drugs = cooccurrence_counts.index.intersection(top_drugs)
    common_aes = cooccurrence_counts.columns.intersection(top_aes)

    if common_drugs.empty or common_aes.empty:
        print("No co-occurrences found between the selected top drugs and top AEs.")
        return

    cooccurrence_filtered = cooccurrence_counts.loc[common_drugs, common_aes]

    plt.figure(figsize=(12, max(6, len(common_drugs) * 0.8))) # Adjust height based on number of drugs
    sns.heatmap(cooccurrence_filtered, cmap="viridis", annot=True, fmt="d", linewidths=.5)
    plt.title(f'Figure 3: Drug-AE Co-occurrence (Top {len(common_drugs)} Drugs, Top {len(common_aes)} Norm. AEs)')
    plt.xlabel('Normalized Adverse Event')
    plt.ylabel('Drug')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout(); plt.show()


def plot_patient_timeline(nlp_results_df, patient_id_to_plot):
    """Plots a timeline of AEs and severity for a single patient (uses normalized names)."""
    print(f"\nPlotting Figure 4: Timeline for Patient {patient_id_to_plot}")
    # Filter NLP results for the specific patient
    patient_data = nlp_results_df[nlp_results_df['patient_id'] == patient_id_to_plot].sort_values('timestamp')

    # Filter for AE entities with a valid severity grade
    patient_aes = patient_data[
        patient_data['entity_type'].isin(AE_LABELS) &
        patient_data['severity_grade'].notna()
    ].copy() # Use .copy() to avoid SettingWithCopyWarning

    if patient_aes.empty:
        print(f"No AEs with severity found for Patient {patient_id_to_plot}.")
        return

    # Ensure severity grade is numeric
    patient_aes['severity_grade'] = pd.to_numeric(patient_aes['severity_grade'], errors='coerce')
    patient_aes.dropna(subset=['severity_grade'], inplace=True) # Drop if conversion failed

    if patient_aes.empty:
        print(f"No numeric severity grades found for Patient {patient_id_to_plot} after conversion.")
        return

    plt.figure(figsize=(14, max(6, patient_aes['entity_text'].nunique() * 0.5))) # Adjust height
    sns.scatterplot(
        data=patient_aes,
        x='timestamp',
        y='entity_text', # Use normalized AE name on y-axis
        size='severity_grade',
        hue='severity_grade',
        sizes=(50, 350), # Adjust size range
        palette='viridis_r', # Reverse viridis for severity (higher = brighter/yellower)
        legend='full'
    )
    plt.title(f'Figure 4: Normalized AE Timeline for Patient {patient_id_to_plot}')
    plt.xlabel('Date')
    plt.ylabel('Normalized Adverse Event')
    plt.legend(title='Severity Grade', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plt.show()


def plot_roc_curve(fpr, tpr, roc_auc):
    """Plots the ROC curve."""
    print("\nPlotting Figure 5: ROC Curve")
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Figure 5: Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right"); plt.grid(alpha=0.5); plt.tight_layout(); plt.show()

def plot_precision_recall_curve(recall, precision, avg_precision):
    """Plots the Precision-Recall curve."""
    print("\nPlotting Figure 6: Precision-Recall Curve")
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, where='post', color='blue', alpha=0.7, label=f'AUPRC = {avg_precision:.3f}')
    # Plot baseline (chance level) - proportion of positive class in test set could be added here
    # plt.fill_between(recall, precision, step='post', alpha=0.2, color='blue') # Optional fill
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05]); plt.xlim([0.0, 1.0])
    plt.title('Figure 6: Precision-Recall Curve')
    plt.legend(loc="lower left"); plt.grid(alpha=0.5); plt.tight_layout(); plt.show()

def plot_shap_summary(shap_values, processed_data, feature_names=None, max_display=20):
    """Plots SHAP summary plot (feature importance)."""
    if not SHAP_AVAILABLE or shap_values is None or processed_data is None:
        print("\nSkipping SHAP summary plot (SHAP not available or data missing).")
        return
    print(f"\nPlotting Figure 7: SHAP Summary Plot (Top {max_display})")
    try:
        plt.figure() # Let SHAP control figure size potentially
        shap.summary_plot(
            shap_values,
            processed_data,
            feature_names=feature_names,
            max_display=max_display,
            show=False # Control show() outside
        )
        plt.title(f'Figure 7: SHAP Feature Importance (Impact on model output)')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting SHAP summary: {e}")
        # Try plotting with default feature names if name mismatch might be the issue
        try:
             print("  Attempting SHAP plot with default feature indices...")
             plt.figure()
             shap.summary_plot(shap_values, processed_data, max_display=max_display, show=False)
             plt.title(f'Figure 7: SHAP Feature Importance (Indices)')
             plt.tight_layout()
             plt.show()
        except Exception as e2:
            print(f"  Second attempt at SHAP plot failed: {e2}")


def plot_calibration_curve(y_test, y_pred_proba, estimator, n_bins=10):
    """Plots the calibration curve."""
    print("\nPlotting Figure 8: Calibration Curve")
    try:
        plt.figure(figsize=(8, 8))
        ax_calibration = plt.gca()
        CalibrationDisplay.from_predictions(
            y_test,
            y_pred_proba,
            n_bins=n_bins,
            name=estimator.__class__.__name__, # Use model name
            ax=ax_calibration,
            strategy='uniform' # Or 'quantile'
        )
        plt.title('Figure 8: Calibration Curve (Reliability Diagram)')
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting calibration curve: {e}")


# --- 7. Main Execution Pipeline ---
if __name__ == "__main__":
    print("--- Starting Enhanced AE Monitoring Analysis Pipeline ---")
    pipeline_start_time = time.time()

    # --- Step 1: Load Data ---
    print("\n=== Step 1: Loading Data ===")
    try:
        patients_df = load_patient_data(PATIENT_DATA_PATH)
        notes_df = load_notes_data(NOTES_DATA_PATH)
        # Determine outcomes based on AE events
        notes_with_outcomes_df = determine_ae_outcomes(notes_df, AE_EVENTS_PATH, OUTCOME_WINDOW_DAYS)
        print(f"Target distribution ('severe_ae_in_period') in full dataset:\n{notes_with_outcomes_df['severe_ae_in_period'].value_counts(normalize=True, dropna=False)}")
    except FileNotFoundError:
        print("\nFATAL ERROR: One or more input files not found. Please check file paths in Configuration.")
        exit()
    except Exception as e:
        print(f"\nFATAL ERROR during data loading or outcome determination: {e}")
        exit()

    # --- Step 2: Extract Entities using Advanced NLP ---
    print("\n=== Step 2: Extracting Entities (NLP) ===")
    if not notes_with_outcomes_df.empty:
        # Process only notes needed (can optimize later if memory is an issue)
        nlp_results_df = extract_entities_advanced_nlp(notes_with_outcomes_df[['note_id', 'patient_id', 'timestamp', 'note_text']])
        print(f"Extracted {len(nlp_results_df)} affirmative entities.")
        if not nlp_results_df.empty:
            print("\nSample Normalized NLP Extraction Output (AEs with Severity):")
            print(nlp_results_df[nlp_results_df['entity_type'].isin(AE_LABELS) & nlp_results_df['severity_grade'].notna()].head())
        else:
            print("Warning: NLP extraction yielded no affirmative entities. Feature engineering might be limited.")
    else:
        print("No notes data available for NLP processing. Exiting.")
        nlp_results_df = pd.DataFrame()

    # --- Step 3: Advanced Feature Engineering ---
    print("\n=== Step 3: Engineering Features ===")
    if not notes_with_outcomes_df.empty and not nlp_results_df.empty:
        feature_df = create_advanced_features(
            nlp_results_df,
            notes_with_outcomes_df,
            patients_df, # Pass original df, indexing happens inside
            lookback_days=LOOKBACK_DAYS,
            recent_window_days=RECENT_WINDOW_DAYS
        )
        print(f"Created {feature_df.shape[1]-3} features for {feature_df.shape[0]} instances.") # Exclude id, patient_id, target
        print(f"Target distribution in final feature set:\n{feature_df['target'].value_counts(normalize=True, dropna=False)}")
    else:
        print("Skipping feature engineering due to missing notes or NLP results.")
        feature_df = pd.DataFrame()

    # --- Step 4: Train and Evaluate Advanced Model ---
    print("\n=== Step 4: Training and Evaluating Model ===")
    model = None
    metrics = {}
    shap_values = None
    X_test_processed = None
    feature_names_processed = None

    if not feature_df.empty and 'target' in feature_df.columns and feature_df['target'].nunique() > 1:
        model, metrics, shap_values, X_test_processed, feature_names_processed = train_evaluate_advanced_model(
            feature_df,
            random_state=RANDOM_STATE,
            test_size=TEST_SIZE,
            cv_folds=CV_FOLDS,
            n_iter_search=N_ITER_SEARCH
        )
        if model:
             print("\nModel training and evaluation completed.")
             # Optional: Save the best model
             # model_filename = f'advanced_ae_predictor_{datetime.now():%Y%m%d_%H%M%S}.joblib'
             # joblib.dump(model, model_filename)
             # print(f"Best model saved to {model_filename}")
        else:
             print("\nModel training failed or was skipped.")

    elif feature_df.empty:
        print("\nFeature DataFrame is empty. Skipping model training and evaluation.")
    elif 'target' not in feature_df.columns:
         print("\nTarget column missing in feature DataFrame. Skipping model training.")
    else: # Target has only one class
        print("\nTarget variable has only one class after feature engineering. Cannot train model.")


    # --- Step 5: Visualization ---
    print("\n=== Step 5: Generating Visualizations ===")
    # NLP Data Visualizations
    if not nlp_results_df.empty:
        try:
            plot_ae_frequency(nlp_results_df)
        except Exception as e: print(f"Error plotting AE frequency: {e}")
        try:
            plot_ae_severity(nlp_results_df)
        except Exception as e: print(f"Error plotting AE severity: {e}")
        try:
            plot_ae_by_drug(nlp_results_df)
        except Exception as e: print(f"Error plotting AE by drug: {e}")
        try:
             # Find a patient with some AEs for the timeline plot
             patients_with_aes = nlp_results_df[nlp_results_df['entity_type'].isin(AE_LABELS)]['patient_id'].unique()
             if len(patients_with_aes) > 0:
                  example_patient_id = random.choice(patients_with_aes)
                  print(f"Attempting timeline plot for example Patient ID: {example_patient_id}")
                  plot_patient_timeline(nlp_results_df, patient_id_to_plot=example_patient_id)
             else:
                  print("\nNo patients with detected AEs found for timeline plot example.")
        except Exception as e: print(f"Error plotting patient timeline: {e}")
    else:
        print("\nNo NLP results to generate NLP-based visualizations.")

    # Model Performance Visualizations
    if model and metrics:
        print("\nGenerating model performance plots...")
        try:
            if 'fpr' in metrics and 'tpr' in metrics:
                plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics.get('roc_auc', None))
            else: print("ROC data missing, skipping plot.")
        except Exception as e: print(f"Error plotting ROC curve: {e}")

        try:
            if 'recall' in metrics and 'precision' in metrics:
                 plot_precision_recall_curve(metrics['recall'], metrics['precision'], metrics.get('avg_precision', None))
            else: print("Precision-Recall data missing, skipping plot.")
        except Exception as e: print(f"Error plotting PR curve: {e}")

        try:
            # Use y_test and y_pred_proba which should be available if metrics were calculated
            if 'y_test' in locals() and 'y_pred_proba' in locals():
                 plot_calibration_curve(y_test, y_pred_proba, model)
            else:
                 print("Test labels or probabilities missing, skipping calibration plot.")
        except Exception as e: print(f"Error plotting calibration curve: {e}")

    else:
         print("\nNo trained model or metrics available for performance plotting.")

    # SHAP Plot
    if SHAP_AVAILABLE and shap_values is not None and X_test_processed is not None:
        print("\nGenerating SHAP summary plot...")
        try:
             plot_shap_summary(shap_values, X_test_processed, feature_names=feature_names_processed)
        except Exception as e: print(f"Error plotting SHAP summary: {e}")
    elif not SHAP_AVAILABLE:
         print("\nSHAP library not installed, skipping SHAP plot.")
    else:
         print("\nSHAP values or processed data not available, skipping SHAP plot.")


    # --- Final Comments ---
    pipeline_end_time = time.time()
    print(f"\n--- Enhanced Analysis Pipeline Finished in {pipeline_end_time - pipeline_start_time:.2f} seconds ---")
    print("\n** Next Steps & Considerations for Publication-Level Work **")
    print("- Deeper NLP: Relation Extraction (AE-Severity, AE-Drug), Temporal Normalization (TIMEX).")
    print("- Advanced Features: Embeddings (sentence/doc level), Feature interactions, External data (labs).")
    print("- Robust Validation: External validation set, More sophisticated temporal splits, Grouped CV thoroughness.")
    print("- Model Comparison: Compare tuned LGBM against XGBoost, Logistic Regression, potentially NNs.")
    print("- Interpretability: Individual SHAP plots, LIME, Deeper error analysis.")
    print("- Clinical Utility: Decision Curve Analysis, Subgroup analysis.")
    print("- Normalization: Map entities to standard ontologies (MedDRA, SNOMED CT) using more advanced tools.")
    print("- Parameter Tuning: Consider Bayesian Optimization (e.g., Optuna) for more efficient tuning.")
