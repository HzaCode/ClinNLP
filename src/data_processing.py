# src/data_processing.py
import pandas as pd
from datetime import datetime, timedelta
import time
import warnings

# Import configuration
from .config import OUTCOME_WINDOW_DAYS

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def load_patient_data(filepath):
    """Loads and performs basic cleaning on baseline patient data."""
    print(f"Loading patient data from: {filepath}")
    try:
        patients_df = pd.read_csv(filepath)
        # --- Add necessary cleaning/validation ---
        required_cols = ['patient_id', 'age', 'cancer_type', 'treatment_regimen']
        if not all(col in patients_df.columns for col in required_cols):
             # Handle missing columns - raise error or add defaults cautiously
             print(f"Warning: Missing one or more required columns in patient data: {required_cols}")
             # Example: Add baseline_risk_score if missing
             if 'baseline_risk_score' not in patients_df.columns and 'age' in patients_df.columns:
                 print("Adding proxy 'baseline_risk_score' based on age.")
                 patients_df['baseline_risk_score'] = (patients_df['age'] / 80).clip(0.1, 1.0) # Example proxy
             # Example: Add empty treatment_regimen if missing
             if 'treatment_regimen' not in patients_df.columns:
                  print("Adding empty 'treatment_regimen' column.")
                  patients_df['treatment_regimen'] = None

        # Ensure essential columns are present before proceeding
        if not all(col in patients_df.columns for col in ['patient_id', 'age', 'cancer_type']):
             raise ValueError("Essential columns 'patient_id', 'age', 'cancer_type' are missing from patient data.")

        # Clean treatment regimen (handle potential NaNs before splitting)
        if 'treatment_regimen' in patients_df.columns:
             patients_df['treatment_regimen'] = patients_df['treatment_regimen'].apply(
                 lambda x: x.split(';') if pd.notna(x) and isinstance(x, str) else ([] if pd.notna(x) else []) # Handle NaNs and non-strings
             )
        else: # Ensure column exists even if added above
             patients_df['treatment_regimen'] = [[] for _ in range(len(patients_df))]


        # Convert types
        patients_df['patient_id'] = patients_df['patient_id'].astype(str)
        patients_df['age'] = pd.to_numeric(patients_df['age'], errors='coerce') # Handle non-numeric ages

        print(f"Loaded and processed {len(patients_df)} patients.")
        return patients_df.dropna(subset=['patient_id', 'age']) # Drop rows with critical missing info

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        raise
    except Exception as e:
        print(f"Error processing patient data: {e}")
        raise


def load_notes_data(filepath):
    """Loads and performs basic cleaning on clinical notes."""
    print(f"Loading clinical notes from: {filepath}")
    try:
        notes_df = pd.read_csv(filepath)
        # --- Add necessary cleaning/validation ---
        required_cols = ['note_id', 'patient_id', 'timestamp', 'note_text']
        if not all(col in notes_df.columns for col in required_cols):
             raise ValueError(f"Missing one or more required columns in notes data: {required_cols}")

        notes_df['timestamp'] = pd.to_datetime(notes_df['timestamp'], errors='coerce')
        notes_df.dropna(subset=['timestamp', 'note_text', 'patient_id'], inplace=True) # Drop essential missing
        notes_df['note_text'] = notes_df['note_text'].astype(str)
        notes_df['note_id'] = notes_df['note_id'].astype(str)
        notes_df['patient_id'] = notes_df['patient_id'].astype(str)

        print(f"Loaded and processed {len(notes_df)} notes.")
        return notes_df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        raise
    except Exception as e:
        print(f"Error processing notes data: {e}")
        raise


def determine_ae_outcomes(notes_df, ae_events_filepath):
    """Determines target variable (severe_ae_in_period) from recorded AE events."""
    print(f"Determining outcomes using AE events from: {ae_events_filepath}")
    try:
        ae_events_df = pd.read_csv(ae_events_filepath)
        required_cols = ['patient_id', 'ae_timestamp', 'ae_grade']
        if not all(col in ae_events_df.columns for col in required_cols):
             raise ValueError(f"Missing one or more required columns in AE events data: {required_cols}")

        ae_events_df['ae_timestamp'] = pd.to_datetime(ae_events_df['ae_timestamp'], errors='coerce')
        ae_events_df['ae_grade'] = pd.to_numeric(ae_events_df['ae_grade'], errors='coerce')
        ae_events_df.dropna(subset=['patient_id', 'ae_timestamp', 'ae_grade'], inplace=True)
        ae_events_df['ae_grade'] = ae_events_df['ae_grade'].astype('Int64') # Keep as nullable Int
        ae_events_df['patient_id'] = ae_events_df['patient_id'].astype(str)

        # Sort for efficiency
        notes_df = notes_df.sort_values(['patient_id', 'timestamp'])
        ae_events_df = ae_events_df.sort_values(['patient_id', 'ae_timestamp'])

        # --- Efficient Outcome Mapping ---
        print(f"Mapping AEs within {OUTCOME_WINDOW_DAYS} days post-note...")
        start_time_outcome = time.time()

        # Create a helper Series for faster lookups within apply
        notes_df['window_end'] = notes_df['timestamp'] + timedelta(days=OUTCOME_WINDOW_DAYS)

        # Pre-group AE events by patient
        ae_events_grouped = ae_events_df.set_index('ae_timestamp').groupby('patient_id')

        target_list = []
        notes_processed_count = 0
        total_notes_for_outcome = len(notes_df)

        for index, note in notes_df.iterrows():
            patient_id = note['patient_id']
            note_timestamp = note['timestamp']
            window_end = note['window_end']
            target = 0 # Default: No severe AE

            if patient_id in ae_events_grouped.groups:
                patient_aes = ae_events_grouped.get_group(patient_id)
                # Filter AEs within the specific note's window
                # Add small offset to timestamp to ensure AE is *after* the note
                aes_in_window = patient_aes.loc[note_timestamp + timedelta(microseconds=1) : window_end]
                if not aes_in_window.empty:
                    # Check if any AE in the window is Grade 3 or higher
                    if (aes_in_window['ae_grade'] >= 3).any():
                        target = 1 # Severe AE occurred

            target_list.append(target)
            notes_processed_count += 1
            if notes_processed_count % 5000 == 0:
                 elapsed = time.time() - start_time_outcome
                 print(f"  ...processed outcomes for {notes_processed_count}/{total_notes_for_outcome} notes ({elapsed:.2f}s)")


        notes_df['severe_ae_in_period'] = target_list
        notes_df.drop(columns=['window_end'], inplace=True) # Clean up helper column

        # We keep all notes; target is now assigned
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

