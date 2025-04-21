# src/config.py
import os
from datetime import timedelta

# --- File Paths ---
# __file__ is the path to this config.py file (e.g., C:\...\ClinNLP\src\config.py)
# os.path.dirname(__file__) is the directory containing config.py (e.g., C:\...\ClinNLP\src)
# os.path.join(..., '..') goes up one level to the project root (e.g., C:\...\ClinNLP)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"Project Base Directory determined as: {BASE_DIR}") # Add print for verification

DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'notebooks') # Added for completeness

PATIENT_DATA_PATH = os.path.join(DATA_DIR, 'patient_baseline_data.csv') # UPDATE if filename differs
NOTES_DATA_PATH = os.path.join(DATA_DIR, 'clinical_notes.csv')       # UPDATE if filename differs
AE_EVENTS_PATH = os.path.join(DATA_DIR, 'recorded_ae_events.csv')   # UPDATE if filename differs

# Ensure output directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# --- Feature Engineering Parameters ---
LOOKBACK_DAYS = 90
RECENT_WINDOW_DAYS = 14
OUTCOME_WINDOW_DAYS = 30

# --- NLP Model ---
NLP_MODEL_NAME = "en_core_web_lg" # Advanced (Requires GPU for reasonable speed!)
# NLP_MODEL_NAME = "en_core_web_lg" # Large non-transformer alternative
NLP_BATCH_SIZE = 16 if NLP_MODEL_NAME.endswith("_trf") else 64

# --- Definitions (CRITICAL: Needs Clinical Input) ---
DRUGS = ["Cisplatin", "Paclitaxel", "Pembrolizumab", "Letrozole", "Trastuzumab", "Oxaliplatin"] # Example - EXPAND/MODIFY
AES = ["Nausea", "Vomiting", "Fatigue", "Anemia", "Neutropenia", "Thrombocytopenia", "Diarrhea", "Rash", "Neuropathy", "Pneumonitis", "Mucositis"] # Example - EXPAND/MODIFY

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
    # Add many more mappings based on data exploration and CLINICAL EXPERTISE
}
# Severity remains unchanged
SEVERITY_TERMS = { "grade 1": 1, "g1": 1, "mild": 1, "grade 2": 2, "g2": 2, "moderate": 2, "grade 3": 3, "g3": 3, "severe": 3, "grade 4": 4, "g4": 4, "life-threatening": 4, "grade 5": 5, "g5": 5, "death": 5 }
SEVERITY_PATTERNS = list(SEVERITY_TERMS.keys())

# Standardized AE Labels (used by Negex and potentially the model)
AE_LABELS = ["AE", "ADVERSE_EVENT", "SYMPTOM"] # Adjust based on your spaCy model's output/training
DRUG_LABELS = ["DRUG", "CHEMICAL"] # Adjust based on spaCy model

# --- Machine Learning Model Parameters ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5 # Folds for cross-validation during tuning
N_ITER_SEARCH = 25 # Number of parameter settings sampled by RandomizedSearchCV

# --- Visualization Parameters ---
TOP_N_AE_FREQ = 15
TOP_N_DRUGS_COOCCUR = 5
TOP_N_AES_COOCCUR = 10
SHAP_MAX_DISPLAY = 20
CALIBRATION_N_BINS = 10

# --- Misc ---
# Flags to be checked dynamically in relevant modules
NEGSPACY_AVAILABLE = True # Default assumption, checked in nlp_extraction.py
SHAP_AVAILABLE = True     # Default assumption, checked in modeling.py

