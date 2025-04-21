# src/nlp_extraction.py
import spacy
import spacy_transformers # Required for en_core_web_trf
from spacy.matcher import Matcher
from spacy.tokens import Span
import pandas as pd
import time
import warnings

# Import configuration and check optional dependencies
from .config import (NLP_MODEL_NAME, SEVERITY_PATTERNS, SEVERITY_TERMS,
                     AE_NORMALIZATION_MAP, AE_LABELS, DRUG_LABELS, NLP_BATCH_SIZE,
                     NEGSPACY_AVAILABLE, SHAP_AVAILABLE) # Import flags too

# Check and update flags based on actual imports
try:
    from negspacy.negation import Negex
    # NEGSPACY_AVAILABLE = True # Already assumed True in config, confirm here
except ImportError:
    print("Warning: negspacy not found. Negation detection will be skipped. Install with: pip install negspacy")
    NEGSPACY_AVAILABLE = False # Update flag if import fails

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Global NLP object (Load once) ---
NLP = None
MATCHER = None

def setup_nlp():
    """Loads the spaCy model, sets up matcher and negex."""
    global NLP, MATCHER
    if NLP is not None:
        print("NLP model already loaded.")
        return NLP, MATCHER

    print(f"\nLoading spaCy NLP model: {NLP_MODEL_NAME}...")
    if NLP_MODEL_NAME.endswith("_trf"):
        try:
            if spacy.prefer_gpu(): print("GPU available, spaCy Transformer model will use it.")
            else: print("Warning: GPU not detected. Transformer model will run on CPU (VERY SLOW!).")
        except Exception as e: print(f"GPU check failed: {e}. Running on CPU.")

    try:
        NLP = spacy.load(NLP_MODEL_NAME)
    except OSError:
        print(f"Model {NLP_MODEL_NAME} not found. Downloading...")
        spacy.cli.download(NLP_MODEL_NAME)
        NLP = spacy.load(NLP_MODEL_NAME)

    # Add Matcher for severity terms
    MATCHER = Matcher(NLP.vocab)
    for term in SEVERITY_PATTERNS:
        pattern = [{"LOWER": word} for word in term.split()]
        MATCHER.add(f"SEVERITY_{term.upper()}", [pattern])
    print("Severity Matcher initialized.")

    # Add Negation Detection pipe (if available)
    if NEGSPACY_AVAILABLE:
        print("Adding negSpacy negation detection pipe...")
        # Check if pipe exists to avoid adding multiple times
        if "negex" not in NLP.pipe_names:
            # Configure negex for relevant entity types (use AE_LABELS from config)
            negex = Negex(NLP, name="negex", ent_types=AE_LABELS)
            NLP.add_pipe("negex", last=True)
            print("negex pipe added.")
        else:
            print("negex pipe already exists.")
    else:
        print("Skipping negation detection (negspacy not installed/failed to import).")

    return NLP, MATCHER


def link_ae_severity_improved(doc, ae_ents, severity_matches):
    """Links AE entities with nearby severity terms (simple proximity)."""
    # (Keep the implementation from the original script)
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
                 sev_text = sev_span.text.lower()
                 if sev_text in SEVERITY_TERMS:
                     linked_severity_grade = SEVERITY_TERMS[sev_text]
                     linked_severity_term = sev_text
                     min_dist_after = sev_span.start - ae.end # Found closer one
                     # Break if you only want the *closest* following term
                     # break

            # Check if severity precedes AE closely
            elif ae.start >= sev_span.end and (ae.start - sev_span.end) < min_dist_before:
                 sev_text = sev_span.text.lower()
                 if sev_text in SEVERITY_TERMS:
                      # Only update if no 'after' severity was found
                      if linked_severity_grade is None:
                           linked_severity_grade = SEVERITY_TERMS[sev_text]
                           linked_severity_term = sev_text
                           # Don't break, an 'after' term might be closer/preferred

        linked_data[ae] = {'severity_grade': linked_severity_grade, 'severity_term': severity_term}
    return linked_data


def extract_entities_advanced_nlp(notes_df):
    """Extracts entities using spaCy, applies normalization, and skips negated AEs."""
    nlp, matcher = setup_nlp() # Ensure NLP model is loaded
    if nlp is None or matcher is None:
        raise RuntimeError("NLP model setup failed.")

    print(f"\nExtracting entities using {NLP_MODEL_NAME} with Negation & Normalization...")
    extracted_results = []
    start_time = time.time()

    # Select only necessary columns for processing
    notes_subset = notes_df[['note_id', 'patient_id', 'timestamp', 'note_text']].copy()
    texts = notes_subset['note_text'].tolist()
    note_ids = notes_subset['note_id'].tolist()
    patient_ids = notes_subset['patient_id'].tolist()
    timestamps = notes_subset['timestamp'].tolist()

    total_notes = len(texts)
    processed_count = 0
    affirmative_ae_count = 0
    negated_ae_count = 0

    # Using nlp.pipe for efficient processing
    print(f"Processing {total_notes} notes in batches of {NLP_BATCH_SIZE}...")
    docs = nlp.pipe(texts, batch_size=NLP_BATCH_SIZE)

    for i, doc in enumerate(docs):
        note_id = note_ids[i]
        patient_id = patient_ids[i]
        timestamp = timestamps[i]

        # Get entities & filter for relevant types
        all_entities = doc.ents
        ae_entities = [ent for ent in all_entities if ent.label_ in AE_LABELS]
        drug_entities = [ent for ent in all_entities if ent.label_ in DRUG_LABELS]

        # Match severity terms
        severity_matches = matcher(doc)
        ae_severity_links = link_ae_severity_improved(doc, ae_entities, severity_matches)

        # Store results, applying normalization and checking negation
        relevant_entities = ae_entities + drug_entities
        for ent in relevant_entities:
            entity_type = ent.label_
            entity_text_original = ent.text.lower().strip() # Lowercase and strip whitespace
            normalized_entity_text = entity_text_original # Default
            severity_grade = None
            severity_term = None
            is_negated = False

            # Process AEs
            if entity_type in AE_LABELS:
                if NEGSPACY_AVAILABLE and hasattr(ent._, 'negex'):
                    is_negated = ent._.negex
                else:
                    is_negated = False # Assume not negated

                if is_negated:
                    negated_ae_count += 1
                    continue # *** SKIP NEGATED AE MENTIONS ***

                affirmative_ae_count += 1
                normalized_entity_text = AE_NORMALIZATION_MAP.get(entity_text_original, entity_text_original)

                if ent in ae_severity_links:
                    severity_grade = ae_severity_links[ent]['severity_grade']
                    severity_term = ae_severity_links[ent]['severity_term']

            # Process Drugs (simple normalization, could be expanded)
            elif entity_type in DRUG_LABELS:
                 normalized_entity_text = entity_text_original # Basic, maybe map to generics later?

            # Only append affirmative AEs or Drugs
            extracted_results.append({
                'note_id': note_id,
                'patient_id': patient_id,
                'timestamp': timestamp,
                'entity_type': entity_type,
                'entity_text': normalized_entity_text, # Use normalized text
                'entity_text_original': entity_text_original,
                'severity_grade': severity_grade,
                'severity_term': severity_term,
            })

        processed_count += 1
        if processed_count % 200 == 0 or processed_count == total_notes:
            elapsed = time.time() - start_time
            print(f"  Processed {processed_count}/{total_notes} notes ({elapsed:.2f} seconds)")

    end_time = time.time()
    print(f"NLP extraction finished in {end_time - start_time:.2f} seconds.")
    print(f"  Found {affirmative_ae_count} affirmative AE mentions.")
    if NEGSPACY_AVAILABLE:
        print(f"  Skipped {negated_ae_count} negated AE mentions.")
    else:
        print("  Negation detection was skipped.")

    if not extracted_results:
         print("Warning: No affirmative entities were extracted.")
         return pd.DataFrame(columns=['note_id', 'patient_id', 'timestamp', 'entity_type', 'entity_text', 'entity_text_original', 'severity_grade', 'severity_term'])

    return pd.DataFrame(extracted_results)

