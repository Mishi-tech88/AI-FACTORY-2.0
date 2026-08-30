# src/nlp_pipeline.py
import pandas as pd
import re
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load('en_core_web_sm')

# Patterns
MACHINE_PATTERN = re.compile(r'Machine\s*([A-Z]\d)')
ISSUE_KEYWORDS = ['oil leak', 'overheating', 'unusual vibration', 'noise', 'error code']
ACTION_KEYWORDS = ['replaced filter', 'cleaned sensor', 'tightened bolts', 'reset controller']

def extract_entities(note):
    doc = nlp(note)
    # Machine ID
    machine_match = MACHINE_PATTERN.search(note)
    machine = machine_match.group(1) if machine_match else None

    # Issue and action using keyword matching
    issue = None
    for kw in ISSUE_KEYWORDS:
        if kw in note.lower():
            issue = kw
            break
    action = None
    for kw in ACTION_KEYWORDS:
        if kw in note.lower():
            action = kw
            break

    # Urgency: if "error code 42" or "overheating" -> high; "noise" or "unusual vibration" -> medium; else low
    urgency = 'low'
    if 'error code 42' in note.lower() or 'overheating' in note.lower():
        urgency = 'high'
    elif 'unusual vibration' in note.lower() or 'noise' in note.lower():
        urgency = 'medium'

    return {
        'machine': machine,
        'issue': issue,
        'action': action,
        'urgency': urgency
    }

def process_all_notes(df_notes):
    """Adds extracted columns to the dataframe."""
    extracted = df_notes['maintenance_note'].apply(extract_entities)
    df_extracted = pd.DataFrame(extracted.tolist())
    return pd.concat([df_notes, df_extracted], axis=1)

# ---- Similarity search ----
def build_similarity_index(notes):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(notes)
    return vectorizer, tfidf_matrix

def find_similar_notes(query, vectorizer, tfidf_matrix, notes_df, top_k=3):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return notes_df.iloc[top_indices][['maintenance_note', 'urgency']]

# ---- Use a small transformer for classification (optional) ----
# We can fine‑tune a BERT for urgency classification, but for demo we use rules.

if __name__ == "__main__":
    df = pd.read_csv('data/raw/maintenance_notes.csv')
    df_enhanced = process_all_notes(df)
    df_enhanced.to_csv('data/processed/maintenance_enhanced.csv', index=False)
    print("Enhanced notes saved.")

    # Build similarity index on all notes
    vectorizer, tfidf = build_similarity_index(df_enhanced['maintenance_note'])
    # Example query
    query = "Machine M2 is making a strange noise and overheating."
    similar = find_similar_notes(query, vectorizer, tfidf, df_enhanced)
    print("\nTop similar past notes:")
    print(similar)