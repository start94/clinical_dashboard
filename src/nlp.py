import spacy
import streamlit as st

@st.cache_resource
def load_model():
    """
    Loads the scispaCy model.
    """
    try:
        nlp = spacy.load("en_core_sci_sm")
    except OSError:
        st.error("Model 'en_core_sci_sm' not found. Please download it by running the command from requirements.txt")
        st.stop()
    return nlp

def extract_entities(text: str):
    """
    Extracts clinical entities from a text.
    """
    nlp = load_model()
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
    return entities
