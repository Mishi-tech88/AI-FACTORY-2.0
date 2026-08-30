# src/integration_demo.py
from matplotlib import transforms
import pandas as pd
from vision_advanced import load_resnet_model, generate_gradcam, Autoencoder, anomaly_score, estimate_severity
from nlp_pipeline import process_all_notes, build_similarity_index, find_similar_notes
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Load models (you need to train autoencoder first)
resnet_model = load_resnet_model()
# For autoencoder, you need to load a trained one; for demo we create a dummy
# In practice: ae = torch.load('models/autoencoder.pth')
# We'll skip AE for now and just use severity.

def inspect_product(image_path):
    # Classification
    transform = transforms.Compose([...])  # as before
    img = transform(Image.open(image_path).convert('L')).unsqueeze(0)
    with torch.no_grad():
        pred = torch.sigmoid(resnet_model(img)).item()
    defect = pred > 0.5
    # Severity
    severity = estimate_severity(image_path)
    # Grad-CAM
    heatmap = generate_gradcam(image_path, resnet_model)
    return {
        'defect_prob': pred,
        'is_defective': defect,
        'severity_ratio': severity,
        'heatmap': heatmap
    }

def process_maintenance_note(note_text, df_notes, vectorizer, tfidf):
    # Extract entities
    from nlp_pipeline import extract_entities
    info = extract_entities(note_text)
    # Similar notes
    similar = find_similar_notes(note_text, vectorizer, tfidf, df_notes)
    return info, similar

if __name__ == "__main__":
    # Load notes and build index
    df_notes = pd.read_csv('data/raw/maintenance_notes.csv')
    df_notes = process_all_notes(df_notes)
    vectorizer, tfidf = build_similarity_index(df_notes['maintenance_note'])

    # Test with an image
    img_path = 'data/images/prod_0.png'  # pick one
    result = inspect_product(img_path)
    print(f"Defect prob: {result['defect_prob']:.2f}, Severity: {result['severity_ratio']:.2%}")
    # Show heatmap
    plt.imshow(result['heatmap'])
    plt.show()

    # Test with a note
    note = "Machine M3 has an oil leak and unusual vibration. Operator John."
    info, similar = process_maintenance_note(note, df_notes, vectorizer, tfidf)
    print("Extracted info:", info)
    print("Similar notes:", similar)