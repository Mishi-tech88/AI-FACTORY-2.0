# src/rag_pipeline.py
import os
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd

class RAGKnowledgeBase:
    def __init__(self, pdf_paths, chunk_size=500):
        self.chunk_size = chunk_size
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.generator_tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
        self.generator = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')
        self.chunks = []
        self.chunk_metadata = []  # source file, page, etc.
        self.index = None

        # Extract and chunk
        for pdf_path in pdf_paths:
            self._extract_pdf(pdf_path)

        if self.chunks:
            self._build_index()

    def _extract_pdf(self, pdf_path):
        print(f"Processing {pdf_path}...")
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            # Chunk by sentences or fixed size
            sentences = full_text.split('. ')
            current_chunk = ""
            for sent in sentences:
                if len(current_chunk) + len(sent) < self.chunk_size:
                    current_chunk += sent + ". "
                else:
                    if current_chunk:
                        self.chunks.append(current_chunk.strip())
                        self.chunk_metadata.append({'source': pdf_path, 'page': page_num})
                    current_chunk = sent + ". "
            if current_chunk:
                self.chunks.append(current_chunk.strip())
                self.chunk_metadata.append({'source': pdf_path, 'page': page_num})

    def _build_index(self):
        embeddings = self.embedder.encode(self.chunks, convert_to_numpy=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        print(f"Built FAISS index with {len(self.chunks)} chunks.")

    def retrieve(self, query, top_k=3):
        if self.index is None:
            return []
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(q_emb, top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append({
                'text': self.chunks[idx],
                'source': self.chunk_metadata[idx]['source'],
                'page': self.chunk_metadata[idx]['page'],
                'score': float(1/(1+dist))  # similarity score
            })
        return results

    def generate_answer(self, query, context_chunks, max_length=150):
        # Combine context and query
        context = " ".join([c['text'] for c in context_chunks])
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        inputs = self.generator_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.generator.generate(**inputs, max_length=max_length, num_beams=4, early_stopping=True)
        answer = self.generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

# ---- Integration with factory context ----
def explain_defect_prediction(defect_prob, severity, image_info=None, model_type="ResNet"):
    """Generate an explanation for a defect prediction using RAG."""
    rag = RAGKnowledgeBase(['data/raw/M1_manual.pdf', 'data/raw/SOP_general.pdf'])
    query = f"Why is this product predicted as defective with probability {defect_prob:.2f}? Severity: {severity:.2%}. Explain possible causes."
    retrieved = rag.retrieve(query, top_k=2)
    if not retrieved:
        return "No relevant knowledge found.", []
    explanation = rag.generate_answer(query, retrieved)
    return explanation, retrieved

def generate_maintenance_advice(issue, machine_id, urgency):
    """Generate maintenance advice based on issue and urgency."""
    rag = RAGKnowledgeBase(['data/raw/M1_manual.pdf', 'data/raw/SOP_general.pdf'])
    query = f"Machine {machine_id} has {issue} with urgency {urgency}. What maintenance actions should be taken according to SOPs?"
    retrieved = rag.retrieve(query, top_k=2)
    if not retrieved:
        return "No relevant SOP found.", []
    advice = rag.generate_answer(query, retrieved)
    return advice, retrieved

if __name__ == "__main__":
    # Test
    print("Building RAG knowledge base...")
    rag = RAGKnowledgeBase(['data/raw/M1_manual.pdf', 'data/raw/SOP_general.pdf'])

    # Test retrieval
    query = "What is the procedure for an error code 42?"
    results = rag.retrieve(query)
    print("\nRetrieved chunks:")
    for r in results:
        print(f"- {r['text'][:100]}... (source: {r['source']})")

    # Test generation
    answer = rag.generate_answer(query, results)
    print("\nGenerated answer:")
    print(answer)