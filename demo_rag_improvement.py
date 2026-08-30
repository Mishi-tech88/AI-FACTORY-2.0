# demo_rag_improvement.py
from src.rag_pipeline import RAGKnowledgeBase

def demo_without_retrieval(query):
    # Use a generic LLM (no context)
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
    model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')
    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=100, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    rag = RAGKnowledgeBase(['data/raw/M1_manual.pdf', 'data/raw/SOP_general.pdf'])
    query = "What should be done if Machine M1 overheats above 90°C?"

    # Without retrieval (just raw LLM)
    ans_no_rag = demo_without_retrieval(query)
    print("LLM without retrieval:")
    print(ans_no_rag)

    # With retrieval
    retrieved = rag.retrieve(query, top_k=2)
    ans_with_rag = rag.generate_answer(query, retrieved)
    print("\nLLM with RAG:")
    print(ans_with_rag)

    print("\nRetrieved source:")
    for r in retrieved:
        print(f"- {r['source']} (score: {r['score']:.2f})")