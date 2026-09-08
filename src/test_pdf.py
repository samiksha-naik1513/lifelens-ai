from document_engine.pdf_reader import extract_text_from_pdf
from document_engine.chunker import chunk_text
from document_engine.embedder import model, create_embeddings
from document_engine.search import search_chunks


pdf_path = "data/Samikshadnaik.pdf"

text = extract_text_from_pdf(pdf_path)

chunks = chunk_text(text)

embeddings = create_embeddings(chunks)

query = "What programming languages does Samiksha know?"

results = search_chunks(
    query,
    chunks,
    embeddings,
    model
)

print("\n----- SEARCH RESULTS -----")

for i, result in enumerate(results):
    print(f"\nResult {i + 1}")
    print("Score:", result["score"])
    print("Text:", result["chunk"])