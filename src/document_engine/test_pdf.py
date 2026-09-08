from document_engine.pdf_reader import extract_text_from_pdf
from document_engine.chunker import chunk_text
from document_engine.embedder import create_embeddings

pdf_path = "data/Samikshadnaik.pdf"

text = extract_text_from_pdf(pdf_path)

chunks = chunk_text(text)

embeddings = create_embeddings(chunks)

print("Total chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)