import os
from dotenv import load_dotenv
from google import genai

from src.document_engine.pdf_reader import extract_text_from_pdf
from src.document_engine.chunker import chunk_text
from src.document_engine.embedder import model, create_embeddings
from src.document_engine.search import search_chunks


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


# 1. Load PDF documents
pdf_files = [
    os.path.join("data", file)
    for file in os.listdir("data")
    if file.lower().endswith(".pdf")
]

if not pdf_files:
    print("No PDF documents found in the data folder.")
    exit()

documents = []

for pdf_path in pdf_files:
    print(f"Loading: {pdf_path}")

    text = extract_text_from_pdf(pdf_path)

    documents.append({
        "source": os.path.basename(pdf_path),
        "text": text
    })


# 2. Split documents into chunks
chunks = []
sources = []

for document in documents:
    document_chunks = chunk_text(document["text"])

    chunks.extend(document_chunks)
    sources.extend([document["source"]] * len(document_chunks))


# 3. Create embeddings
embeddings = create_embeddings(chunks)


# 4. Ask a question
query = input("\nAsk LifeLens AI: ")


# 5. Search relevant chunks
results = search_chunks(
    query,
    chunks,
    embeddings,
    model,
    sources,
    top_k=2
)


# 6. Prepare context for Gemini
context = "\n\n".join(
    f"Source: {result['source']}\n{result['chunk']}"
    for result in results
)

# 7. Ask Gemini using the retrieved context
prompt = f"""
You are LifeLens AI, a personal AI assistant.

Answer the user's question using ONLY the information provided
in the document context below.

If the answer is not present in the context, say:
"I couldn't find that information in your document."

Document context:
{context}

User question:
{query}
"""


response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=prompt,
    config={
        "automatic_function_calling": {
            "disable": True
        }
    }
)

print("\n----- LIFELENS AI -----")
print(response.text)