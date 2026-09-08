import numpy as np


def search_chunks(query, chunks, embeddings, model, sources, top_k=2):
    query_embedding = model.encode([query])[0]

    scores = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1)
        * np.linalg.norm(query_embedding)
    )

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append({
            "chunk": chunks[index],
            "score": float(scores[index]),
            "source": sources[index]
        })

    return results