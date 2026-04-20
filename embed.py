import json
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
CHUNKS_FILE    = r"F:\ontology\chunks.json"
FAISS_INDEX    = r"F:\ontology\arupmind.index"
METADATA_FILE  = r"F:\ontology\metadata.pkl"
MODEL_NAME     = "all-MiniLM-L6-v2"
BATCH_SIZE     = 64   # process 64 chunks at a time
DIM            = 384  # MiniLM output dimension

# ─────────────────────────────────────────
# STEP A — Load chunks
# ─────────────────────────────────────────
print("Loading chunks...")
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"Loaded {len(chunks)} chunks\n")

# ─────────────────────────────────────────
# STEP B — Load embedding model
# ─────────────────────────────────────────
print("Loading embedding model (downloads on first run ~90MB)...")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded\n")

# ─────────────────────────────────────────
# STEP C — Generate embeddings in batches
# ─────────────────────────────────────────
texts = [chunk["text"] for chunk in chunks]
all_embeddings = []

print(f"Embedding {len(texts)} chunks in batches of {BATCH_SIZE}...")
for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i : i + BATCH_SIZE]
    batch_embeddings = model.encode(batch, show_progress_bar=False)
    all_embeddings.append(batch_embeddings)

    # Progress update every 10 batches
    if (i // BATCH_SIZE) % 10 == 0:
        done = min(i + BATCH_SIZE, len(texts))
        print(f"  Embedded {done}/{len(texts)} chunks...")

embeddings = np.vstack(all_embeddings).astype("float32")
print(f"\nAll embeddings shape: {embeddings.shape}")
# Should print: (5798, 384)

# ─────────────────────────────────────────
# STEP D — Normalize vectors (for cosine similarity)
# ─────────────────────────────────────────
faiss.normalize_L2(embeddings)

# ─────────────────────────────────────────
# STEP E — Build FAISS index
# ─────────────────────────────────────────
print("\nBuilding FAISS index...")
index = faiss.IndexFlatIP(DIM)  # Inner Product = cosine similarity after normalization
index.add(embeddings)
print(f"FAISS index built — {index.ntotal} vectors stored")

# ─────────────────────────────────────────
# STEP F — Save FAISS index to disk
# ─────────────────────────────────────────
faiss.write_index(index, FAISS_INDEX)
print(f"FAISS index saved → {FAISS_INDEX}")

# ─────────────────────────────────────────
# STEP G — Save metadata (chunk info) to disk
# ─────────────────────────────────────────
metadata = [{
    "chunk_id":      c["chunk_id"],
    "source_file":   c["source_file"],
    "page":          c["page"],
    "ontology_tags": c["ontology_tags"],
    "discipline":    c["discipline"],
    "text":          c["text"]
} for c in chunks]

with open(METADATA_FILE, "wb") as f:
    pickle.dump(metadata, f)
print(f"Metadata saved → {METADATA_FILE}")

# ─────────────────────────────────────────
# STEP H — Quick search test
# ─────────────────────────────────────────
print("\n=== Quick Search Test ===")
test_query = "earthquake proofing for hospital buildings"
print(f"Query: '{test_query}'\n")

query_vec = model.encode([test_query]).astype("float32")
faiss.normalize_L2(query_vec)

D, I = index.search(query_vec, k=3)  # get top 3 results

for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
    chunk = metadata[idx]
    print(f"Result #{rank}")
    print(f"  Score      : {score:.4f}")
    print(f"  Source     : {chunk['source_file']} (page {chunk['page']})")
    print(f"  Discipline : {chunk['discipline']}")
    print(f"  Tags       : {chunk['ontology_tags']}")
    print(f"  Text       : {chunk['text'][:200]}...")
    print()

print("✅ Stage 4A — FAISS complete!")