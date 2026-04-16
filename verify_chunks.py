import json

with open(r"F:\ontology\chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Total chunks: {len(chunks)}\n")

# Show first 3 chunks as samples
for chunk in chunks[:3]:
    print("=" * 60)
    print(f"Chunk ID     : {chunk['chunk_id']}")
    print(f"Source       : {chunk['source_file']} (page {chunk['page']})")
    print(f"Discipline   : {chunk['discipline']}")
    print(f"Tags         : {chunk['ontology_tags']}")
    print(f"Text preview : {chunk['text'][:200]}...")
    print()