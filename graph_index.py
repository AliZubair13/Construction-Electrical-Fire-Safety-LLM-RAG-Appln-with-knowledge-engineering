import json
from neo4j import GraphDatabase

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
CHUNKS_FILE = r"F:\ontology\chunks.json"
NEO4J_URI   = "bolt://localhost:7687"
NEO4J_USER  = "neo4j"
NEO4J_PASS  = "password"   # ← whatever you set in Neo4j Desktop
BATCH_SIZE  = 100

# ─────────────────────────────────────────
# STEP A — Connect to Neo4j
# ─────────────────────────────────────────
print("Connecting to Neo4j...")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# Test connection
with driver.session() as session:
    result = session.run("RETURN 'Connected!' AS msg")
    print(result.single()["msg"])

# ─────────────────────────────────────────
# STEP B — Load chunks
# ─────────────────────────────────────────
print("\nLoading chunks...")
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"Loaded {len(chunks)} chunks")

# ─────────────────────────────────────────
# STEP C — Clear existing data (fresh start)
# ─────────────────────────────────────────
print("\nClearing existing graph data...")
with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")
print("Graph cleared")

# ─────────────────────────────────────────
# STEP D — Create constraints for speed
# ─────────────────────────────────────────
with driver.session() as session:
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (concept:Concept) REQUIRE concept.name IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Source) REQUIRE s.filename IS UNIQUE")
print("Constraints created")

# ─────────────────────────────────────────
# STEP E — Insert chunks in batches
# ─────────────────────────────────────────
def insert_batch(tx, batch):
    tx.run("""
        UNWIND $chunks AS chunk

        MERGE (c:Chunk {chunk_id: chunk.chunk_id})
        SET c.text      = chunk.text,
            c.page      = chunk.page,
            c.discipline = chunk.discipline,
            c.tag_count = chunk.tag_count

        MERGE (s:Source {filename: chunk.source_file})
        MERGE (c)-[:FROM_SOURCE]->(s)

        WITH c, chunk
        UNWIND chunk.ontology_tags AS tag
        MERGE (concept:Concept {name: tag})
        MERGE (c)-[:TAGGED_WITH]->(concept)
    """, chunks=batch)

print(f"\nInserting {len(chunks)} chunks into Neo4j...")
for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i : i + BATCH_SIZE]
    with driver.session() as session:
        session.execute_write(insert_batch, batch)

    done = min(i + BATCH_SIZE, len(chunks))
    print(f"  Inserted {done}/{len(chunks)} chunks...")

print("\n✅ All chunks inserted!")

# ─────────────────────────────────────────
# STEP F — Verify with sample queries
# ─────────────────────────────────────────
print("\n=== Graph Stats ===")
with driver.session() as session:
    result = session.run("MATCH (c:Chunk) RETURN count(c) AS total")
    print(f"  Total Chunk nodes    : {result.single()['total']}")

    result = session.run("MATCH (s:Source) RETURN count(s) AS total")
    print(f"  Total Source nodes   : {result.single()['total']}")

    result = session.run("MATCH (concept:Concept) RETURN count(concept) AS total")
    print(f"  Total Concept nodes  : {result.single()['total']}")

    result = session.run("MATCH ()-[r:TAGGED_WITH]->() RETURN count(r) AS total")
    print(f"  Total relationships  : {result.single()['total']}")

# ─────────────────────────────────────────
# STEP G — Test a graph query
# ─────────────────────────────────────────
print("\n=== Test Query: Find chunks about SeismicDesign ===")
with driver.session() as session:
    result = session.run("""
        MATCH (c:Chunk)-[:TAGGED_WITH]->(concept:Concept {name: 'SeismicDesign'})
        RETURN c.chunk_id, c.text, c.page
        LIMIT 3
    """)
    for record in result:
        print(f"  Chunk {record['c.chunk_id']} (page {record['c.page']})")
        print(f"  {record['c.text'][:150]}...\n")

driver.close()
print("✅ Stage 4B — Neo4j complete!")