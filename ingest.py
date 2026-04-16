import os
import json
from datetime import datetime
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rdflib import Graph, SKOS

# ─────────────────────────────────────────
# CONFIGURATION — update these paths
# ─────────────────────────────────────────
PDF_FOLDER      = r"F:\ontology\Construction"   # ✅ FIXED HERE
ONTOLOGY_FILE   = r"F:\ontology\arupmind_ontology.ttl"
OUTPUT_FILE     = r"F:\ontology\chunks.json"
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 64

# ─────────────────────────────────────────
# STEP A — Load ontology and build keyword map
# ─────────────────────────────────────────
print("Loading ontology...")
g = Graph()
g.parse(ONTOLOGY_FILE, format="turtle")

keyword_map = {}

for s, p, o in g.triples((None, SKOS.prefLabel, None)):
    keyword_map[str(o).lower()] = str(s).split("#")[-1]

for s, p, o in g.triples((None, SKOS.altLabel, None)):
    keyword_map[str(o).lower()] = str(s).split("#")[-1]

print(f"Loaded {len(keyword_map)} keywords from ontology\n")

# ─────────────────────────────────────────
# STEP B — Tag function
# ─────────────────────────────────────────
def tag_chunk(chunk_text):
    tags = []
    text_lower = chunk_text.lower()
    for keyword, concept in keyword_map.items():
        if keyword in text_lower:
            tags.append(concept)
    return list(set(tags))

# ─────────────────────────────────────────
# STEP C — Parse PDF
# ─────────────────────────────────────────
def parse_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "page_num": page_num,
                "text": text.strip()
            })
    return pages

# ─────────────────────────────────────────
# STEP D — Chunking
# ─────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " "]
)

# ─────────────────────────────────────────
# STEP E — Discipline mapping
# ─────────────────────────────────────────
DISCIPLINE_MAP = {
    "SeismicDesign": "Structural Engineering",
    "LoadAnalysis": "Structural Engineering",
    "WindDesign": "Structural Engineering",
    "StructuralFailure": "Structural Engineering",
    "FoundationSystems": "Structural Engineering",
    "StructuralEngineering": "Structural Engineering",

    "FireDetection": "Fire Safety",
    "FireSuppression": "Fire Safety",
    "FireResistantMaterials": "Fire Safety",
    "SmokeControl": "Fire Safety",
    "EgressDesign": "Fire Safety",
    "FireSafety": "Fire Safety",

    "PowerDistribution": "Electrical",
    "CircuitProtection": "Electrical",
    "CableManagement": "Electrical",
    "ElectricalHazards": "Electrical",
    "ElectricalSafety": "Electrical",
    "GroundingAndBonding": "Electrical",
    "Electrical": "Electrical",

    "BuildingMaterials": "Construction",
    "ConcreteStructures": "Construction",
    "SteelStructures": "Construction",
    "Construction_safety": "Construction",
    "Construction": "Construction",
}

def infer_discipline(tags):
    if not tags:
        return "General"
    discipline_counts = {}
    for tag in tags:
        discipline = DISCIPLINE_MAP.get(tag, "General")
        discipline_counts[discipline] = discipline_counts.get(discipline, 0) + 1
    return max(discipline_counts, key=discipline_counts.get)

# ─────────────────────────────────────────
# STEP F — Main loop
# ─────────────────────────────────────────
all_chunks = []
chunk_id = 0
today = datetime.today().strftime("%Y-%m-%d")

pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
print(f"Found {len(pdf_files)} PDF files to process\n")

for pdf_file in pdf_files:
    pdf_path = os.path.join(PDF_FOLDER, pdf_file)
    print(f"Processing: {pdf_file}")

    try:
        pages = parse_pdf(pdf_path)
    except Exception as e:
        print(f"  ERROR reading {pdf_file}: {e}")
        continue

    for page in pages:
        chunks = splitter.split_text(page["text"])

        for chunk_text in chunks:
            if len(chunk_text.strip()) < 50:
                continue

            tags = tag_chunk(chunk_text)

            chunk_doc = {
                "chunk_id": chunk_id,
                "text": chunk_text.strip(),
                "source_file": pdf_file,
                "page": page["page_num"],
                "ontology_tags": tags,
                "discipline": infer_discipline(tags),
                "doc_type": "research_paper",
                "date_ingested": today,
                "tag_count": len(tags)
            }

            all_chunks.append(chunk_doc)
            chunk_id += 1

    print(f"  Pages: {len(pages)} | Chunks so far: {chunk_id}")

# ─────────────────────────────────────────
# STEP G — Save output
# ─────────────────────────────────────────
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print("\n✅ DONE!")
print(f"Total chunks created : {len(all_chunks)}")
print(f"Output saved to      : {OUTPUT_FILE}")

# ─────────────────────────────────────────
# STEP H — Stats
# ─────────────────────────────────────────
tagged = sum(1 for c in all_chunks if c["tag_count"] > 0)
untagged = sum(1 for c in all_chunks if c["tag_count"] == 0)

discipline_counts = {}
for c in all_chunks:
    d = c["discipline"]
    discipline_counts[d] = discipline_counts.get(d, 0) + 1

print("\n=== Stats ===")
print(f"Tagged chunks   : {tagged}")
print(f"Untagged chunks : {untagged}")

for discipline, count in sorted(discipline_counts.items()):
    print(f"{discipline:30} → {count} chunks")