import os
import json
from datetime import datetime
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rdflib import Graph, SKOS

# ─────────────────────────────────────────
# CONFIGURATION — update these paths
# ─────────────────────────────────────────
PDF_FOLDER    = r"F:\ontology\Construction"    # ← your PDF folder
ONTOLOGY_FILE = r"F:\ontology\arupmind_ontology.ttl"
OUTPUT_FILE   = r"F:\ontology\chunks.json"
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64

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

print(f"Loaded {len(keyword_map)} keywords from ontology")

# ─────────────────────────────────────────
# STEP A2 — Add short/common extra keywords
# These are single words too short for the ontology
# but very common in construction/fire/electrical PDFs
# ─────────────────────────────────────────
extra_keywords = {
    # Fire
    "fire":           "FireSafety",
    "smoke":          "SmokeControl",
    "flame":          "FireSafety",
    "combustion":     "FireSafety",
    "ignition":       "FireSafety",
    "flashover":      "FireSafety",
    "sprinkler":      "FireSuppression",
    "suppression":    "FireSuppression",
    "detector":       "FireDetection",
    "alarm":          "FireDetection",
    "evacuation":     "EgressDesign",
    "escape":         "EgressDesign",
    "egress":         "EgressDesign",
    "intumescent":    "FireResistantMaterials",
    "firewall":       "FireResistantMaterials",

    # Electrical
    "electrical":     "ElectricalSafety",
    "electric":       "ElectricalSafety",
    "voltage":        "PowerDistribution",
    "current":        "PowerDistribution",
    "wiring":         "CableManagement",
    "cable":          "CableManagement",
    "conduit":        "CableManagement",
    "circuit":        "CircuitProtection",
    "breaker":        "CircuitProtection",
    "grounding":      "GroundingAndBonding",
    "earthing":       "GroundingAndBonding",
    "electrocution":  "ElectricalHazards",
    "arc flash":      "ElectricalHazards",
    "shock":          "ElectricalHazards",
    "hazard":         "ElectricalHazards",

    # Structural
    "seismic":        "SeismicDesign",
    "earthquake":     "SeismicDesign",
    "vibration":      "SeismicDesign",
    "structural":     "StructuralEngineering",
    "structure":      "StructuralEngineering",
    "foundation":     "FoundationSystems",
    "footing":        "FoundationSystems",
    "soil":           "FoundationSystems",
    "load":           "LoadAnalysis",
    "stress":         "LoadAnalysis",
    "deflection":     "LoadAnalysis",
    "wind":           "WindDesign",
    "hurricane":      "WindDesign",
    "storm":          "WindDesign",
    "collapse":       "StructuralFailure",
    "failure":        "StructuralFailure",
    "fracture":       "StructuralFailure",

    # Construction
    "construction":   "Construction",
    "building":       "Construction",
    "concrete":       "ConcreteStructures",
    "reinforcement":  "ConcreteStructures",
    "rebar":          "ConcreteStructures",
    "steel":          "SteelStructures",
    "beam":           "SteelStructures",
    "column":         "SteelStructures",
    "material":       "BuildingMaterials",
    "insulation":     "BuildingMaterials",
    "safety":         "Construction_safety",
    "worker":         "Construction_safety",
    "accident":       "Construction_safety",
    "injury":         "Construction_safety",
    "ppe":            "Construction_safety",
}

keyword_map.update(extra_keywords)
print(f"Total keywords after manual additions: {len(keyword_map)}\n")

# ─────────────────────────────────────────
# STEP B — Improved Tag function
# Uses 3 matching strategies:
# 1. Exact full phrase match
# 2. All words in phrase present in text
# 3. Single keyword present in text
# ─────────────────────────────────────────
def tag_chunk(chunk_text):
    tags = []
    text_lower = chunk_text.lower()

    for keyword, concept in keyword_map.items():
        # Strategy 1: exact phrase match
        if keyword in text_lower:
            tags.append(concept)
            continue

        # Strategy 2: all meaningful words in phrase present
        keyword_words = [w for w in keyword.split() if len(w) > 4]
        if len(keyword_words) > 1:
            if all(w in text_lower for w in keyword_words):
                tags.append(concept)

    return list(set(tags))  # remove duplicates

# ─────────────────────────────────────────
# STEP C — Parse a single PDF (page by page)
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
# STEP D — Text splitter
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
    "SeismicDesign":        "Structural Engineering",
    "LoadAnalysis":         "Structural Engineering",
    "WindDesign":           "Structural Engineering",
    "StructuralFailure":    "Structural Engineering",
    "FoundationSystems":    "Structural Engineering",
    "StructuralEngineering":"Structural Engineering",

    "FireDetection":        "Fire Safety",
    "FireSuppression":      "Fire Safety",
    "FireResistantMaterials":"Fire Safety",
    "SmokeControl":         "Fire Safety",
    "EgressDesign":         "Fire Safety",
    "FireSafety":           "Fire Safety",

    "PowerDistribution":    "Electrical",
    "CircuitProtection":    "Electrical",
    "CableManagement":      "Electrical",
    "ElectricalHazards":    "Electrical",
    "ElectricalSafety":     "Electrical",
    "GroundingAndBonding":  "Electrical",
    "Electrical":           "Electrical",

    "BuildingMaterials":    "Construction",
    "ConcreteStructures":   "Construction",
    "SteelStructures":      "Construction",
    "Construction_safety":  "Construction",
    "Construction":         "Construction",
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
# STEP F — Main ingestion loop
# ─────────────────────────────────────────
all_chunks = []
chunk_id   = 0
today      = datetime.today().strftime("%Y-%m-%d")
skipped_pdfs = []

pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
print(f"Found {len(pdf_files)} PDF files to process\n")

for pdf_file in pdf_files:
    pdf_path = os.path.join(PDF_FOLDER, pdf_file)
    print(f"Processing: {pdf_file}")

    try:
        pages = parse_pdf(pdf_path)
    except Exception as e:
        print(f"  ❌ ERROR reading {pdf_file}: {e}")
        skipped_pdfs.append(pdf_file)
        continue

    if len(pages) == 0:
        print(f"  ⚠️  Skipped — 0 readable pages (likely scanned image PDF)")
        skipped_pdfs.append(pdf_file)
        continue

    for page in pages:
        chunks = splitter.split_text(page["text"])

        for chunk_text in chunks:
            if len(chunk_text.strip()) < 50:
                continue  # skip tiny fragments

            tags       = tag_chunk(chunk_text)
            discipline = infer_discipline(tags)

            chunk_doc = {
                "chunk_id":      chunk_id,
                "text":          chunk_text.strip(),
                "source_file":   pdf_file,
                "page":          page["page_num"],
                "ontology_tags": tags,
                "discipline":    discipline,
                "doc_type":      "research_paper",
                "date_ingested": today,
                "tag_count":     len(tags)
            }

            all_chunks.append(chunk_doc)
            chunk_id += 1

    print(f"  Pages: {len(pages)} | Chunks so far: {chunk_id}")

# ─────────────────────────────────────────
# STEP G — Save to JSON
# ─────────────────────────────────────────
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print(f"\n✅ DONE!")
print(f"   Total chunks created : {len(all_chunks)}")
print(f"   Output saved to      : {OUTPUT_FILE}")

# ─────────────────────────────────────────
# STEP H — Detailed stats
# ─────────────────────────────────────────
tagged   = sum(1 for c in all_chunks if c["tag_count"] > 0)
untagged = sum(1 for c in all_chunks if c["tag_count"] == 0)

discipline_counts = {}
for c in all_chunks:
    d = c["discipline"]
    discipline_counts[d] = discipline_counts.get(d, 0) + 1

print(f"\n=== Tagging Stats ===")
print(f"   Tagged chunks   : {tagged}  ({100*tagged//len(all_chunks)}%)")
print(f"   Untagged chunks : {untagged} ({100*untagged//len(all_chunks)}%)")

print(f"\n=== Chunks by Discipline ===")
for discipline, count in sorted(discipline_counts.items(), key=lambda x: -x[1]):
    bar = "█" * (count // 50)
    print(f"   {discipline:30} → {count:5} chunks  {bar}")

print(f"\n=== Top 10 Most Used Tags ===")
tag_counts = {}
for c in all_chunks:
    for tag in c["ontology_tags"]:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"   {tag:30} → {count} chunks")

if skipped_pdfs:
    print(f"\n=== Skipped PDFs ({len(skipped_pdfs)}) ===")
    for f in skipped_pdfs:
        print(f"   ⚠️  {f}")