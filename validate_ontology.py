from rdflib import Graph, SKOS
from rdflib.namespace import RDFS

g = Graph()
g.parse(r"F:\ontology\arupmind_ontology.ttl", format="turtle")  # ← fixed path

print(f"Total triples loaded: {len(g)}\n")

print("=== All Concepts ===")
for s, p, o in g.triples((None, SKOS.prefLabel, None)):
    print(f"  {str(s).split('#')[-1]:30} → {o}")

print("\n=== All Synonyms ===")
for s, p, o in g.triples((None, SKOS.altLabel, None)):
    print(f"  {str(s).split('#')[-1]:30} → {o}")