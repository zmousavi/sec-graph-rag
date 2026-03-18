"""
read_results.py
===============
Pretty-print a results JSON file from retrieve.py.

Usage:
  python read_results.py results/results_20260316_054413.json
"""

import json
import os
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "results/results_20260316_054413.json"
data = json.load(open(path))

txt_path = os.path.splitext(path)[0] + ".txt"
out = open(txt_path, "w")

def p(s=""):
    print(s)
    out.write(s + "\n")

for r in data:
    p(f"{'='*65}")
    p(f"Q: {r['question']}")
    p(f"Mode:     {r['mode']}")
    p(f"Tickers:  {r.get('tickers_detected', [])}")
    p(f"Clusters: {r.get('clusters_used', 'N/A')}")
    p(f"Latency:  {r.get('latency_breakdown', {})}")
    p()
    p(f"ANSWER:\n{r['answer']}")
    p()
    paths = r.get('supporting_paths', [])
    if paths:
        p("PATHS:")
        for path_item in paths:
            p(f"  [{path_item['score']}] {path_item['label']}")
            p(f"           {' -> '.join(path_item['edge_types'])} -> {path_item['node_ids'][-1]}")
    else:
        p("PATHS: none")
    p()

out.close()
print(f"Saved → {txt_path}")
