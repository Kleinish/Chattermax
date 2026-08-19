#!/usr/bin/env python3
import subprocess,sys
steps=[
 ['python3','scripts/download_corpus_sources.py'],
 ['python3','scripts/generate_specialized_corpus.py'],
 ['python3','scripts/build_corpus.py','--shuffle'],
 ['python3','scripts/analyze_phonemes.py'],
]
for s in steps:
    print('\n>>>',' '.join(s),flush=True); subprocess.run(s,check=True)
print('\nCorpus preparation complete. Review corpus.txt and reports/phoneme_coverage.json.')
