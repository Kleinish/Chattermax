#!/usr/bin/env python3
import collections, json, subprocess
from pathlib import Path
ROOT=Path('/workspace') if Path('/workspace').exists() else Path(__file__).resolve().parents[1]
corpus=ROOT/'corpus.txt'; reports=ROOT/'reports'; reports.mkdir(exist_ok=True)
phones=collections.Counter(); diphones=collections.Counter(); failures=0; phrase_count=0
for text in corpus.read_text(encoding='utf-8').splitlines():
    if not text.strip(): continue
    phrase_count += 1
    p=subprocess.run(['espeak-ng','-q','-x','--sep=|','-v','en-us',text],capture_output=True,text=True)
    if p.returncode: failures+=1; continue
    toks=[]
    for token in p.stdout.replace('\n','|').split('|'):
        token=token.strip()
        if token: toks.append(token)
    phones.update(toks); diphones.update(zip(toks,toks[1:]))
report={
 'phrases':phrase_count,
 'unique_phonemes':len(phones),
 'unique_diphones':len(diphones),
 'phonemize_failures':failures,
 'least_common_phonemes':phones.most_common()[-30:],
 'least_common_diphones':[(' + '.join(k),v) for k,v in diphones.most_common()[-50:]],
}
(reports/'phoneme_coverage.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
with (reports/'phoneme_counts.csv').open('w',encoding='utf-8') as f:
    f.write('phoneme,count\n')
    for k,v in phones.most_common(): f.write(f'"{k.replace(chr(34), chr(34)*2)}",{v}\n')
print(json.dumps({k:report[k] for k in ['phrases','unique_phonemes','unique_diphones','phonemize_failures']},indent=2))
print('Reports written to reports/')
