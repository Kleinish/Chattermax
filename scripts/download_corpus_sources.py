#!/usr/bin/env python3
import csv, io, re, subprocess, sys, urllib.request
from pathlib import Path

ROOT=Path('/workspace') if Path('/workspace').exists() else Path(__file__).resolve().parents[1]
OUT=ROOT/'corpus_sources'; CACHE=ROOT/'corpus_cache'; OUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
UA={'User-Agent':'piper-corpus-builder/1.0'}

def get(url):
    req=urllib.request.Request(url,headers=UA)
    with urllib.request.urlopen(req,timeout=90) as r: return r.read()

def write_lines(name, lines):
    vals=[]
    for x in lines:
        x=re.sub(r'\s+',' ',x.strip())
        if x: vals.append(x)
    (OUT/name).write_text('\n'.join(vals)+'\n',encoding='utf-8')
    print(f'{name}: {len(vals):,} phrases')

def ljspeech():
    urls=[
      'https://huggingface.co/datasets/flexthink/ljspeech/resolve/main/metadata.csv',
      'https://huggingface.co/datasets/keithito/lj_speech/resolve/main/metadata.csv',
    ]
    data=None
    for u in urls:
        try: data=get(u); break
        except Exception as e: print('LJSpeech mirror failed:',e)
    if not data: raise RuntimeError('Could not download LJSpeech metadata')
    lines=[]
    for row in csv.reader(io.StringIO(data.decode('utf-8')),delimiter='|'):
        if len(row)>=3: lines.append(row[2])
        elif len(row)>=2: lines.append(row[1])
    write_lines('ljspeech.txt',lines)

def harvard():
    u='https://raw.githubusercontent.com/tidyverse/stringr/main/data-raw/harvard-sentences.txt'
    write_lines('harvard.txt',get(u).decode('utf-8').splitlines())

def arctic():
    urls=[
      'https://raw.githubusercontent.com/rhasspy/dataset-voice-kathleen/master/cmuarctic.data',
      'https://raw.githubusercontent.com/psibre/arctic-prompts/master/src/main/resources/cmuarctic.data',
    ]
    data=None
    for u in urls:
        try: data=get(u).decode('utf-8'); break
        except Exception as e: print('ARCTIC mirror failed:',e)
    if not data: raise RuntimeError('Could not download ARCTIC prompts')
    lines=[]
    for line in data.splitlines():
        m=re.search(r'"(.*)"',line)
        if m: lines.append(m.group(1))
    write_lines('arctic.txt',lines)

def piper_prompts():
    repo=CACHE/'piper-recording-studio'
    if not repo.exists():
        subprocess.run(['git','clone','--depth','1','https://github.com/rhasspy/piper-recording-studio.git',str(repo)],check=True)
    lines=[]
    prompts=repo/'prompts'
    for p in prompts.rglob('*.txt'):
        low=str(p.parent).lower()
        if ('english' not in low) and ('en_' not in low) and ('en-' not in low) and not low.endswith('/en'):
            continue
        for raw in p.read_text(encoding='utf-8',errors='ignore').splitlines():
            raw=raw.strip()
            if not raw: continue
            if '\t' in raw: raw=raw.split('\t',1)[1]
            lines.append(raw)
    write_lines('piper.txt',lines)

def main():
    funcs=[('LJSpeech',ljspeech),('CMU ARCTIC',arctic),('Harvard',harvard),('Piper prompts',piper_prompts)]
    failed=[]
    for name,fn in funcs:
        print(f'\n== {name} ==')
        try: fn()
        except Exception as e:
            failed.append(name); print(f'WARNING: {name} failed: {e}',file=sys.stderr)
    if failed: print('\nOptional source failures:',', '.join(failed))
if __name__=='__main__': main()
