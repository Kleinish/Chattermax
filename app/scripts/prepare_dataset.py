#!/usr/bin/env python3
import csv, math, os, shutil
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm
TARGET_SR=int(os.getenv('SAMPLE_RATE','22050'))
TTS_MODEL=os.getenv('TTS_MODEL','turbo').lower()
if TTS_MODEL not in {'turbo','original'}: raise SystemExit(f"Invalid TTS_MODEL '{TTS_MODEL}'")
SRC_META=Path('dataset_raw')/TTS_MODEL/'metadata.csv'; SRC_DIR=Path('dataset_raw')/TTS_MODEL/'wavs'
DST_META=Path('dataset/metadata.csv'); DST_DIR=Path('dataset/wavs')
def resample(a,s,d):
    if s==d:return a
    g=math.gcd(s,d); return resample_poly(a,d//g,s//g)
def main():
    print(f'TTS model:       {TTS_MODEL}\nSource metadata: {SRC_META}\nSource audio:    {SRC_DIR}\nTarget rate:     {TARGET_SR} Hz')
    if not SRC_META.exists(): raise SystemExit(f'Missing source metadata: {SRC_META}')
    if not SRC_DIR.exists(): raise SystemExit(f'Missing source WAV directory: {SRC_DIR}')
    with SRC_META.open('r',encoding='utf-8',newline='') as f: rows=[r for r in csv.reader(f,delimiter='|') if len(r)>=2]
    print(f'Source metadata contains {len(rows)} utterances')
    if DST_DIR.exists(): shutil.rmtree(DST_DIR)
    DST_DIR.mkdir(parents=True,exist_ok=True); out=[]; missing=0
    for filename,text,*_ in tqdm(rows,desc='Preparing Piper audio'):
        src=SRC_DIR/filename
        if not src.exists(): missing+=1; continue
        audio,sr=sf.read(src,dtype='float32',always_2d=False)
        if audio.ndim>1: audio=audio.mean(axis=1)
        audio=resample(audio,sr,TARGET_SR); peak=float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak>0.98: audio=audio*(0.98/peak)
        sf.write(DST_DIR/filename,audio,TARGET_SR,subtype='PCM_16'); out.append((filename,text))
    DST_META.parent.mkdir(parents=True,exist_ok=True)
    with DST_META.open('w',encoding='utf-8',newline='') as f: csv.writer(f,delimiter='|',lineterminator='\n').writerows(out)
    print(f'\nDataset preparation complete\n  Source entries: {len(rows)}\n  Prepared:       {len(out)}\n  Missing WAVs:   {missing}\n  Metadata:       {DST_META}\n  Audio:          {DST_DIR}')
if __name__=='__main__': main()
