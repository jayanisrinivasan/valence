#!/usr/bin/env python3
import json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'public'/'data.json'
REQUIRED={
 'sym':str,'name':str,'placement_rationale':str,
 'evidence_provenance':list,'sensitivity':list,
 'confidence_components':list,'confidence_score':(int,float),
 'confidence_tier':str,'trs_post':(int,float),'family':str
}

def main():
    rows=json.loads(DATA.read_text())
    errors=[]
    if not isinstance(rows,list): errors.append('data.json root must be an array')
    seen=set()
    for i,o in enumerate(rows):
        if not isinstance(o,dict): errors.append(f'row {i} is not an object'); continue
        sym=o.get('sym',f'row {i}')
        if sym in seen: errors.append(f'duplicate symbol: {sym}')
        seen.add(sym)
        for key,typ in REQUIRED.items():
            if key not in o: errors.append(f'{sym}: missing {key}')
            elif not isinstance(o[key],typ): errors.append(f'{sym}: {key} has wrong type')
        score=o.get('confidence_score')
        if isinstance(score,(int,float)) and not 0<=score<=1: errors.append(f'{sym}: confidence_score outside [0,1]')
        if not o.get('fblock') and (o.get('col') is None or o.get('period') is None):
            errors.append(f'{sym}: main-table occupation missing col/period')
    if errors:
        print('\n'.join(errors),file=sys.stderr); return 1
    print(f'OK: {len(rows)} occupations; all pre-computed fields present.')
    return 0
if __name__=='__main__': raise SystemExit(main())
