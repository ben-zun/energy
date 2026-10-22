"""Select failed files from the same frozen history scope for parser repair."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--log',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
failed=[]
for line in a.log.open():
 r=json.loads(line)
 if r.get('status')=='failed':failed.append(r['path'])
source=json.loads(a.inventory.read_text());rows=[r for r in source['files'] if r['path'] in set(failed)]
assert len(rows)==len(set(failed))
a.output.write_text(json.dumps({'roots':source['roots'],'files':rows,'role':'reparse failures within unchanged frozen history scope'},indent=2)+'\n');print('Repair inputs:',len(rows))
