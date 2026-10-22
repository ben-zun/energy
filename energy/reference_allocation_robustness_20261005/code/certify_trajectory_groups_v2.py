"""Task and geometry certification in fixed order; no model predictions read."""
import argparse,csv,gzip,hashlib,json
from pathlib import Path
from collections import defaultdict,Counter
from pymatgen.core import Structure
from pymatgen.core.structure_matcher import StructureMatcher,ElementComparator
from check_mp_trajectory_source import MATCHER

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--snapshots',type=Path,action='append',required=True);p.add_argument('--history',type=Path,action='append',required=True);p.add_argument('--task-registry',type=Path,action='append',required=True);p.add_argument('--history-certificate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 certificate=json.loads(a.history_certificate.read_text())
 if certificate.get('history_scope_complete') is not True:raise ValueError('Complete historical audit certificate required')
 protocol=json.loads(a.protocol.read_text())
 if protocol['manifests_sha256'].get(a.manifest.name)!=digest(a.manifest):raise ValueError('Frozen prefix manifest hash mismatch')
 order=list(csv.DictReader(a.manifest.open()));rank={r['canonical_task_id']:i+1 for i,r in enumerate(order)}
 if len(rank)!=len(order):raise ValueError('Duplicate fixed prefix task')
 hashes=[hashlib.sha256(r['canonical_task_id'].encode()).hexdigest() for r in order]
 if hashes!=sorted(hashes):raise ValueError('Fixed task SHA256 ordering changed')
 task_exposure=set()
 for file in a.task_registry:
  if file.name!='TASK_REGISTRY_A.json':raise ValueError('Only explicit task registry A allowed')
  task_exposure.update(json.loads(file.read_text()))
 groups=defaultdict(list);seen=set()
 for file in a.snapshots:
  for r in map(json.loads,gzip.open(file,'rt')):
   if r['canonical_task_id'] not in rank or r['snapshot_id'] in seen:raise ValueError('Unexpected/duplicate candidate snapshot')
   seen.add(r['snapshot_id']);groups[r['canonical_task_id']].append(r)
 pairable={k:v for k,v in groups.items() if len(v)>=2 and k not in task_exposure};keys={v[0]['composition_key'] for v in pairable.values()}
 history=defaultdict(list);hist_hashes=set();source_counts=Counter()
 for file in a.history:
  for row in map(json.loads,gzip.open(file,'rt')):
   if row['composition_key'] not in keys:continue
   raw=json.dumps(row['structure'],sort_keys=True,separators=(',',':'));h=hashlib.sha256(raw.encode()).hexdigest()
   if h in hist_hashes:continue
   hist_hashes.add(h);s=Structure.from_dict(row['structure']);history[row['composition_key']].append((row['source'],s));source_counts[row['source']]+=1
 matcher=StructureMatcher(**MATCHER,comparator=ElementComparator());eligible=[];exclusions=[];task_records=[]
 for identifier in sorted(groups,key=lambda k:rank[k]):
  rows=sorted(groups[identifier],key=lambda r:r['ionic_step_index'])
  if len({r['composition_key'] for r in rows})!=1 or len({r['run_type'] for r in rows})!=1:raise ValueError('Task composition/run type changed')
  record={'task_id':identifier,'fixed_prefix_rank':rank[identifier],'representatives':len(rows),'history_task_hit':identifier in task_exposure}
  kept=[]
  if identifier not in task_exposure and len(rows)>=2:
   for row in rows:
    s=Structure.from_dict(row['structure']);old=sorted(history.get(row['composition_key'],[]),key=lambda x:abs(s.volume-x[1].volume))
    hit=next((source for source,geom in old if matcher.fit(s,geom)),None)
    if hit:exclusions.append({'snapshot_id':row['snapshot_id'],'task_id':identifier,'source':hit})
    else:kept.append(row)
  record['after_history_snapshots']=len(kept);record['certified_eligible']=len(kept)>=2;task_records.append(record)
  if len(kept)>=2:eligible.append((identifier,kept))
  if len(rows)>=2:print(json.dumps({'rank':rank[identifier],'task_id':identifier,'retained_after_history':len(kept),'certified_groups':len(eligible)}),flush=True)
  if len(eligible)==50:break
 a.output.mkdir(parents=True)
 chosen=[r for _,rows in eligible for r in rows]
 with gzip.open(a.output/'CERTIFIED_SNAPSHOTS_PRELOCK.jsonl.gz','wt') as f:
  for row in chosen:f.write(json.dumps(row,allow_nan=False)+'\n')
 with (a.output/'CERTIFIED_TASKS_PRELOCK.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['canonical_task_id','fixed_prefix_rank','snapshots']);w.writeheader();w.writerows({'canonical_task_id':i,'fixed_prefix_rank':rank[i],'snapshots':len(rows)} for i,rows in eligible)
 for name,records,fields in [('GROUP_CERTIFICATION.csv',task_records,['task_id','fixed_prefix_rank','representatives','history_task_hit','after_history_snapshots','certified_eligible']),('GEOMETRY_EXCLUSIONS.csv',exclusions,['snapshot_id','task_id','source'])]:
  with (a.output/name).open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
 result={'history_scope_complete':True,'certified_task_groups':len(eligible),'snapshots':len(chosen),'pairs':sum(len(v)*(len(v)-1)//2 for _,v in eligible),'fixed_candidate_prefix':len(order),'stopping_rank':rank[eligible[-1][0]] if len(eligible)==50 else len(order),'stopping_target_reached':len(eligible)==50,'actual_composition_keys':len({r['composition_key'] for r in chosen}),'task_registry_ids':len(task_exposure),'history_geometry_bucket_counts':dict(source_counts),'geometry_excluded_snapshots':len(exclusions),'model_forwards':0,'certified_snapshots_sha256':digest(a.output/'CERTIFIED_SNAPSHOTS_PRELOCK.jsonl.gz'),'input_sha256':{str(q):digest(q) for q in [a.protocol,a.manifest,*a.snapshots,*a.history,*a.task_registry,a.history_certificate]}}
 (a.output/'CERTIFICATION_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in {'history_geometry_bucket_counts','input_sha256'}}),flush=True)
if __name__=='__main__':main()
