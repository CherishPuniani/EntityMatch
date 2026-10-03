"""France empty-S1 rescue: select (S1 with no link in final8, its best-scoring candidate T unclaimed by any S1 and T's best S1 is this S1,
same house number, every S1 street word fuzzy-present in T's address, p2n >= 0.2)."""
import polars as pl, re, unicodedata, sys
from rapidfuzz import fuzz
SP='/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/857adbd6-e53a-426a-b05f-5a47245c73c9/scratchpad/'
PMIN=float(sys.argv[1]) if len(sys.argv)>1 else 0.2
x=pl.read_parquet(SP+'emp_cls_test.parquet').filter((pl.col('country')=='France')&(pl.col('acls')=='same')&(pl.col('tbest')==1)&(pl.col('p2n')>=PMIN))
raw=pl.concat([pl.read_csv(f'dataset/test/test_source{s}.tsv',separator='\t',quote_char=None,schema_overrides={'business_address':pl.String,'business_name':pl.String}).filter(pl.col('country')=='France').select((pl.col('entity_id').str.slice(3).cast(pl.Int64)+s*10_000_000_000).alias('id'),'entity_id','business_name','business_address') for s in (1,2,3)])
info={r[0]:r[1:] for r in raw.iter_rows()}
STOP={'rue','r','avenue','av','ave','boulevard','bd','bld','allee','all','impasse','imp','place','pl','chemin','ch','route','rte','cours','quai','de','du','des','la','le','les','l','d','et','bis','ter','no','n','nº','n°','cedex','parvis','square','sq','passage','voie','residence','res'}
def norm(s):
    s=unicodedata.normalize('NFKD',s or '').encode('ascii','ignore').decode().lower()
    return re.sub(r"[^a-z0-9 ,]"," ",s)
def street_words(a):
    a=norm(a)
    seg=[p for p in a.split(',') if re.search(r'\d',p)]
    seg=seg[0] if seg else ''
    return [w for w in seg.split() if not w.isdigit() and w not in STOP and len(w)>=3 and not re.fullmatch(r'\d+\w*',w)]
def ok(s1,t):
    ws=street_words(info[s1][2]); tw=norm(info[t][2]).replace(',',' ').split()
    if not ws or not tw: return False
    return all(max(fuzz.ratio(w,v) for v in tw)>=75 for w in ws)
x=x.with_columns(sok=pl.struct('s1','t').map_elements(lambda r: ok(r['s1'],r['t']),return_dtype=pl.Boolean))
print(x.group_by('sok').agg(pl.len(),pl.col('p2n').mean()))
sel=x.filter(pl.col('sok')).sort('p2n',descending=True).unique('t',keep='first').unique('s1',keep='first')
sel=sel.with_columns(s1id=pl.col('s1').replace_strict({k:v[0] for k,v in info.items()},default=None),tid=pl.col('t').replace_strict({k:v[0] for k,v in info.items()},default=None))
sel.select('s1','t','p2n','s1id','tid').write_parquet(SP+'rescue_sel.parquet')
print('selected',sel.height)
if len(sys.argv)>2:
    rej=x.filter(~pl.col('sok')).sample(min(10,x.filter(~pl.col('sok')).height),seed=1)
    for s1,t,p in rej.select('s1','t','p2n').iter_rows(): print('REJ',round(p,3),info[s1][1][:35],'|',info[s1][2][:55],'\n        ',info[t][1][:35],'|',info[t][2][:55])
    for s1,t,p in sel.sample(12,seed=5).select('s1','t','p2n').iter_rows(): print('SEL',round(p,3),info[s1][1][:35],'|',info[s1][2][:55],'\n        ',info[t][1][:35],'|',info[t][2][:55])
