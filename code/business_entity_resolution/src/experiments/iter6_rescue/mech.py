import polars as pl, re, unicodedata, sys
from rapidfuzz import fuzz
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220)
SP='/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/857adbd6-e53a-426a-b05f-5a47245c73c9/scratchpad/'
x=pl.read_parquet(SP+'pool9.parquet').filter((pl.col('country')=='France')&(pl.col('acls')=='same')&(pl.col('p2n')>=0.05))
raw=pl.concat([pl.read_csv(f'dataset/test/test_source{s}.tsv',separator='\t',quote_char=None,schema_overrides={'business_address':pl.String,'business_name':pl.String}).filter(pl.col('country')=='France').select((pl.col('entity_id').str.slice(3).cast(pl.Int64)+s*10_000_000_000).alias('id'),'entity_id','business_name','business_address') for s in (1,2,3)])
info={r[0]:r[1:] for r in raw.iter_rows()}
def nrm(s): return unicodedata.normalize('NFKD',s or '').encode('ascii','ignore').decode().lower()
LFS={'sarl','sas','sasu','eurl','sa','sci','ei','snc','selarl','scop','scm','scp','sca'}
def lf(name):
    s=nrm(name); s=re.sub(r'(?<=\b[a-z])\.(?=[a-z]\b)','',s); s=s.replace('.','')
    return set(w for w in re.split(r'[^a-z0-9]+',s) if w in LFS)
def toks(name):
    s=nrm(name); s=re.sub(r'(?<=\b[a-z])\.(?=[a-z]\b)','',s); s=s.replace('.','')
    return set(w for w in re.split(r'[^a-z0-9]+',s) if w)
FILL={'developpement','france','groupe','associes','fils','freres','cie','et','services','and','partners','co','le','la','les','de','du','des'}
STOP={'rue','r','avenue','av','ave','boulevard','bd','bld','allee','all','impasse','imp','place','pl','chemin','ch','route','rte','cours','quai','de','du','des','la','le','les','l','d','et','bis','ter','no','n','cedex','parvis','square','sq','passage','voie','residence','res'}
def street_words(a):
    a=re.sub(r"[^a-z0-9 ,]"," ",nrm(a)); seg=[p for p in a.split(',') if re.search(r'\d',p)]; seg=seg[0] if seg else ''
    return [w for w in seg.split() if not w.isdigit() and w not in STOP and len(w)>=3 and not re.fullmatch(r'\d+\w*',w)]
def sok(s1,t):
    ws=street_words(info[s1][2]); tw=re.sub(r"[^a-z0-9 ]"," ",nrm(info[t][2])).split()
    return bool(ws) and bool(tw) and all(max(fuzz.ratio(w,v) for v in tw)>=75 for w in ws)
def kind(s1,t):
    a,b=info[s1][1],info[t][1]; la,lb=lf(a),lf(b)
    if la and lb and la!=lb: return 'lf_change'
    A,B=toks(a)-LFS,toks(b)-LFS
    if A==B: return 'lf_add' if (lb and not la) else 'same_name'
    if not A&B: return 'nooverlap'
    if not (B-A-FILL): return 'filler'
    return 'other'
if len(sys.argv)>1 and sys.argv[1]=='check722':
    r=pl.read_parquet(SP+'final9/rescue_final9.parquet'); r=r.with_columns(k=pl.struct('s1','t').map_elements(lambda z: kind(z['s1'],z['t']),return_dtype=pl.String)); r.filter(pl.col('k')!='lf_change').drop('k').write_parquet(SP+'rescue_sel.parquet'); print('kept',r.filter(pl.col('k')!='lf_change').height)
    print(r.with_columns(k=pl.struct('s1','t').map_elements(lambda z: kind(z['s1'],z['t']),return_dtype=pl.String)).group_by('k').agg(pl.len()))
    sys.exit()
x=x.with_columns(k=pl.struct('s1','t').map_elements(lambda z: kind(z['s1'],z['t']),return_dtype=pl.String),
                 sk=pl.struct('s1','t').map_elements(lambda z: sok(z['s1'],z['t']),return_dtype=pl.Boolean))
x.write_parquet(SP+'pool9_fr_same.parquet')
x=x.with_columns(b2=pl.col('p2n').cut([0.1,0.2,0.3,0.5]))
print(x.group_by('emp','sk','k').agg(n=pl.len(),p=pl.col('p2n').mean().round(3)).sort('emp','sk','n',descending=[False,False,True]))
print(x.filter(pl.col('sk')&~pl.col('emp')).group_by('k','b2').agg(n=pl.len()).pivot(on='b2',index='k',values='n'))
