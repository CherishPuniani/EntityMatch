import polars as pl, re, unicodedata
SP='/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/857adbd6-e53a-426a-b05f-5a47245c73c9/scratchpad/'
s=pl.read_parquet(SP+'rescue_sel_p05.parquet').filter(pl.col('p2n')<0.2)
base=pl.read_parquet(SP+'final9/rescue_sel_final9.parquet')
assert s.join(base,on='s1',how='semi').height==0 and s.join(base,on='t',how='semi').height==0
raw=pl.concat([pl.read_csv(f'dataset/test/test_source{k}.tsv',separator='\t',quote_char=None) for k in (1,2,3)]).filter(pl.col('entity_id').is_in(s['s1id'].to_list()+s['tid'].to_list()))
info={r[0]:(r[1],r[2]) for r in raw.iter_rows()}
def tok(x): x=unicodedata.normalize('NFKD',x or '').encode('ascii','ignore').decode().lower(); return set(w for w in re.split(r'[^a-z0-9]+',x) if w)
LF={'sarl','sas','sasu','eurl','sa','sci','ei','s','a','r','l','u','e','c','i'}
FILL={'developpement','france','groupe','associes','fils','freres','cie','et','services','and','partners','co','compagnie'}
def kind(a,b):
    A,B=tok(info[a][0]),tok(info[b][0])
    if not (A-LF)&(B-LF): return 'nooverlap'
    return 'filler' if not (B-A-LF-FILL) else 'other'
s=s.with_columns(kind=pl.struct('s1id','tid').map_elements(lambda r: kind(r['s1id'],r['tid']),return_dtype=pl.String))
print(s.group_by('kind').agg(pl.len(),pl.col('p2n').mean()))
f=s.filter(pl.col('kind')=='filler'); f.write_parquet(SP+'rescue_ext.parquet')
for a,b,p in f.sample(min(24,f.height),seed=11).select('s1id','tid','p2n').iter_rows():
    print(round(p,3),info[a][0][:34],'|',(info[a][1] or '')[:44],'\n      ',info[b][0][:34],'|',(info[b][1] or '')[:44])
