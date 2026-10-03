import polars as pl, sys
SP='/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/857adbd6-e53a-426a-b05f-5a47245c73c9/scratchpad/'
OP='/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad/'
sel=pl.read_parquet(SP+'rescue_sel.parquet')
assert sel['s1id'].null_count()==0 and sel['tid'].null_count()==0 and sel['s1id'].n_unique()==sel.height and sel['tid'].n_unique()==sel.height
R=dict(zip(sel['s1id'].to_list(),sel['tid'].to_list()))
out=open(SP+'final9/output_final9/matching_results.tsv','w'); n=0
for i,line in enumerate(open('output_final8/matching_results.tsv')):
    if i==0: out.write(line); continue
    a,_,b=line.rstrip('\n').partition('\t')
    if a in R:
        assert b.strip()=='', (a,b); line=f'{a}\t{R[a]}\n'; n+=1
    out.write(line)
out.close(); print('rescued rows',n)
# candidate file: ensure every rescued pair is present
cout=open(SP+'final9/output_final9/candidate_pairs.tsv','w'); miss=0
for i,line in enumerate(open(OP+'final8/output_final8/candidate_pairs.tsv')):
    if i>0:
        a,_,b=line.rstrip('\n').partition('\t')
        if a in R and R[a] not in b.split(','):
            miss+=1; line=f'{a}\t{R[a]},{b}\n' if b else f'{a}\t{R[a]}\n'
    cout.write(line)
cout.close(); print('candidate lines patched',miss)
