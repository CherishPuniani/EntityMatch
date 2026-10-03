"""Canonical legal-form set of a business name (all legal-form vocabularies of norm.py, dots removed)."""
import polars as pl
CANON = {"inc": "inc", "incorporated": "inc", "lnc": "inc", "llc": "llc", "lc": "llc", "ltd": "ltd", "limited": "ltd",
         "pvt": "pvt", "private": "pvt", "corp": "corp", "corporation": "corp", "co": "co", "company": "co", "lp": "lp",
         "llp": "llp", "pllc": "pllc", "pc": "pc", "plc": "plc", "pa": "pa", "sarl": "sarl", "sas": "sas", "sasu": "sasu",
         "sa": "sa", "sci": "sci", "eurl": "eurl", "snc": "snc", "selarl": "selarl", "scop": "scop", "scm": "scm",
         "gie": "gie", "pty": "pty", "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv"}
def lf(col):
    toks = pl.col(col).str.to_lowercase().str.replace_all(r"\.", "").str.extract_all(r"[a-z]+")
    return toks.list.eval(pl.element().replace_strict(CANON, default=None)).list.drop_nulls().list.unique().list.sort().list.join(",")
