"""Normalization for business names / addresses (country-agnostic, with small
domain dictionaries for US / India / France abbreviations)."""
import re, unicodedata, json, os

_HERE = os.path.dirname(os.path.abspath(__file__))
_D = json.load(open(os.path.join(_HERE, "indic_dict.json"), encoding="utf-8"))
INDIC_NAME = _D["name"]
INDIC_ADDR = _D["addr"]

# ---------------------------------------------------------------- characters
_SPECIAL = str.maketrans({"ß": "ss", "æ": "ae", "œ": "oe", "ø": "o", "ł": "l", "đ": "d",
                          "ð": "d", "þ": "th", "ı": "i", "’": "", "‘": "", "`": "", "'": "",
                          "‌": "", "‍": ""})


def is_indic(s):
    return any(0x0900 <= ord(c) <= 0x0DFF for c in s)


# rule-based fallback transliteration for OOV Indic words: map every Brahmic
# block onto the Devanagari layout (same ISCII offsets) then to Latin.
_DEV = {}
_cons = "क ka ख kha ग ga घ gha ङ na च cha छ chha ज ja झ jha ञ na ट ta ठ tha ड da ढ dha ण na त ta थ tha द da ध dha न na ऩ na प pa फ pha ब ba भ bha म ma य ya र ra ऱ ra ल la ळ la ऴ la व va श sha ष sha स sa ह ha क़ qa ख़ kha ग़ ga ज़ za ड़ da ढ़ dha फ़ fa य़ ya".split()
for i in range(0, len(_cons), 2):
    _DEV[_cons[i]] = ("C", _cons[i + 1][:-1])
_vow = "अ a आ aa इ i ई ee उ u ऊ oo ऋ ri ए e ऐ ai ओ o औ au ऍ e ऑ o ऎ e ऒ o".split()
for i in range(0, len(_vow), 2):
    _DEV[_vow[i]] = ("V", _vow[i + 1])
_sig = "ा aa ि i ी ee ु u ू oo ृ ri े e ै ai ो o ौ au ॅ e ॉ o ॆ e ॊ o".split()
for i in range(0, len(_sig), 2):
    _DEV[_sig[i]] = ("M", _sig[i + 1])
_DEV["्"] = ("X", "")
_DEV["ं"] = ("A", "n")
_DEV["ँ"] = ("A", "n")
_DEV["ः"] = ("A", "h")
_DEV["़"] = ("A", "")
_DEV["ॐ"] = ("V", "om")
for d in range(10):
    _DEV[chr(0x0966 + d)] = ("D", str(d))


def translit_word(w):
    out = []
    pend = False  # consonant waiting for inherent vowel
    for ch in w:
        o = ord(ch)
        if 0x0980 <= o <= 0x0DFF:
            ch = chr(0x0900 + (o & 0x7F))
        t = _DEV.get(ch)
        if t is None:
            if pend:
                out.append("a"); pend = False
            if ch.isascii():
                out.append(ch)
            continue
        k, v = t
        if k == "C":
            if pend:
                out.append("a")
            out.append(v); pend = True
        elif k == "M":
            out.append(v); pend = False
        elif k == "X":
            pend = False
        else:
            if pend:
                out.append("a"); pend = False
            out.append(v)
    # final inherent vowel is usually silent (schwa deletion)
    return "".join(out)


def indic_name_to_latin(s):
    if not is_indic(s):
        return s, 0
    ws = []
    oov = 0
    for w in s.split():
        if is_indic(w):
            if w in INDIC_NAME:
                ws.append(INDIC_NAME[w])
            else:
                ws.append(translit_word(w)); oov += 1
        else:
            ws.append(w)
    return " ".join(ws), oov


def fold(s):
    s = s.translate(_SPECIAL)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower()


# ---------------------------------------------------------------- names
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b", "@": "a", "$": "s"})
LEGAL = set("""inc incorporated llc l l c ltd limited pvt private corp corporation co company cos lp llp pllc pc plc
pa lnc lc sarl sas sasu sa sci eurl snc selarl scop scm gie ets cie et fils sons pty gmbh ag bv nv
llp p c""".split())
HONOR = set("mr mrs ms dr smt sri shri shree m s the messrs".split())
NOISE = set("center centre services service partners com www id and dba formerly aka fka nee doing business as known "
            "f k a d b trading sys one labs group holdings".split())
ALIAS_RE = re.compile(r"\b(?:f/k/a|d/b/a|a/k/a|fka|dba|aka|formerly known as|formerly|doing business as|trading as|also known as|known as|nee|née|n/k/a)\b", re.I)


def name_forms(raw):
    """Return dict of name representations."""
    lat, oov = indic_name_to_latin(raw)
    f = fold(lat)
    f = f.replace("&", " and ").replace("+", " and ")
    # remove parenthetical ID noise like (ID: 13882)
    f = re.sub(r"\(?\bid\s*:\s*\d+\)?", " ", f)
    alias = None
    m = ALIAS_RE.search(f)
    if m:
        pre, post = f[:m.start()], f[m.end():]
        alias = (pre.strip(), post.strip())
    toks_raw = re.findall(r"[a-z0-9]+", f)
    # de-leet tokens mixing letters and digits (keep pure numbers)
    toks = []
    for t in toks_raw:
        if any(c.isdigit() for c in t) and any(c.isalpha() for c in t):
            t = t.translate(_LEET)
        toks.append(t)
    return f, toks, alias, oov


# ---------------------------------------------------------------- addresses
US_STATES = {"alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
             "connecticut": "ct", "delaware": "de", "district of columbia": "dc", "florida": "fl", "georgia": "ga",
             "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
             "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
             "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
             "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
             "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
             "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
             "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
             "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
             "puerto rico": "pr"}
IN_STATES = {"maharashtra": "mh", "delhi": "dl", "karnataka": "ka", "uttar pradesh": "up", "tamil nadu": "tn",
             "west bengal": "wb", "gujarat": "gj", "telangana": "tg", "haryana": "hr", "rajasthan": "rj",
             "kerala": "kl", "keralam": "kl", "bihar": "br", "madhya pradesh": "mp", "andhra pradesh": "ap",
             "punjab": "pb", "orissa": "od", "odisha": "od", "assam": "as", "jharkhand": "jh", "chhattisgarh": "cg",
             "uttarakhand": "uk", "goa": "ga", "himachal pradesh": "hp", "jammu and kashmir": "jk",
             "chandigarh": "ch", "puducherry": "py", "pondicherry": "py"}
ALL_STATES = {}
ALL_STATES.update(US_STATES)
ALL_STATES.update(IN_STATES)
STATE_CODES = set(US_STATES.values()) | set(IN_STATES.values())
ABBR = {"street": "st", "saint": "st", "str": "st", "road": "rd", "drive": "dr", "drife": "dr", "avenue": "ave",
        "av": "ave", "lane": "ln", "court": "ct", "circle": "cir", "place": "pl", "boulevard": "blvd",
        "trail": "trl", "parkway": "pkwy", "highway": "hwy", "terrace": "ter", "crossing": "xing",
        "mount": "mt", "fort": "ft", "north": "n", "south": "s", "east": "e", "west": "w", "cove": "cv",
        "square": "sq", "expressway": "expy", "freeway": "fwy", "point": "pt", "ridge": "rdg",
        "heights": "hts", "center": "ctr", "centre": "ctr", "junction": "jct", "route": "rte", "way": "wy",
        "loop": "lp", "run": "run", "pike": "pke", "turnpike": "tpke", "extension": "ext", "village": "vlg",
        "apartment": "apt", "suite": "ste", "floor": "fl", "building": "bldg", "room": "rm",
        # france
        "rue": "r", "allee": "all", "impasse": "imp", "chemin": "chem", "quai": "qu", "cours": "crs",
        "faubourg": "fg", "residence": "res", "bis": "bis",
        # india
        "nagar": "ngr", "near": "nr", "opposite": "opp", "behind": "bh", "sector": "sec", "colony": "col",
        "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6", "seventh": "7",
        "eighth": "8", "ninth": "9", "tenth": "10",
        # city aliases
        "bombay": "mumbai", "bengaluru": "bangalore", "calcutta": "kolkata", "poona": "pune",
        "madras": "chennai", "gurugram": "gurgaon", "ctiy": "city", "citty": "city", "cty": "city", "icty": "city",
        "ciy": "city"}
ADDR_DROP = set("null n a na unit cdp city township twp town of county no door h hn hno house flat plot block "
                "the region greater urban suburban divreportingcircle hq dist district apt ste fl bldg rm po pmb box".split())
NUM_RE = re.compile(r"\d+")
POBOX_RE = re.compile(r"\b(?:p\.?\s*o\.?\s*box|pmb)\s*#?\s*\d+", re.I)
ORD_RE = re.compile(r"^(\d+)(?:st|nd|rd|th)$")


def addr_forms(raw):
    a = raw
    if is_indic(a):
        parts = []
        for comp in a.split(","):
            c = comp.strip().strip('"')
            if is_indic(c):
                c = INDIC_ADDR.get(c, c)
            parts.append(c)
        a = ", ".join(parts)
    f = fold(a)
    f = POBOX_RE.sub(" ", f)
    comps = [c.strip() for c in f.split(",") if c.strip() and c.strip() not in ("null", "n/a", "na")]
    toks = []
    for c in comps:
        cc = c
        if cc in ALL_STATES:
            toks.append(ALL_STATES[cc]); continue
        for t in re.findall(r"[a-z0-9]+", cc):
            m = ORD_RE.match(t)
            if m:
                t = m.group(1)
            t = ABBR.get(t, t)
            if t.isdigit():
                t = t.lstrip("0") or "0"
            toks.append(t)
    return f, comps, toks
