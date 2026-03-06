# geo_condition_extractor.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Iterable, Any, Set
import re
import json
import pandas as pd


# =========================
# 0) Config
# =========================

@dataclass
class ExtractConfig:
    window_size: int = 12  # ±N characters around root mention
    # sentence separators for tie-breaking & phrase assembly
    sent_seps: str = "。；;！!？?\n"
    # treat comma as weaker boundary (optional)
    weak_seps: str = "，,"
    # max span gap to attach stratigraphy prefix to lithology in phrase
    strat_to_lith_max_gap: int = 6
    # connector patterns for lithology phrase assembly
    connectors: Tuple[str, ...] = ("夹", "互层", "与", "及")
    # overlap policy: longest-match wins
    prefer_longest: bool = True


# =========================
# 1) Lexicons
# =========================

@dataclass(frozen=True)
class RootInfo:
    canonical: str
    category: str

class RootLexicon:
    """term -> (canonical, category)"""
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        required = {"term", "canonical", "category"}
        if not required.issubset(df.columns):
            raise ValueError(f"Root lexicon missing columns: {required - set(df.columns)}")

        self.term2info: Dict[str, RootInfo] = {}
        for _, r in df.iterrows():
            term = str(r["term"]).strip()
            canonical = str(r["canonical"]).strip()
            category = str(r["category"]).strip()
            if term:
                self.term2info[term] = RootInfo(canonical=canonical, category=category)

        self.terms: List[str] = list(self.term2info.keys())

    def get(self, term: str) -> RootInfo:
        return self.term2info[term]


class AttrLexicon:
    """term -> [category...]"""
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        required = {"term", "category"}
        if not required.issubset(df.columns):
            raise ValueError(f"Attr lexicon missing columns: {required - set(df.columns)}")

        self.term2cats: Dict[str, List[str]] = {}
        for _, r in df.iterrows():
            term = str(r["term"]).strip()
            cat = str(r["category"]).strip()
            if not term:
                continue
            self.term2cats.setdefault(term, [])
            if cat and cat not in self.term2cats[term]:
                self.term2cats[term].append(cat)

        self.terms: List[str] = list(self.term2cats.keys())

    def get_categories(self, term: str) -> List[str]:
        return self.term2cats.get(term, [])


def assert_no_overlap(root_lex: RootLexicon, attr_lex: AttrLexicon) -> None:
    root_terms = set(root_lex.terms)
    root_canons = set(info.canonical for info in root_lex.term2info.values())
    attr_terms = set(attr_lex.terms)
    overlap = attr_terms.intersection(root_terms.union(root_canons))
    if overlap:
        # In your prepared data this should be empty.
        raise ValueError(f"Root/Attr overlap found (should be 0): {sorted(list(overlap))[:50]}")


# =========================
# 2) Aho–Corasick matcher (pure Python)
# =========================

class ACAutomaton:
    class Node:
        __slots__ = ("next", "fail", "out")
        def __init__(self):
            self.next: Dict[str, int] = {}
            self.fail: int = 0
            self.out: List[str] = []

    def __init__(self, patterns: Iterable[str]):
        self.nodes: List[ACAutomaton.Node] = [ACAutomaton.Node()]
        for p in patterns:
            self._add_pattern(p)
        self._build_fail()

    def _add_pattern(self, p: str) -> None:
        cur = 0
        for ch in p:
            nxt = self.nodes[cur].next.get(ch)
            if nxt is None:
                self.nodes.append(ACAutomaton.Node())
                nxt = len(self.nodes) - 1
                self.nodes[cur].next[ch] = nxt
            cur = nxt
        self.nodes[cur].out.append(p)

    def _build_fail(self) -> None:
        from collections import deque
        q = deque()

        # init depth-1 fail links
        for ch, nxt in self.nodes[0].next.items():
            self.nodes[nxt].fail = 0
            q.append(nxt)

        # BFS
        while q:
            r = q.popleft()
            for ch, u in self.nodes[r].next.items():
                q.append(u)
                f = self.nodes[r].fail
                while f and ch not in self.nodes[f].next:
                    f = self.nodes[f].fail
                self.nodes[u].fail = self.nodes[f].next.get(ch, 0)
                self.nodes[u].out.extend(self.nodes[self.nodes[u].fail].out)

    def find_all(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Return list of (start, end_exclusive, matched_pattern)
        """
        res: List[Tuple[int, int, str]] = []
        state = 0
        for i, ch in enumerate(text):
            while state and ch not in self.nodes[state].next:
                state = self.nodes[state].fail
            state = self.nodes[state].next.get(ch, 0)
            if self.nodes[state].out:
                for pat in self.nodes[state].out:
                    start = i - len(pat) + 1
                    res.append((start, i + 1, pat))
        return res


# =========================
# 3) Data structures
# =========================

@dataclass
class Span:
    start: int
    end: int  # exclusive
    term: str

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2.0


@dataclass
class RootSpan(Span):
    canonical: str = ""
    category: str = ""

@dataclass
class AttrSpan(Span):
    categories: List[str] = None
    assigned_root_idx: Optional[int] = None


# =========================
# 4) Preprocess utilities
# =========================

_space_re = re.compile(r"\s+")

def normalize_text(text: str) -> str:
    # Keep Chinese punctuation; just remove excessive whitespace
    return _space_re.sub("", text.strip())


def sentence_id_map(text: str, cfg: ExtractConfig) -> List[int]:
    """
    Map each character position to a sentence id for tie-breaking.
    """
    sid = 0
    ids = [0] * len(text)
    for i, ch in enumerate(text):
        ids[i] = sid
        if ch in cfg.sent_seps:
            sid += 1
    return ids


# =========================
# 5) Overlap resolution (longest-match wins)
# =========================

def resolve_overlaps(spans: List[Tuple[int, int, str]], prefer_longest: bool = True) -> List[Tuple[int, int, str]]:
    """
    Greedy: sort by (start asc, length desc) and keep if no overlap with kept.
    This is stable and works well for dictionary phrases (per-block match count is small).
    """
    items = list(spans)
    if prefer_longest:
        items.sort(key=lambda x: (x[0], -(x[1] - x[0]), x[1], x[2]))
    else:
        items.sort(key=lambda x: (x[0], x[1], x[2]))

    kept: List[Tuple[int, int, str]] = []
    for s, e, t in items:
        overlapped = False
        for ks, ke, _ in kept:
            if not (e <= ks or s >= ke):
                overlapped = True
                break
        if not overlapped:
            kept.append((s, e, t))
    kept.sort(key=lambda x: (x[0], x[1]))
    return kept


# =========================
# 6) Core extraction
# =========================

class GeoConditionExtractor:
    def __init__(self, root_lex: RootLexicon, attr_lex: AttrLexicon, cfg: ExtractConfig):
        self.root_lex = root_lex
        self.attr_lex = attr_lex
        self.cfg = cfg

        # optional: ensure no exact overlaps
        assert_no_overlap(root_lex, attr_lex)

        self.root_matcher = ACAutomaton(root_lex.terms)
        self.attr_matcher = ACAutomaton(attr_lex.terms)

    def extract(self, block_text: str, block_id: str = "") -> Dict[str, Any]:
        raw = block_text
        text = normalize_text(block_text)
        if not text:
            return {"block_id": block_id, "text": raw, "roots": [], "block_level_attributes": [], "rendered": []}

        sid_map = sentence_id_map(text, self.cfg)

        # --- Step 1: match roots
        root_hits = self.root_matcher.find_all(text)
        root_hits = resolve_overlaps(root_hits, prefer_longest=self.cfg.prefer_longest)

        roots: List[RootSpan] = []
        for s, e, term in root_hits:
            info = self.root_lex.get(term)
            roots.append(RootSpan(start=s, end=e, term=term, canonical=info.canonical, category=info.category))

        # --- Step 2: match attributes globally (once)
        attr_hits = self.attr_matcher.find_all(text)
        attr_hits = resolve_overlaps(attr_hits, prefer_longest=self.cfg.prefer_longest)

        attrs: List[AttrSpan] = []
        for s, e, term in attr_hits:
            cats = self.attr_lex.get_categories(term)
            attrs.append(AttrSpan(start=s, end=e, term=term, categories=cats, assigned_root_idx=None))

        # --- Step 3: assign attributes to nearest root if within window
        block_level_attrs: List[AttrSpan] = []
        for a in attrs:
            candidates: List[Tuple[float, int]] = []
            for idx, r in enumerate(roots):
                w_s = max(0, r.start - self.cfg.window_size)
                w_e = min(len(text), r.end + self.cfg.window_size)
                if a.center >= w_s and a.center <= w_e:
                    dist = abs(a.center - r.center)
                    candidates.append((dist, idx))

            if not candidates:
                block_level_attrs.append(a)
                continue

            candidates.sort(key=lambda x: x[0])
            best_dist, best_idx = candidates[0]

            # tie-break: same sentence
            best_sid = sid_map[int(min(max(0, a.center), len(text)-1))]
            # if multiple with same dist, pick same sentence with attribute
            tied = [c for c in candidates if abs(c[0] - best_dist) < 1e-6]
            if len(tied) > 1:
                same_sent = []
                for _, idx in tied:
                    r = roots[idx]
                    rid = sid_map[r.start] if r.start < len(text) else sid_map[-1]
                    if rid == best_sid:
                        same_sent.append(idx)
                if same_sent:
                    best_idx = min(same_sent, key=lambda i: roots[i].start)

            a.assigned_root_idx = best_idx

        # aggregate attributes per root
        root_attrs: List[Dict[str, List[str]]] = [dict() for _ in roots]
        for a in attrs:
            if a.assigned_root_idx is None:
                continue
            bag = root_attrs[a.assigned_root_idx]
            # if term maps to multiple categories, store under each
            if not a.categories:
                bag.setdefault("unknown", [])
                if a.term not in bag["unknown"]:
                    bag["unknown"].append(a.term)
            else:
                for c in a.categories:
                    bag.setdefault(c, [])
                    if a.term not in bag[c]:
                        bag[c].append(a.term)

        # --- Optional: assemble lithology phrase (stratigraphy + lithology + connector + lithology)
        phrases = self._assemble_phrases(text, roots, sid_map)

        # --- Render (paper-friendly list)
        rendered = self._render(text, roots, root_attrs, block_level_attrs, phrases)

        # --- Output
        out_roots = []
        for i, r in enumerate(roots):
            out_roots.append({
                "canonical": r.canonical,
                "category": r.category,
                "surface": r.term,
                "span": [r.start, r.end],
                "attributes": root_attrs[i],
            })

        return {
            "block_id": block_id,
            "text": raw,
            "text_norm": text,
            "phrases": phrases,
            "roots": out_roots,
            "block_level_attributes": [asdict(x) for x in block_level_attrs],
            "rendered": rendered,
        }

    def _assemble_phrases(self, text: str, roots: List[RootSpan], sid_map: List[int]) -> List[str]:
        """
        Heuristic phrase: [stratigraphy] + lith1 + connector + lith2
        """
        if not roots:
            return []

        # group roots by sentence id
        sent2roots: Dict[int, List[RootSpan]] = {}
        for r in roots:
            sid = sid_map[r.start] if r.start < len(sid_map) else sid_map[-1]
            sent2roots.setdefault(sid, []).append(r)

        phrases: List[str] = []
        for sid, rs in sent2roots.items():
            rs_sorted = sorted(rs, key=lambda x: x.start)

            # find stratigraphy prefix (closest before first lithology)
            strat = None
            for r in rs_sorted:
                if r.category == "地层时代":
                    strat = r
                    break

            # identify lithology roots in this sentence
            lith = [r for r in rs_sorted if r.category == "岩性"]
            if len(lith) < 2:
                continue

            # pick first adjacent pair that has a connector between them
            for i in range(len(lith)-1):
                a, b = lith[i], lith[i+1]
                between = text[a.end:b.start]
                conn = None
                for c in self.cfg.connectors:
                    if c in between:
                        conn = c
                        break
                if not conn:
                    continue

                # prefix: stratigraphy near the first lithology
                prefix = ""
                if strat and strat.end <= a.start and (a.start - strat.end) <= self.cfg.strat_to_lith_max_gap:
                    prefix = strat.term

                phrase = f"{prefix}{a.term}{conn}{b.term}"
                if phrase not in phrases:
                    phrases.append(phrase)
                break

        return phrases

    def _render(
        self,
        text: str,
        roots: List[RootSpan],
        root_attrs: List[Dict[str, List[str]]],
        block_level_attrs: List[AttrSpan],
        phrases: List[str],
    ) -> List[str]:
        """
        Render in a stable order close to your example:
        phrase -> weathering -> strength -> discontinuity -> integrity -> bonding -> grade -> rest
        """
        out: List[str] = []

        # 1) phrases
        for p in phrases:
            out.append(p)

        # helper: collect attributes by preferred order
        preferred_attr_order = [
            "weathering",
            "strength_report", "strength_modifier",
            "discontinuity_development", "discontinuity_development_degree",
            "integrity", "integrity_degree",
            "discontinuity_bonding",
            "surrounding_rock_grade",
            "field_identification",
            "heterogeneity",
            "phenomena",
            "unknown",
        ]

        def append_attrs_from_root(i: int):
            bag = root_attrs[i]
            for k in preferred_attr_order:
                if k in bag:
                    for v in bag[k]:
                        if v not in out:
                            out.append(v)
            # append any other categories not in preferred list
            for k, vs in bag.items():
                if k in preferred_attr_order:
                    continue
                for v in vs:
                    if v not in out:
                        out.append(v)

        # 2) For each root (in appearance order), append its attributes
        #    Optionally also output some roots that are not already covered by phrase.
        for i, r in enumerate(sorted(list(enumerate(roots)), key=lambda x: x[1].start)):
            idx, rr = r
            # keep the surface root term in readable list (optional)
            # if you don't want to output bare roots, you can comment this out.
            if rr.term not in out and rr.category in ("岩性", "地层时代", "水文地质", "构造与结构面", "岩溶地貌", "土与第四纪堆积", "不良地质体"):
                out.append(rr.term)
            append_attrs_from_root(idx)

        # 3) block-level attrs (unassigned)
        for a in block_level_attrs:
            if a.term not in out:
                out.append(a.term)

        return out


# =========================
# 7) Example usage
# =========================

def demo():
    root_csv = "./root_match_lexicon_zh_v3_4454_no_attr_overlap.csv"
    attr_csv = "./attribute_lexicon_step2_zh_v2_556_no_root_overlap.csv"

    root_lex = RootLexicon(root_csv)
    attr_lex = AttrLexicon(attr_csv)
    cfg = ExtractConfig(window_size=12)

    extractor = GeoConditionExtractor(root_lex, attr_lex, cfg)

    text = "llllllllTBMDK12+12444k可以石炭二叠系板岩夹变质砂岩，弱风化，岩质硬，节理裂隙发育，岩体破碎，结构面结合程度一般，围岩级别Ⅳ级，局部见灰岩。加油梦萌感"
    res = extractor.extract(text, block_id="demo-1")

    print("Rendered:")
    print("、".join(res["rendered"]))
    print("\nStructured JSON:")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    demo()
