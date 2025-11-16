from google.adk.agents import Agent

import arxiv
import json
import os
from typing import List, Dict, Any, Tuple, Set, Optional
import re
from datetime import datetime, timedelta
from google.adk.models.lite_llm import LiteLlm

PAPER_DIR="papers"

def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Search for papers on arXiv based on a topic and store their information.
    
    Args:
        topic: The topic to search for
        max_results: Maximum number of results to retrieve (default: 5)
        
    Returns:
        List of paper IDs found in the search
    """
    
    # Use arxiv to find the papers 
    client = arxiv.Client()

    # Search for the most relevant articles matching the queried topic
    search = arxiv.Search(
        query = topic,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.Relevance
    )

    papers = client.results(search)
    
    # Create directory for this topic
    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)
    
    file_path = os.path.join(path, "papers_info.json")

    # Try to load existing papers info
    try:
        with open(file_path, "r") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    # Process each paper and add to papers_info  
    paper_ids = []
    for paper in papers:
        paper_ids.append(paper.get_short_id())
        paper_info = {
            'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'summary': paper.summary,
            'pdf_url': paper.pdf_url,
            'published': str(paper.published.date())
        }
        papers_info[paper.get_short_id()] = paper_info
    
    # Save updated papers_info to json file
    with open(file_path, "w") as json_file:
        json.dump(papers_info, json_file, indent=2)
    
    print(f"Results are saved in: {file_path}")
    
    return paper_ids

def extract_info(paper_id: str) -> str:
    """
    Search for information about a specific paper across all topic directories.
    
    Args:
        paper_id: The ID of the paper to look for
        
    Returns:
        JSON string with paper information if found, error message if not found
    """
 
    for item in os.listdir(PAPER_DIR):
        item_path = os.path.join(PAPER_DIR, item)
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path, "papers_info.json")
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r") as json_file:
                        papers_info = json.load(json_file)
                        if paper_id in papers_info:
                            return json.dumps(papers_info[paper_id], indent=2)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue
    
    return f"There's no saved information related to paper {paper_id}."

def _ensure_topic_dir(topic: str) -> Tuple[str, str]:
    safe = topic.lower().replace(" ", "_")
    path = os.path.join(PAPER_DIR, safe)
    os.makedirs(path, exist_ok=True)
    return path, os.path.join(path, "papers_info.json")

def _load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def _save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def _paper_to_dict(p: arxiv.Result) -> Dict[str, Any]:
    return {
        "id": p.get_short_id(),
        "title": p.title,
        "authors": [a.name for a in p.authors],
        "summary": p.summary,
        "pdf_url": p.pdf_url,
        "primary_category": getattr(p, "primary_category", None),
        "categories": list(getattr(p, "categories", []) or []),
        "published": str(p.published.date()) if p.published else None,
        "updated": p.updated.isoformat() if getattr(p, "updated", None) else None,
    }

def find_papers_by_benchmark(benchmark: str, max_results: int = 50) -> List[str]:
    """
    按基准/细分领域关键词检索论文，保存信息，并返回按时间倒序的短 ID 列表。
    """
    client = arxiv.Client()
    # 提高命中率：在标题与摘要匹配
    query = f'ti:"{benchmark}" OR abs:"{benchmark}"'
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    results = list(client.results(search))
    # 最新在前
    results.sort(key=lambda r: r.published or datetime.min, reverse=True)

    path, file_path = _ensure_topic_dir(benchmark)
    store = _load_json(file_path)

    paper_ids: List[str] = []
    for p in results:
        d = _paper_to_dict(p)
        store[d["id"]] = d
        paper_ids.append(d["id"])

    _save_json(file_path, store)
    print(f"Saved {len(paper_ids)} papers to: {file_path}")
    return paper_ids

_METRIC_PATTERNS = [
    r"(?:accuracy|acc)\s*(?:of|=|:)?\s*(\d{1,2}(?:\.\d+)?|\d{3})(?:\s*%|\b)",
    r"(?:f1|f1-score)\s*(?:of|=|:)?\s*(\d{1,2}(?:\.\d+)?|\d{3})(?:\s*%|\b)",
    r"(?:bleu)\s*(?:of|=|:)?\s*(\d{1,2}(?:\.\d+)?)(?:\s*%|\b)",
    r"(?:rouge[-_]?(?:l|1|2)?)\s*(?:of|=|:)?\s*(\d{1,2}(?:\.\d+)?)(?:\s*%|\b)",
    r"(?:mmlu|mmmu|mmlu_score)\s*(?:of|=|:)?\s*(\d{1,2}(?:\.\d+)?|\d{3})(?:\s*%|\b)",
]
_SOTA_HINTS = ["state-of-the-art", "sota", "sets new state of the art", "new sota", "achiev", "surpass", "outperform"]

def _extract_metric(text: str) -> Tuple[str, float]:
    text_l = text.lower()
    for pat in _METRIC_PATTERNS:
        m = re.search(pat, text_l)
        if m:
            val = m.group(1)
            try:
                score = float(val)
                # 百分数统一到 0-100
                if score <= 1.0:
                    score *= 100.0
                return pat, score
            except ValueError:
                continue
    return "", -1.0

def _looks_like_sota(title: str, summary: str) -> bool:
    blob = (title + " " + summary).lower()
    return any(h in blob for h in _SOTA_HINTS)

_SCOPE_PATTERNS: Dict[str, List[str]] = {
    "self-supervised": [
        r"\bself[- ]?supervised\b", r"\bunsupervised pretraining\b", r"\bssl\b"
    ],
    "supervised": [
        r"\bsupervised\b", r"\bfully[- ]?supervised\b"
    ],
    "reinforcement": [
        r"\breinforcement learning\b", r"\brl\b"
    ],
    "semi-supervised": [
        r"\bsemi[- ]?supervised\b"
    ],
    "weakly-supervised": [
        r"\bweakly[- ]?supervised\b", r"\bweak supervision\b"
    ],
    "unsupervised": [
        r"\bunsupervised\b"
    ],
    "zero-shot": [
        r"\bzero[- ]?shot\b"
    ],
    "few-shot": [
        r"\bfew[- ]?shot\b", r"\bone[- ]?shot\b"
    ],
}

def _detect_scopes(title: str, summary: str) -> Set[str]:
    text = f"{title}\n{summary}".lower()
    found: Set[str] = set()
    for scope, pats in _SCOPE_PATTERNS.items():
        for pat in pats:
            if re.search(pat, text):
                found.add(scope)
                break
    return found

# ------------------- 通用 Constraints 解析（范式/数据/模态/tricks/资源） -------------------
_DATA_REGIME_PATTERNS: Dict[str, List[str]] = {
    "no-extra-data": [r"\bno extra data\b", r"\bwithout extra data\b", r"\btraining data\b.*\bonly\b", r"\bno external\b"],
    "extra-data": [r"\bextra data\b", r"\bexternal data\b", r"\badditional data\b", r"\bweb[- ]?scale\b"],
    "pretrained": [r"\bpre[- ]?train", r"\bpretrained\b", r"\bfoundation model\b"],
    "distilled": [r"\bdistill", r"\bteacher[- ]?student\b"],
}
_MODALITY_PATTERNS: Dict[str, List[str]] = {
    "rgb": [r"\brgb\b"],
    "rgbd": [r"\brgb[- ]?d\b"],
    "multimodal": [r"\bmultimodal\b", r"\bvision[- ]?language\b"],
    "infrared": [r"\bir\b", r"\binfrared\b", r"\bthermal\b"],
    "event": [r"\bevent camera\b", r"\bevent[- ]?based\b"],
}
_TRICKS_PATTERNS: Dict[str, List[str]] = {
    "tta": [r"\btest[- ]?time augmentation\b", r"\btta\b"],
    "ensemble": [r"\bensemble\b"],
    "prompting": [r"\bprompt[- ]?(tuning|engineering)\b"],
}
_RESOURCE_PATTERNS: Dict[str, List[str]] = {
    "realtime": [r"\breal[- ]?time\b", r"\b\d+\s*fps\b"],
    "lightweight": [r"\b(lightweight|tiny|small) model\b", r"\b<\s*\d+\s*(m|b)\s*params\b"],
}

def _extract_constraints(title: str, summary: str) -> Dict[str, Any]:
    text = f"{title}\n{summary}".lower()
    def match_dict(pats: Dict[str, List[str]]) -> List[str]:
        out: List[str] = []
        for k, lst in pats.items():
            if any(re.search(p, text) for p in lst):
                out.append(k)
        return sorted(list(set(out)))
    return {
        "scopes": sorted(list(_detect_scopes(title, summary))),
        "data_regime": match_dict(_DATA_REGIME_PATTERNS),
        "modality": match_dict(_MODALITY_PATTERNS),
        "tricks": match_dict(_TRICKS_PATTERNS),
        "resources": match_dict(_RESOURCE_PATTERNS),
    }

# ------------------- 数据集与指标抽取及方向配置 -------------------
_DATASET_PATTERNS: Dict[str, List[str]] = {
    # 跟踪常见数据集（示例）
    "LaSOT": [r"\blasot\b"],
    "GOT-10k": [r"\bgot[- ]?10k\b"],
    "OTB": [r"\botb(?:50|100)?\b"],
    "TrackingNet": [r"\btrackingnet\b"],
    "TNL2K": [r"\btnl2k\b"],
}
_DATASET_METRIC_CONFIG: Dict[str, Dict[str, Any]] = {
    # 每个数据集的主指标与方向（True=越大越好）
    "LaSOT": {"primary": ["success", "precision", "success_rate"], "larger_is_better": True},
    "GOT-10k": {"primary": ["ao", "success", "mAP"], "larger_is_better": True},
    "OTB": {"primary": ["success", "precision"], "larger_is_better": True},
    "TrackingNet": {"primary": ["success", "precision"], "larger_is_better": True},
    "TNL2K": {"primary": ["success", "precision"], "larger_is_better": True},
}

def _detect_datasets(text: str) -> List[str]:
    tl = text.lower()
    found: List[str] = []
    for name, pats in _DATASET_PATTERNS.items():
        if any(re.search(p, tl) for p in pats):
            found.append(name)
    return sorted(list(set(found)))

def _extract_metric_with_dataset(title: str, summary: str) -> Dict[str, Any]:
    text = f"{title}\n{summary}"
    pat, score = _extract_metric(text)
    datasets = _detect_datasets(text)
    return {"metric_pattern": pat, "metric_score": (score if score >= 0 else None), "datasets": datasets}

def _score_key_for_dataset(metric_score: Optional[float], datasets: List[str]) -> float:
    # 若能识别数据集，直接使用指标分数；否则稍微降权
    if metric_score is None:
        return -1.0
    return metric_score if datasets else (metric_score * 0.95)

def get_latest_sota(benchmark: str, window_days: int = 365, max_results: int = 100, scope: str = "overall", constraints: Optional[Dict[str, Any]] = None) -> str:
    """
    根据关键词检索近 window_days 天的论文，启发式识别 SOTA，返回包含最新 SOTA 的 JSON。
    scope 可选：overall（默认）、self-supervised、supervised、semi-supervised、weakly-supervised、unsupervised、zero-shot、few-shot
    constraints 可选：{data_regime:[], modality:[], tricks:[], resources:[], require_dataset:bool}
    """
    client = arxiv.Client()
    query = f'ti:"{benchmark}" OR abs:"{benchmark}"'
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    now = datetime.utcnow().date()
    candidates: List[Dict[str, Any]] = []
    constraints = constraints or {}
    # 兼容：若传入 scope 且非 overall，等价于 include_scopes=[scope]
    include_scopes: Set[str] = set(map(str, (constraints.get("include_scopes") or [])))
    if scope and scope != "overall":
        include_scopes.add(scope)
    exclude_scopes: Set[str] = set(map(str, (constraints.get("exclude_scopes") or [])))
    strict_scope: bool = bool(constraints.get("strict_scope", False))
    dataset_filter: Set[str] = set(map(str, (constraints.get("datasets") or [])))
    require_dataset = bool(constraints.get("require_dataset", False))

    for p in client.results(search):
        if not p.published:
            continue
        if (now - p.published.date()) > timedelta(days=window_days):
            continue
        d = _paper_to_dict(p)
        ctz = _extract_constraints(d["title"], d["summary"] or "")
        d.update(ctz)
        scopes = set(ctz.get("scopes") or [])
        # 作用域过滤（包含/排除/严格）
        if include_scopes:
            if strict_scope:
                # 严格模式：论文的 scopes 必须是 include_scopes 的子集，且非空
                if not scopes or not scopes.issubset(include_scopes):
                    continue
            else:
                # 宽松模式：与 include_scopes 有交集即可
                if not (scopes & include_scopes):
                    continue
        if exclude_scopes and (scopes & exclude_scopes):
            continue
        # 其它 constraints 过滤（若用户提供）
        def match_list(key: str) -> bool:
            want = set(constraints.get(key) or [])
            if not want:
                return True
            have = set(ctz.get(key) or [])
            return bool(want & have)  # 交集匹配
        if not all([match_list("data_regime"), match_list("modality"), match_list("tricks"), match_list("resources")]):
            continue
        # 必含/必排术语（标题+摘要）
        required_terms = list(map(str, (constraints.get("required_terms") or [])))
        forbidden_terms = list(map(str, (constraints.get("forbidden_terms") or [])))
        blob = f"{d['title']}\n{d.get('summary') or ''}".lower()
        if required_terms and not all(term.lower() in blob for term in required_terms):
            continue
        if forbidden_terms and any(term.lower() in blob for term in forbidden_terms):
            continue

        # 必含/必排术语（标题+摘要）
        required_terms = list(map(str, (constraints.get("required_terms") or [])))
        forbidden_terms = list(map(str, (constraints.get("forbidden_terms") or [])))
        blob = f"{d['title']}\n{d.get('summary') or ''}".lower()
        if required_terms and not all(term.lower() in blob for term in required_terms):
            continue
        if forbidden_terms and any(term.lower() in blob for term in forbidden_terms):
            continue
        met = _extract_metric_with_dataset(d["title"], d["summary"] or "")
        d.update(met)
        if require_dataset and not d["datasets"]:
            continue
        if dataset_filter:
            if not set(d["datasets"] or []) & dataset_filter:
                continue
        is_sota = _looks_like_sota(d["title"], d["summary"])
        # 排序信号：SOTA 线索、指标（结合是否识别出数据集）、显式匹配 scope、时间
        metric_key = _score_key_for_dataset(d.get("metric_score"), d.get("datasets") or [])
        rank_score = (
            1 if is_sota else 0,
            metric_key,
            1 if (scope != "overall" and scope in scopes) else 0,
            p.published,
        )
        d.update({
            "sota_signal": is_sota,
            "rank_score": [rank_score[0], rank_score[1], rank_score[2], str(rank_score[3])],
            "evidence": (d["summary"] or "")[:400],  # 证据片段（摘要前 400 字符）
        })
        candidates.append(d)

    if not candidates:
        return json.dumps({"benchmark": benchmark, "scope": scope, "constraints": constraints, "sota": None, "message": "未检索到近一年可能的 SOTA 论文"}, ensure_ascii=False, indent=2)

    # 排序：SOTA 线索优先，指标越大越好，时间越新越好
    candidates.sort(key=lambda x: (
        1 if x.get("sota_signal") else 0,
        (x.get("metric_score") or -1.0),
        (1 if (include_scopes and (set(x.get("scopes") or []) & include_scopes)) else 0),
        datetime.fromisoformat(x["updated"]) if x.get("updated") else datetime.min,
    ), reverse=True)

    best = candidates[0]
    result = {
        "benchmark": benchmark,
        "scope": scope,
        "constraints": constraints,
        "sota": {
            "id": best["id"],
            "title": best["title"],
            "published": best["published"],
            "pdf_url": best["pdf_url"],
            "metric": best.get("metric_score"),
            "metric_pattern": best.get("metric_pattern"),
            "sota_signal": best.get("sota_signal"),
            "scopes": best.get("scopes") or [],
            "datasets": best.get("datasets") or [],
            "evidence": best.get("evidence"),
        },
        "top_candidates": [
            {
                "id": c["id"],
                "title": c["title"],
                "published": c["published"],
                "metric": c.get("metric_score"),
                "sota_signal": c.get("sota_signal"),
                "scopes": c.get("scopes") or [],
                "datasets": c.get("datasets") or [],
                "pdf_url": c["pdf_url"],
                "evidence": (c.get("summary") or "")[:200],
            } for c in candidates[:5]
        ]
    }
    return json.dumps(result, ensure_ascii=False, indent=2)

def list_recent_papers(benchmark: str, limit: int = 10, window_days: int = 180, scope: str = "overall", constraints: Optional[Dict[str, Any]] = None) -> str:
    """
    返回近期（window_days）相关论文列表（按时间倒序），包含基本元信息、启发式指标、范围与约束标签。
    scope 可选：overall（默认）、self-supervised、supervised、semi-supervised、weakly-supervised、unsupervised、zero-shot、few-shot
    constraints 可选：同 get_latest_sota
    """
    client = arxiv.Client()
    query = f'ti:"{benchmark}" OR abs:"{benchmark}"'
    search = arxiv.Search(
        query=query,
        max_results=max(100, limit),
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    now = datetime.utcnow().date()
    items: List[Dict[str, Any]] = []
    constraints = constraints or {}
    include_scopes: Set[str] = set(map(str, (constraints.get("include_scopes") or [])))
    if scope and scope != "overall":
        include_scopes.add(scope)
    exclude_scopes: Set[str] = set(map(str, (constraints.get("exclude_scopes") or [])))
    strict_scope: bool = bool(constraints.get("strict_scope", False))
    dataset_filter: Set[str] = set(map(str, (constraints.get("datasets") or [])))
    require_dataset = bool(constraints.get("require_dataset", False))

    for p in client.results(search):
        if not p.published:
            continue
        if (now - p.published.date()) > timedelta(days=window_days):
            continue
        d = _paper_to_dict(p)
        ctz = _extract_constraints(d["title"], d["summary"] or "")
        d.update(ctz)
        scopes = set(ctz.get("scopes") or [])
        # 作用域过滤
        if include_scopes:
            if strict_scope:
                if not scopes or not scopes.issubset(include_scopes):
                    continue
            else:
                if not (scopes & include_scopes):
                    continue
        if exclude_scopes and (scopes & exclude_scopes):
            continue
        def match_list(key: str) -> bool:
            want = set(constraints.get(key) or [])
            if not want:
                return True
            have = set(ctz.get(key) or [])
            return bool(want & have)
        if not all([match_list("data_regime"), match_list("modality"), match_list("tricks"), match_list("resources")]):
            continue
        met = _extract_metric_with_dataset(d["title"], d["summary"] or "")
        d.update(met)
        if require_dataset and not d["datasets"]:
            continue
        if dataset_filter:
            if not set(d["datasets"] or []) & dataset_filter:
                continue
        d["sota_signal"] = _looks_like_sota(d["title"], d["summary"])
        d["evidence"] = (d["summary"] or "")[:200]
        items.append(d)

    items.sort(key=lambda d: (d.get("metric_score") or -1.0, d.get("published") or ""), reverse=True)
    return json.dumps(items[:limit], ensure_ascii=False, indent=2)

# ------------------- 自然语言解析 → benchmark / constraints / 时间窗口 -------------------
_CN_SCOPE_SYNONYMS = {
    "self-supervised": ["自监督", "自我监督"],
    "supervised": ["监督学习", "有监督", "纯监督", "完全监督", "只看监督"],
    "reinforcement": ["强化学习", "增强学习", "RL"],
    "semi-supervised": ["半监督"],
    "weakly-supervised": ["弱监督"],
    "unsupervised": ["无监督"],
    "zero-shot": ["零样本"],
    "few-shot": ["小样本", "少样本", "单样本", "一-shot", "one-shot"],
}

_CN_DATA_REGIME_SYNONYMS = {
    "no-extra-data": ["不使用额外数据", "无额外数据", "只用官方数据", "仅训练集"],
    "extra-data": ["使用额外数据", "外部数据", "额外数据", "网络规模", "web规模"],
    "pretrained": ["预训练", "预训练模型", "foundation"],
    "distilled": ["蒸馏", "distill"],
}

_CN_TRICKS_SYNONYMS = {
    "tta": ["测试时增强", "TTA"],
    "ensemble": ["集成", "ensemble"],
    "prompting": ["提示工程", "prompt", "prompting", "提示微调"],
}

_CN_RES_SYNONYMS = {
    "realtime": ["实时", "实时性", "fps"],
    "lightweight": ["轻量", "小模型", "tiny"],
}

_CN_TIME_PATTERNS = [
    (r"(近|最近)(\d{1,3})天", "days"),
    (r"(近|最近)(\d{1,2})个月", "months"),
    (r"(近|最近)(\d{1,2})年", "years"),
    (r"(过去|最近)(\d{1,3})\s*days?", "days"),
    (r"(过去|最近)(\d{1,2})\s*months?", "months"),
    (r"(过去|最近)(\d{1,2})\s*years?", "years"),
]

def _nl_detect_benchmark(text: str) -> str:
    for ds in _DATASET_PATTERNS.keys():
        if re.search(rf"\b{re.escape(ds)}\b", text, flags=re.I):
            return ds
    return text.strip()

def _nl_detect_scopes(text: str) -> Tuple[Set[str], Set[str], bool]:
    t = text.lower()
    include: Set[str] = set()
    exclude: Set[str] = set()
    strict = False
    for scope, syns in _CN_SCOPE_SYNONYMS.items():
        if any(s in text for s in syns) or scope in t:
            include.add(scope)
    if any(kw in text for kw in ["不要自监督", "排除自监督", "不含自监督", "exclude self"]):
        exclude.add("self-supervised")
    if any(kw in text for kw in ["不要强化学习", "排除强化学习", "不含强化学习", "exclude rl", "exclude reinforcement"]):
        exclude.add("reinforcement")
    if "只看监督" in text or "纯监督" in text or "strict supervised" in t:
        include.add("supervised")
        strict = True
    return include, exclude, strict

def _nl_detect_constraints(text: str) -> Dict[str, Any]:
    c: Dict[str, Any] = {"data_regime": [], "modality": [], "tricks": [], "resources": []}
    def match_syn(syn_map: Dict[str, List[str]], key: str):
        for k, syns in syn_map.items():
            if any(s in text for s in syns):
                c[key].append(k)
    match_syn(_CN_DATA_REGIME_SYNONYMS, "data_regime")
    match_syn(_CN_TRICKS_SYNONYMS, "tricks")
    match_syn(_CN_RES_SYNONYMS, "resources")
    # 通用“必排/必含”术语解析（泛化，不依赖预置列表）
    def _clean_terms(terms):
        out = []
        for t in terms:
            t = re.sub(r"^[，。,.；;、\s]+|[，。,.；;、\s]+$", "", t)
            t = t.strip()
            if 1 <= len(t) <= 32:
                out.append(t)
        return list(dict.fromkeys(out))
    forbidden = []
    # 中文：不要X / 不含X / 排除X
    for m in re.findall(r"(?:不要|不含|排除)([^\s，。,.；;、]{1,32})", text):
        forbidden.append(m)
    # 英文：exclude X / without X / no X
    for m in re.findall(r"(?:exclude|without|no)\s+([A-Za-z0-9\-\+_\/ ]{1,32})", text, flags=re.I):
        forbidden.append(m.strip())
    required = []
    # 中文：必须X / 只要X / 包含X
    for m in re.findall(r"(?:必须|只要|包含)([^\s，。,.；;、]{1,32})", text):
        required.append(m)
    # 英文：must contain X / require X / with X
    for m in re.findall(r"(?:must\s+contain|require|with)\s+([A-Za-z0-9\-\+_\/ ]{1,32})", text, flags=re.I):
        required.append(m.strip())
    forbidden = _clean_terms(forbidden)
    required = _clean_terms(required)
    if forbidden:
        c["forbidden_terms"] = forbidden
    if required:
        c["required_terms"] = required
    datasets = []
    for ds in _DATASET_PATTERNS.keys():
        if re.search(rf"\b{re.escape(ds)}\b", text, flags=re.I):
            datasets.append(ds)
    if datasets:
        c["datasets"] = sorted(list(set(datasets)))
        c["require_dataset"] = True
    if any(k in text for k in ["RGB", "rgb", "只用RGB", "纯RGB"]):
        c["modality"].append("rgb")
    c["data_regime"] = sorted(list(set(c["data_regime"])))
    c["tricks"] = sorted(list(set(c["tricks"])))
    c["resources"] = sorted(list(set(c["resources"])))
    return c

def _nl_detect_window_days(text: str, default_days: int = 365) -> int:
    for pat, unit in _CN_TIME_PATTERNS:
        m = re.search(pat, text, flags=re.I)
        if m:
            n = int(m.group(2))
            if unit == "days":
                return max(1, n)
            if unit == "months":
                return max(1, n * 30)
            if unit == "years":
                return max(1, n * 365)
    if any(k in text for k in ["最新", "近期", "最近", "这半年"]):
        return 180
    return default_days

def sota_by_nl(query: str) -> str:
    """
    自然语言查询最新 SOTA（支持中文）：自动解析 benchmark / scope / constraints / 时间窗口。
    例：'找 GOT-10k 上纯监督、不要自监督、近一年最新的 SOTA，且不使用额外数据'
    """
    benchmark = _nl_detect_benchmark(query)
    include_scopes, exclude_scopes, strict_scope = _nl_detect_scopes(query)
    constraints = _nl_detect_constraints(query)
    if include_scopes:
        constraints["include_scopes"] = sorted(list(include_scopes))
    if exclude_scopes:
        constraints["exclude_scopes"] = sorted(list(exclude_scopes))
    if strict_scope:
        constraints["strict_scope"] = True
    window_days = _nl_detect_window_days(query, 365)
    return get_latest_sota(
        benchmark=benchmark,
        window_days=window_days,
        max_results=150,
        scope="overall",
        constraints=constraints
    )

def recent_by_nl(query: str) -> str:
    """
    自然语言查询近期论文列表（支持中文），自动解析限制条件。
    """
    benchmark = _nl_detect_benchmark(query)
    include_scopes, exclude_scopes, strict_scope = _nl_detect_scopes(query)
    constraints = _nl_detect_constraints(query)
    if include_scopes:
        constraints["include_scopes"] = sorted(list(include_scopes))
    if exclude_scopes:
        constraints["exclude_scopes"] = sorted(list(exclude_scopes))
    if strict_scope:
        constraints["strict_scope"] = True
    window_days = _nl_detect_window_days(query, 180)
    return list_recent_papers(
        benchmark=benchmark,
        limit=20,
        window_days=window_days,
        scope="overall",
        constraints=constraints
    )

use_model = "gemini"

if use_model == "deepseek":
    model = LiteLlm(model="deepseek/deepseek-chat")
if use_model == "gpt-4o":
    model = LiteLlm(model="azure/gpt-4o")
if use_model == "gemini":
    model = LiteLlm(model="gemini/gemini-2.5-flash")


root_agent = Agent(
    name="search_papers_agent",
    model=model,
    description=(
        "Agent to answer questions about the papers."
    ),
    instruction=(
        "You are a helpful agent who can answer user questions about the papers."
    ),
    tools=[
        search_papers,
        extract_info,
        find_papers_by_benchmark,
        get_latest_sota,
        list_recent_papers,
        sota_by_nl,
        recent_by_nl,
    ],
)