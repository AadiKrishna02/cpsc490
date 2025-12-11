from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict, Counter

from ..storage.db import get_session
from ..storage.models import Document, Page
from .openai_client import get_openai_client
from .citation_validator import validate_file, validate_directory


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENTITY_RESULTS = _PROJECT_ROOT / "data" / "simple_entity_extraction_results.json"
WIKI_DIR = _PROJECT_ROOT / "data" / "wiki"


def load_entity_data() -> dict:
    if not ENTITY_RESULTS.exists():
        raise FileNotFoundError(
            f"Entity results not found at {ENTITY_RESULTS}. Run entity extraction first."
        )
    return json.loads(ENTITY_RESULTS.read_text(encoding="utf-8"))


def create_topic_clusters(entity_data: dict) -> tuple[dict, dict]:
    """
    Discover specific, contextual topics dynamically from entity extraction results.
    
    Strategy:
    - Create compound topics by combining entities (e.g., "FARC Colombia", "Operation Green Ice")
    - Filter out overly generic terms (e.g., "operation", "intelligence")
    - Require minimum document count for relevance
    - Build context-aware topic names
    
    Returns:
        tuple: (topic_clusters, topic_definitions)
    """
    DOC_MIN = 3  # minimum number of documents mentioning the entity
    INCLUDE_CATEGORIES = {"organizations", "countries", "operations", "topics"}
    
    GENERIC_TERMS = {
        "operation", "the operation", "an operation", "operations", "the  operation", "an  operation", "of  operation",
        "intelligence", "assessment", "analysis", "report", "information",
        "security", "military", "forces", "government", "agency",
        "plan", "the plan", "a plan", "the  plan", "program", "activity",
        "group", "unit", "force", "command", "service",
        "training", "missions", "mission", "deployment", "threat",
        "weapons", "arms", "ammunition", "explosives",  # Too generic without context
        "surveillance", "united states", "terrorism"  # Too broad
    }

    detailed = entity_data.get("detailed_extractions", {})
    document_stats = entity_data.get("document_stats", {})

    doc_entities: dict[str, set[str]] = {}
    entity_doc_count: Counter[str] = Counter()

    for doc_id, pages in detailed.items():
        ents_set: set[str] = set()
        for page_no, page_data in pages.items():
            entities = page_data.get("entities", {})
            for cat, ents in entities.items():
                if cat not in INCLUDE_CATEGORIES:
                    continue
                for e in ents:
                    txt = (e.get("text") or "").strip().lower()
                    if not txt:
                        continue
                    ents_set.add(f"{cat}:{txt}")
        if ents_set:
            doc_entities[doc_id] = ents_set
            normalized_entities = {s.split(":", 1)[1] for s in ents_set}
            for ent in normalized_entities:
                entity_doc_count[ent] += 1

    candidates = {
        ent for ent, dcount in entity_doc_count.items() 
        if dcount >= DOC_MIN and ent not in GENERIC_TERMS
    }
    
    compound_topics = {}
    for doc_id, ents in doc_entities.items():
        texts = {s.split(":", 1)[1] for s in ents}
        categories = {s.split(":", 1)[0]: s.split(":", 1)[1] for s in ents}
        
        if "organizations" in [s.split(":", 1)[0] for s in ents] and "countries" in [s.split(":", 1)[0] for s in ents]:
            for ent in ents:
                cat, txt = ent.split(":", 1)
                if cat == "organizations" and txt in candidates:
                    for ent2 in ents:
                        cat2, txt2 = ent2.split(":", 1)
                        if cat2 == "countries" and txt2 in candidates:
                            compound = f"{txt.title()} in {txt2.title()}"
                            if compound not in compound_topics:
                                compound_topics[compound] = []
                            compound_topics[compound].append(doc_id)
        
        for ent in ents:
            cat, txt = ent.split(":", 1)
            if cat == "operations" and txt not in GENERIC_TERMS:
                if any(generic in txt.lower() for generic in ["the ", "an ", "a ", "of "]):
                    continue
                context = None
                for ent2 in ents:
                    cat2, txt2 = ent2.split(":", 1)
                    if cat2 in ["countries", "organizations"] and txt2 not in GENERIC_TERMS:
                        context = txt2.title()
                        break
                if context:
                    compound = f"{txt.title()} ({context})"
                    if compound not in compound_topics:
                        compound_topics[compound] = []
                    compound_topics[compound].append(doc_id)
        
        for ent in ents:
            cat, txt = ent.split(":", 1)
            if cat == "topics":
                for ent2 in ents:
                    cat2, txt2 = ent2.split(":", 1)
                    if cat2 == "countries" and txt2 not in GENERIC_TERMS:
                        compound = f"{txt.title()} in {txt2.title()}"
                        if compound not in compound_topics:
                            compound_topics[compound] = []
                        compound_topics[compound].append(doc_id)

    clusters = defaultdict(list)
    
    for topic, docs in compound_topics.items():
        if len(set(docs)) >= DOC_MIN:  # Ensure minimum doc count
            clusters[topic] = list(set(docs))
    
    for doc_id, ents in doc_entities.items():
        texts = {s.split(":", 1)[1] for s in ents}
        for cand in candidates:
            if cand in texts:
                topic_name = cand.title()
                clusters[topic_name].append(doc_id)

    defs = {}
    for topic in clusters.keys():
        keywords = [w.lower() for w in topic.replace("(", "").replace(")", "").split() if len(w) > 2]
        
        if " in " in topic:
            parts = topic.split(" in ")
            desc = f"Intelligence and operations related to {parts[0]} in {parts[1]}"
        elif "(" in topic and ")" in topic:
            main = topic.split("(")[0].strip()
            context = topic.split("(")[1].replace(")", "").strip()
            desc = f"Documents about {main} in the context of {context}"
        else:
            desc = f"Intelligence documents related to {topic}"
        
        defs[topic] = {
            "keywords": keywords,
            "description": desc
        }

    if not clusters:
        legacy_clusters = defaultdict(list)
        legacy_defs = {
            "Colombian Counterinsurgency": {
                "keywords": ["colombia", "counterinsurgency", "insurgency", "farc", "eln"],
                "description": "Intelligence on Colombian insurgent groups and counterinsurgency operations",
            },
            "Intelligence Operations": {
                "keywords": ["intelligence", "operations", "assessment", "analysis"],
                "description": "General intelligence operations and assessments",
            },
            "Drug Trafficking": {
                "keywords": ["drug trafficking", "narcotics", "cartel", "cocaine"],
                "description": "Drug trafficking operations and related intelligence",
            },
            "Government Agencies": {
                "keywords": ["dia", "cia", "fbi", "dod", "state"],
                "description": "US government agencies and their activities",
            },
        }
        for doc_id, doc_data in document_stats.items():
            if doc_data.get("total_entities", 0) == 0:
                continue
            doc_text = " ".join(
                (
                    (e.get("text") or "").lower()
                    for p in detailed.get(doc_id, {}).values()
                    for es in p.get("entities", {}).values()
                    for e in es
                )
            )
            scored = []
            for topic, td in legacy_defs.items():
                score = sum(doc_text.count(kw) for kw in td["keywords"])
                if score > 0:
                    scored.append((topic, score))
            if scored:
                best = max(scored, key=lambda x: x[1])[0]
                legacy_clusters[best].append(doc_id)
        return dict(legacy_clusters), legacy_defs

    for t in list(clusters.keys()):
        clusters[t] = sorted(clusters[t])
    return dict(clusters), defs


def build_topic_context(topic: str, doc_ids: list[str], entity_data: dict, max_chars: int = 12000) -> dict:
    pages_payload = []
    doc_summaries = []
    entity_counts = Counter()

    with get_session() as session:
        for doc_id in doc_ids:
            stats = entity_data["document_stats"][doc_id]
            doc_summaries.append(
                {
                    "id": doc_id,
                    "title": stats["title"],
                    "source": stats["source"],
                    "pages": stats["pages"],
                    "words": stats["total_words"],
                }
            )
            pages = (
                session.query(Page)
                .join(Document)
                .filter(Document.external_id == doc_id)
                .order_by(Page.page_no)
                .all()
            )
            for p in pages:
                if p.text and len(p.text.strip()) > 100:
                    pages_payload.append(
                        {
                            "document_id": doc_id,
                            "page_no": p.page_no,
                            "text": p.text,
                            "word_count": len(p.text.split()),
                            "citation_key": f"{doc_id}_p{p.page_no}",
                        }
                    )
            doc_extr = entity_data["detailed_extractions"].get(doc_id, {})
            for page_data in doc_extr.values():
                for cat, ents in page_data["entities"].items():
                    for e in ents:
                        entity_counts[f"{cat}:{e['text']}"] += 1

    pages_payload.sort(key=lambda x: x["word_count"], reverse=True)

    selected, used = [], 0
    for pg in pages_payload:
        ln = len(pg["text"]) if pg["text"] else 0
        if used + ln <= max_chars:
            selected.append(pg)
            used += ln
        else:
            break

    return {
        "topic": topic,
        "documents": doc_summaries,
        "top_entities": dict(entity_counts.most_common(15)),
        "pages": selected,
        "stats": {"total_chars": used, "estimated_tokens": used // 4, "selected_pages": len(selected)},
    }


def build_prompt(topic: str, context: dict, topic_defs: dict) -> str:
    desc = topic_defs.get(topic, {}).get("description", "Intelligence topic")
    parts = []
    parts.append(
        f"You are an intelligence analyst creating a Wikipedia-style article about \"{topic}\".\n\n"
        f"TOPIC DESCRIPTION: {desc}\n\n"
        f"CRITICAL CITATION REQUIREMENTS:\n"
        f"- EVERY factual claim MUST be cited with [Doc_ID_PageN]\n"
        f"- Target: At least 1 citation per 50 words (2 per 100 words)\n"
        f"- EVERY sentence with specific information needs a citation\n"
        f"- Multiple citations per paragraph are REQUIRED\n"
        f"- Only cite information that is actually present in the source passages below\n"
        f"- Use neutral, factual tone based strictly on the sources\n\n"
        f"Available documents:\n"
    )
    for d in context["documents"]:
        parts.append(f"- {d['id']}: {d['title']} ({d['source']}, {d['pages']} pages)")
    parts.append("\nTop entities: " + ", ".join(list(context["top_entities"].keys())[:10]))
    parts.append(f"\n\nSource passages ({len(context['pages'])} passages):\n")
    for i, pg in enumerate(context["pages"], 1):
        parts.append(f"\n--- SOURCE {i} [{pg['citation_key']}] ---\n{pg['text']}\n--- END SOURCE ---\n")
    parts.append(
        "\nNow write the wiki-style article with these sections: \n"
        "## Overview\n## Background\n## Key Organizations/Entities\n"
        "## Operations and Activities\n## Analysis and Assessment\n## Timeline\n"
        "## References\n\n"
        "CITATION STYLE EXAMPLES:\n"
        "- Good: 'The DIA reported 150 casualties in the operation [DIA_FileId_239168_p80].'\n"
        "- Good: 'Violence increased by 40% between 2010-2015 [DIA_FileId_238705_p12], with cartel activity concentrated in border regions [DIA_FileId_239183_p45].'\n"
        "- Bad: 'There was significant violence in the region.' (no citation)\n"
        "- Bad: 'Multiple operations were conducted.' (vague, no citation)\n\n"
        "Remember: EVERY factual statement needs a citation. Aim for 1 citation per 50 words minimum (2-3 citations per paragraph). "
        "In the References section, list all cited documents with their full details in this format:\n"
        "- [Document ID]: [Title], [Source], [Pages] pages, [Date if available]"
    )
    return "\n".join(parts)


def enhance_references_section(markdown_text: str) -> str:
    """
    Enhance the References section with clickable links to PDF viewer.
    Uses the same hyperlink format as in-text citations: /pdf/{doc_id}#page={page_no}
    Handles formats like:
    - "- DocID_pPageNo"
    - "DocID_pPageNo"
    - "- DocID: description"
    """
    from ..storage.db import get_session
    from ..storage.models import Document
    
    if "## References" not in markdown_text:
        return markdown_text
    
    parts = markdown_text.split("## References", 1)
    if len(parts) != 2:
        return markdown_text
    
    before_refs = parts[0]
    refs_section = parts[1]
    
    doc_ids = set()
    with get_session() as session:
        docs = session.query(Document.external_id).filter(Document.external_id.isnot(None)).all()
        doc_ids = {d[0] for d in docs}
    
    lines = refs_section.split('\n')
    enhanced_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            enhanced_lines.append(line)
            continue
        
        match = re.match(r'^(\d+\.|\-)\s+([A-Za-z0-9_.-]+(?:_p\d+|_chunk\d+)?)$', line_stripped)
        if match:
            prefix = match.group(1)
            full_ref = match.group(2)
            
            doc_id = full_ref
            page_no = None
            
            if '_p' in full_ref:
                parts_split = full_ref.rsplit('_p', 1)
                if len(parts_split) == 2 and parts_split[1].isdigit():
                    doc_id = parts_split[0]
                    page_no = parts_split[1]
            elif '_chunk' in full_ref:
                parts_split = full_ref.rsplit('_chunk', 1)
                if len(parts_split) == 2 and parts_split[1].isdigit():
                    doc_id = parts_split[0]
            
            if doc_id in doc_ids:
                doc_number = doc_id.split('_')[-1] if '_' in doc_id else doc_id
                
                if page_no:
                    link = f"/pdf/{doc_id}#page={page_no}"
                    display_text = f"Doc #{doc_number}, p.{page_no}"
                    enhanced = f"{prefix} <a href=\"{link}\" class=\"reference-link\" target=\"_blank\">{display_text}</a>"
                else:
                    link = f"/pdf/{doc_id}"
                    display_text = f"Doc #{doc_number}"
                    enhanced = f"{prefix} <a href=\"{link}\" class=\"reference-link\" target=\"_blank\">{display_text}</a>"
                
                enhanced_lines.append(enhanced)
            else:
                enhanced_lines.append(line)
            continue
        
        match = re.match(r'^(\d+\.|\-)\s+([A-Za-z0-9_.-]+(?:_p\d+|_chunk\d+)?):\s*(.+)$', line_stripped)
        if match:
            prefix = match.group(1)
            doc_id_full = match.group(2)
            rest = match.group(3)
            
            doc_id = doc_id_full
            page_no = None
            
            if '_p' in doc_id_full:
                parts_split = doc_id_full.rsplit('_p', 1)
                if len(parts_split) == 2 and parts_split[1].isdigit():
                    doc_id = parts_split[0]
                    page_no = parts_split[1]
            elif '_chunk' in doc_id_full:
                parts_split = doc_id_full.rsplit('_chunk', 1)
                if len(parts_split) == 2 and parts_split[1].isdigit():
                    doc_id = parts_split[0]
            
            if doc_id in doc_ids:
                doc_number = doc_id.split('_')[-1] if '_' in doc_id else doc_id
                
                if page_no:
                    link = f"/pdf/{doc_id}#page={page_no}"
                    display_text = f"Doc #{doc_number}, p.{page_no}"
                else:
                    link = f"/pdf/{doc_id}"
                    display_text = f"Doc #{doc_number}"
                
                enhanced = f"{prefix} <a href=\"{link}\" class=\"reference-link\" target=\"_blank\">{display_text}</a>: {rest}"
                enhanced_lines.append(enhanced)
                continue
        
        match = re.match(r'^([A-Za-z0-9_.-]+(?:_p\d+|_chunk\d+)?)$', line_stripped)
        if match:
            full_ref = match.group(1)
            
            doc_id = full_ref
            page_no = None
            
            if '_p' in full_ref:
                parts_split = full_ref.rsplit('_p', 1)
                if len(parts_split) == 2 and parts_split[1].isdigit():
                    doc_id = parts_split[0]
                    page_no = parts_split[1]
            elif '_chunk' in full_ref:
                parts_split = full_ref.rsplit('_chunk', 1)
                if len(parts_split) == 2 and parts_split[1].isdigit():
                    doc_id = parts_split[0]
            
            if doc_id in doc_ids:
                doc_number = doc_id.split('_')[-1] if '_' in doc_id else doc_id
                
                if page_no:
                    link = f"/pdf/{doc_id}#page={page_no}"
                    display_text = f"Doc #{doc_number}, p.{page_no}"
                else:
                    link = f"/pdf/{doc_id}"
                    display_text = f"Doc #{doc_number}"
                
                enhanced = f"- <a href=\"{link}\" class=\"reference-link\" target=\"_blank\">{display_text}</a>"
                enhanced_lines.append(enhanced)
                continue
        
        enhanced_lines.append(line)
    
    return before_refs + "## References\n" + '\n'.join(enhanced_lines)


def make_citations_clickable(markdown_text: str, topic: str) -> str:
    """
    Convert [Doc_ID_PageN] citations to cleaner, clickable format.
    Converts [DIA_FileId_238677_p5] to <a href="/pdf/DIA_FileId_238677#page=5">(Doc #238677, p.5)</a>
    Uses HTML links instead of markdown to hide the URL.
    """
    def replace_citation(match):
        full_cite = match.group(0)  # e.g., [DIA_FileId_238677_p5]
        inner = match.group(1)  # e.g., DIA_FileId_238677_p5
        
        if '_p' in inner:
            parts = inner.rsplit('_p', 1)
            if len(parts) == 2:
                doc_id, page_str = parts
                
                doc_number = doc_id.split('_')[-1] if '_' in doc_id else doc_id
                
                clean_text = f"Doc #{doc_number}, p.{page_str}"
                
                link = f"/pdf/{doc_id}#page={page_str}"
                
                return f'<a href="{link}" class="citation">({clean_text})</a>'
        
        return full_cite
    
    pattern = r'\[([A-Za-z0-9_.-]+_p\d+)\]'
    return re.sub(pattern, replace_citation, markdown_text)


def list_generated_pages() -> List[Path]:
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(WIKI_DIR.glob("*.md"))


def validate_wiki_dir() -> List[dict]:
    reports = validate_directory(WIKI_DIR)
    out = []
    for r in reports:
        out.append(
            {
                "path": str(r.path),
                "total": r.total_citations,
                "valid": r.valid,
                "invalid": r.invalid,
                "issues": [
                    {"citation": i.citation.raw, "issue": i.issue} for i in r.issues
                ],
            }
        )
    return out


def search_documents_for_topic(topic: str, entity_data: dict) -> list[str]:
    """
    Search all documents for mentions of a custom topic.
    Returns list of document IDs that mention the topic keywords.
    """
    keywords = [w.lower() for w in topic.split() if len(w) > 2]
    
    doc_scores = {}
    
    with get_session() as session:
        all_docs = session.query(Document).all()
        
        for doc in all_docs:
            if not doc.external_id:
                continue
                
            score = 0
            pages = session.query(Page).filter_by(document_id=doc.id).all()
            
            for page in pages:
                if not page.text:
                    continue
                    
                page_text = page.text.lower()
                for keyword in keywords:
                    score += page_text.count(keyword)
            
            detailed = entity_data.get("detailed_extractions", {})
            if doc.external_id in detailed:
                for page_no, page_data in detailed[doc.external_id].items():
                    entities = page_data.get("entities", {})
                    for cat, ents in entities.items():
                        for e in ents:
                            entity_text = e.get("text", "").lower()
                            for keyword in keywords:
                                if keyword in entity_text:
                                    score += 2  # Weight entity matches higher
            
            if score > 0:
                doc_scores[doc.external_id] = score
    
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, score in sorted_docs[:50]]  # Max 50 documents


def build_topic_context(topic: str, doc_ids: list[str], entity_data: dict, max_chars: int = 12000) -> dict:
    """
    Build context for wiki generation with quality thresholds.
    
    Raises:
        ValueError: If insufficient information is found
    """
    pages_payload = []
    doc_summaries = []
    entity_counts = Counter()

    with get_session() as session:
        for doc_id in doc_ids:
            stats = entity_data["document_stats"].get(doc_id)
            if not stats:
                continue
                
            doc_summaries.append({
                "id": doc_id,
                "title": stats["title"],
                "source": stats["source"],
                "pages": stats["pages"],
                "words": stats["total_words"],
            })
            
            doc_obj = session.query(Document).filter(Document.external_id == doc_id).first()
            if not doc_obj:
                continue
            
            pages = session.query(Page).filter_by(document_id=doc_obj.id).order_by(Page.page_no).all()
            
            for p in pages:
                if p.text and len(p.text.strip()) > 100:
                    pages_payload.append({
                        "document_id": doc_id,
                        "page_no": p.page_no,
                        "text": p.text,
                        "word_count": len(p.text.split()),
                        "citation_key": f"{doc_id}_p{p.page_no}",
                    })
            
            doc_extr = entity_data["detailed_extractions"].get(doc_id, {})
            for page_data in doc_extr.values():
                for cat, ents in page_data["entities"].items():
                    for e in ents:
                        entity_counts[f"{cat}:{e['text']}"] += 1

    pages_payload.sort(key=lambda x: x["word_count"], reverse=True)

    selected, used = [], 0
    for pg in pages_payload:
        ln = len(pg["text"]) if pg["text"] else 0
        if used + ln <= max_chars:
            selected.append(pg)
            used += ln

    MIN_DOCUMENTS = 2
    MIN_PAGES = 3
    MIN_WORDS = 500
    
    total_words = sum(pg["word_count"] for pg in selected)
    
    if len(doc_summaries) < MIN_DOCUMENTS:
        raise ValueError(
            f"Insufficient information: Only found {len(doc_summaries)} document(s) about '{topic}'. "
            f"Need at least {MIN_DOCUMENTS} documents to generate a wiki page."
        )
    
    if len(selected) < MIN_PAGES:
        raise ValueError(
            f"Insufficient information: Only found {len(selected)} page(s) about '{topic}'. "
            f"Need at least {MIN_PAGES} pages to generate a wiki page."
        )
    
    if total_words < MIN_WORDS:
        raise ValueError(
            f"Insufficient information: Only found {total_words} words about '{topic}'. "
            f"Need at least {MIN_WORDS} words to generate a meaningful wiki page."
        )

    return {
        "topic": topic,
        "documents": doc_summaries,
        "top_entities": dict(entity_counts.most_common(15)),
        "pages": selected,
        "stats": {
            "total_chars": used,
            "estimated_tokens": used // 4,
            "selected_pages": len(selected),
            "total_words": total_words,
            "total_documents": len(doc_summaries)
        },
    }


def calculate_quality_score(report) -> float:
    """
    Calculate a quality score (0-100) based on validation report.
    
    Scoring:
    - Citation density: 40 points (2.0+ = full points)
    - Valid citations: 40 points (100% valid = full points)
    - No invalid citations: 20 points (0 invalid = full points)
    """
    density_score = min(40, (report.citation_density / 2.0) * 40)
    
    if report.total_citations > 0:
        valid_pct = report.valid / report.total_citations
        valid_score = valid_pct * 40
    else:
        valid_score = 0
    
    if report.total_citations > 0:
        invalid_pct = report.invalid / report.total_citations
        invalid_penalty = invalid_pct * 20
    else:
        invalid_penalty = 0
    
    total_score = density_score + valid_score + (20 - invalid_penalty)
    return total_score


def build_improvement_prompt(topic: str, context: dict, topic_defs: dict, previous_output: str, previous_validation: dict) -> str:
    """
    Build a prompt to improve the wiki page based on validation feedback.
    """
    report = previous_validation["report"]
    score = previous_validation["score"]
    
    issues = []
    if report.citation_density < 2.0:
        issues.append(f"- Citation density is too low ({report.citation_density:.2f} per 100 words). Add more citations to reach 2.0+")
    
    if report.invalid > 0:
        issues.append(f"- {report.invalid} invalid citations found. Only cite pages that exist in the source passages")
    
    if report.total_citations < 10:
        issues.append(f"- Only {report.total_citations} citations. Add more specific citations throughout the article")
    
    issues_text = "\n".join(issues) if issues else "- General quality improvements needed"
    
    desc = topic_defs.get(topic, {}).get("description", "Intelligence topic")
    parts = []
    parts.append(
        f"You are revising a Wikipedia-style article about \"{topic}\" to improve citation quality.\n\n"
        f"PREVIOUS VERSION QUALITY SCORE: {score:.1f}/100\n"
        f"TARGET SCORE: 80+/100\n\n"
        f"ISSUES TO FIX:\n{issues_text}\n\n"
        f"REQUIREMENTS:\n"
        f"- EVERY factual claim MUST be cited with [Doc_ID_PageN]\n"
        f"- Target: At least 2.0 citations per 100 words (1 per 50 words)\n"
        f"- Only cite information from the source passages below\n"
        f"- Ensure ALL citations reference valid pages\n\n"
        f"Available documents:\n"
    )
    for d in context["documents"]:
        parts.append(f"- {d['id']}: {d['title']} ({d['source']}, {d['pages']} pages)")
    
    parts.append(f"\n\nSource passages ({len(context['pages'])} passages):\n")
    for i, pg in enumerate(context["pages"], 1):
        parts.append(f"\n--- SOURCE {i} [{pg['citation_key']}] ---\n{pg['text']}\n--- END SOURCE ---\n")
    
    parts.append(
        f"\n\nPREVIOUS VERSION:\n{previous_output}\n\n"
        f"Now rewrite the article fixing the issues above. Maintain the same structure but:\n"
        f"1. Add more citations (aim for 1 every 50 words)\n"
        f"2. Ensure every citation is valid (check source passages)\n"
        f"3. Keep all factual information but cite it properly\n"
        f"4. Be specific and detailed with citations\n"
    )
    return "\n".join(parts)


def generate_topic(topic: str, max_chars: int = 12000, temperature: float = 0.3, max_iterations: int = 3) -> Path:
    """
    Generate a wiki page for a topic.
    
    Args:
        topic: Topic name (can be from discovered topics or custom)
        max_chars: Maximum characters to include in context
        temperature: LLM temperature (0-1)
    
    Returns:
        Path to generated markdown file
    
    Raises:
        ValueError: If insufficient information is found about the topic
    """
    entity_data = load_entity_data()
    clusters, defs = create_topic_clusters(entity_data)
    
    if topic not in clusters:
        doc_ids = search_documents_for_topic(topic, entity_data)
        if not doc_ids:
            raise ValueError(
                f"No documents found about '{topic}'. "
                f"Try a different topic or check the Documents tab to see available content."
            )
    else:
        doc_ids = clusters[topic]
    
    context = build_topic_context(topic, doc_ids, entity_data, max_chars=max_chars)
    
    client = get_openai_client(default_model="gpt-5-nano")
    
    from .citation_validator import validate_citations
    
    best_output = None
    best_score = -1
    validation_history = []
    
    for iteration in range(max_iterations):
        print(f"\nGeneration Iteration {iteration + 1}/{max_iterations}")
        
        if iteration == 0:
            prompt = build_prompt(topic, context, defs)
        else:
            prompt = build_improvement_prompt(topic, context, defs, best_output, validation_history[-1])
        
        output = client.generate(prompt, temperature=temperature)
        
        report = validate_citations(output, semantic_check=False, use_llm=False)  # Fast validation first
        
        quality_score = calculate_quality_score(report)
        
        print(f"Quality Score: {quality_score:.1f}/100")
        print(f"Citations: {report.total_citations}")
        print(f"Valid: {report.valid}")
        print(f"Invalid: {report.invalid}")
        print(f"Density: {report.citation_density:.2f} per 100 words")
        
        validation_history.append({
            "iteration": iteration + 1,
            "report": report,
            "score": quality_score,
            "output": output
        })
        
        if quality_score > best_score:
            best_score = quality_score
            best_output = output
        
        MIN_QUALITY_SCORE = 80  # Out of 100
        if quality_score >= MIN_QUALITY_SCORE:
            print(f"Quality threshold met ({quality_score:.1f} >= {MIN_QUALITY_SCORE})")
            break
        else:
            print(f"Below threshold ({quality_score:.1f} < {MIN_QUALITY_SCORE}), iterating...")
    
    output = best_output
    print(f"\nFinal Quality Score: {best_score:.1f}/100")
    
    output = make_citations_clickable(output, topic)
    output = enhance_references_section(output)

    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    out_path = WIKI_DIR / f"{topic.replace(' ', '_')}.md"
    out_path.write_text(output, encoding="utf-8")
    
    final_report = validate_citations(output, semantic_check=False, use_llm=False)
    print(f"\nSaved to: {out_path}")
    print(f"Final Stats: {final_report.total_citations} citations, {final_report.citation_density:.2f} density")
    
    return out_path
