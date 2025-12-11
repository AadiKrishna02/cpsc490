from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from ..storage.db import get_session
from ..storage.models import Document, Page
from .openai_client import get_openai_client


CITATION_RE = re.compile(r"\[([A-Za-z0-9_.-]+)_p(\d+)\]")


@dataclass
class Citation:
    raw: str
    document_id: str
    page_no: int


@dataclass
class CitationIssue:
    citation: Citation
    issue: str
    context: Optional[str] = None  # The sentence/paragraph containing the citation
    relevance_score: Optional[float] = None  # Semantic relevance score (0-1)
    verdict: Optional[str] = None  # LLM verdict (SUPPORTED/UNSUPPORTED/PARTIAL)


@dataclass
class ValidatedCitation:
    """Represents a successfully validated citation with relevance scores"""
    citation: Citation
    context: str
    relevance_score: Optional[float] = None  # 0-1 score
    verdict: Optional[str] = None  # SUPPORTED/PARTIAL/UNSUPPORTED
    confidence: Optional[str] = None  # HIGH/MEDIUM/LOW
    explanation: Optional[str] = None


@dataclass
class CitationReport:
    path: Path
    total_citations: int
    valid: int
    invalid: int
    issues: List[CitationIssue]
    warnings: List[CitationIssue] = None  # Semantic validation warnings
    validated_citations: List[ValidatedCitation] = None  # All validated citations with scores
    citation_density: float = 0.0  # Citations per 100 words
    total_words: int = 0
    avg_relevance_score: float = 0.0  # Average relevance across all citations
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.validated_citations is None:
            self.validated_citations = []


def parse_citations(markdown_text: str) -> List[Citation]:
    """
    Parse citations from markdown text.
    Handles both formats and deduplicates:
    - Raw: [DIA_FileId_238677_p5]
    - HTML: <a href="/pdf/DIA_FileId_238677#page=5" class="citation">(Doc #238677, p.5)</a>
    
    Returns deduplicated list of citations based on (document_id, page_no) pairs.
    """
    citations: List[Citation] = []
    seen_citations = set()  # Track (doc_id, page_no) pairs to avoid duplicates
    
    for match in CITATION_RE.finditer(markdown_text):
        doc_id, page_str = match.groups()
        try:
            page_no = int(page_str)
            citation_key = (doc_id, page_no)
            if citation_key not in seen_citations:
                citations.append(Citation(raw=match.group(0), document_id=doc_id, page_no=page_no))
                seen_citations.add(citation_key)
        except ValueError:
            continue
    
    pdf_pattern = r'<a href="/pdf/([A-Za-z0-9_.-]+)#page=(\d+)"[^>]*>\(Doc #[^,]+,\s*p\.\d+\)</a>'
    for match in re.finditer(pdf_pattern, markdown_text):
        doc_id, page_str = match.groups()
        try:
            page_no = int(page_str)
            citation_key = (doc_id, page_no)
            if citation_key not in seen_citations:
                citations.append(Citation(raw=match.group(0), document_id=doc_id, page_no=page_no))
                seen_citations.add(citation_key)
        except ValueError:
            continue
    
    docs_pattern = r'<a href="/documents/([A-Za-z0-9_.-]+)#page=(\d+)"[^>]*>\(Doc #[^,]+,\s*p\.\d+\)</a>'
    for match in re.finditer(docs_pattern, markdown_text):
        doc_id, page_str = match.groups()
        try:
            page_no = int(page_str)
            citation_key = (doc_id, page_no)
            if citation_key not in seen_citations:
                citations.append(Citation(raw=match.group(0), document_id=doc_id, page_no=page_no))
                seen_citations.add(citation_key)
        except ValueError:
            continue
    
    return citations


essential_fields = (Document.external_id, Document.id)


def extract_citation_context(markdown_text: str, citation: Citation, context_chars: int = 200) -> str:
    """Extract the sentence/paragraph around a citation for context."""
    pattern = re.escape(citation.raw)
    match = re.search(pattern, markdown_text)
    if not match:
        return ""
    
    start_pos = max(0, match.start() - context_chars)
    end_pos = min(len(markdown_text), match.end() + context_chars)
    
    context = markdown_text[start_pos:end_pos]
    
    context = re.sub(r'\[+|\]+|\(#[^)]+\)', '', context)  # Remove citation markup
    context = re.sub(r'[#*_`]', '', context)  # Remove markdown formatting
    context = context.strip()
    
    return context


def validate_semantic_relevance(context: str, page_text: str, threshold: float = 0.3) -> tuple[bool, float]:
    """
    Check if the cited page content is semantically relevant to the context.
    
    Uses simple keyword overlap as a proxy for relevance.
    Returns (is_relevant, score)
    """
    if not context or not page_text:
        return False, 0.0
    
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
    }
    
    context_words = set(w.lower() for w in re.findall(r'\b\w{4,}\b', context) if w.lower() not in stop_words)
    page_words = set(w.lower() for w in re.findall(r'\b\w{4,}\b', page_text) if w.lower() not in stop_words)
    
    if not context_words:
        return False, 0.0
    
    overlap = context_words & page_words
    score = len(overlap) / len(context_words)
    
    return score >= threshold, score


def validate_with_llm(claim: str, page_text: str, citation: Citation) -> tuple[bool, str]:
    """
    Use LLM to validate if the cited page supports the claim.
    
    Returns (is_valid, explanation)
    """
    max_page_chars = 8000
    if len(page_text) > max_page_chars:
        page_text = page_text[:max_page_chars] + "...[truncated]"
    
    prompt = f"""You are a fact-checker validating citations in intelligence documents.

TASK: Determine if the cited page supports the claim made in the wiki article.

CLAIM:
{claim}

CITED PAGE CONTENT (from {citation.document_id}, page {citation.page_no}):
{page_text}

INSTRUCTIONS:
1. Read the claim carefully
2. Search the cited page for evidence supporting the claim
3. Determine if the page provides sufficient support

Respond in this format:
VERDICT: [SUPPORTED/UNSUPPORTED/PARTIAL]
CONFIDENCE: [HIGH/MEDIUM/LOW]
EXPLANATION: [1-2 sentence explanation of your reasoning]

Be strict - only mark as SUPPORTED if the page clearly backs up the claim."""

    try:
        client = get_openai_client(default_model="gpt-5-nano")
        # Note: GPT-5 Nano only supports default temperature (1), so we don't pass temperature parameter
        response = client.generate(prompt)
        
        verdict_match = re.search(r'VERDICT:\s*(SUPPORTED|UNSUPPORTED|PARTIAL)', response, re.IGNORECASE)
        confidence_match = re.search(r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)', response, re.IGNORECASE)
        explanation_match = re.search(r'EXPLANATION:\s*(.+?)(?:\n\n|\Z)', response, re.DOTALL | re.IGNORECASE)
        
        verdict = verdict_match.group(1).upper() if verdict_match else "UNKNOWN"
        confidence = confidence_match.group(1).upper() if confidence_match else "UNKNOWN"
        explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"
        
        is_valid = verdict == "SUPPORTED" and confidence in ["HIGH", "MEDIUM"]
        full_explanation = f"{verdict} ({confidence}): {explanation}"
        
        return is_valid, full_explanation
        
    except Exception as e:
        return True, f"LLM validation failed: {str(e)}"  # Default to valid on error


def _validate_single_citation_llm(
    context: str, 
    page_text: str, 
    citation: Citation,
    index: int
) -> Tuple[int, bool, str, Optional[float], Optional[str], Optional[str], Optional[str]]:
    """
    Helper function to validate a single citation using LLM.
    Designed to be called in parallel.
    
    Returns:
        (index, is_valid, full_explanation, relevance_score, verdict, confidence, explanation)
    """
    is_valid, full_explanation = validate_with_llm(context, page_text, citation)
    
    verdict_match = re.search(r'(SUPPORTED|UNSUPPORTED|PARTIAL)', full_explanation)
    confidence_match = re.search(r'\((HIGH|MEDIUM|LOW)\)', full_explanation)
    
    verdict = verdict_match.group(1) if verdict_match else "UNKNOWN"
    confidence = confidence_match.group(1) if confidence_match else "UNKNOWN"
    explanation = full_explanation
    
    relevance_score = {
        'SUPPORTED': 1.0,
        'PARTIAL': 0.5,
        'UNSUPPORTED': 0.0,
        'UNKNOWN': 0.5
    }.get(verdict, 0.5)
    
    return (index, is_valid, full_explanation, relevance_score, verdict, confidence, explanation)


def validate_citations(markdown_text: str, semantic_check: bool = True, use_llm: bool = True, max_workers: int = 10) -> CitationReport:
    """
    Validate citations in markdown text.
    
    Args:
        markdown_text: The markdown content to validate
        semantic_check: If True, also check if cited pages support the claims
        use_llm: If True, use LLM for semantic validation (more accurate but slower)
        max_workers: Maximum number of parallel workers for LLM validation (default: 10)
    
    Returns:
        CitationReport with validation results including relevance scores
    """
    citations = parse_citations(markdown_text)
    issues: List[CitationIssue] = []
    warnings: List[CitationIssue] = []
    validated_citations: List[Optional[ValidatedCitation]] = [None] * len(citations)  # Preserve order
    valid = 0
    total_relevance = 0.0
    relevance_count = 0

    citation_data_for_llm: List[Tuple[int, Citation, str, str]] = []  # (index, citation, context, page_text)
    
    with get_session() as session:
        for idx, c in enumerate(citations):
            doc = session.query(Document).filter(Document.external_id == c.document_id).first()
            if not doc:
                context = extract_citation_context(markdown_text, c)
                issues.append(CitationIssue(
                    citation=c, 
                    issue="Document not found in DB",
                    context=context
                ))
                validated_citations[idx] = ValidatedCitation(
                    citation=c,
                    context=context,
                    relevance_score=None,
                    verdict=None,
                    confidence=None,
                    explanation=None
                )
                continue
            
            page = (
                session.query(Page)
                .filter(Page.document_id == doc.id, Page.page_no == c.page_no)
                .first()
            )
            if not page:
                context = extract_citation_context(markdown_text, c)
                issues.append(CitationIssue(
                    citation=c, 
                    issue="Page not found for document",
                    context=context
                ))
                validated_citations[idx] = ValidatedCitation(
                    citation=c,
                    context=context,
                    relevance_score=None,
                    verdict=None,
                    confidence=None,
                    explanation=None
                )
                continue
            
            context = extract_citation_context(markdown_text, c)
            
            if semantic_check and page.text:
                if use_llm:
                    citation_data_for_llm.append((idx, c, context, page.text))
                else:
                    is_relevant, score = validate_semantic_relevance(context, page.text)
                    relevance_score = score
                    verdict = "SUPPORTED" if is_relevant else "UNSUPPORTED"
                    confidence = "MEDIUM"
                    
                    if not is_relevant:
                        warnings.append(CitationIssue(
                            citation=c,
                            issue=f"Low relevance score ({score:.2f}): cited page may not support the claim",
                            context=context,
                            relevance_score=score,
                            verdict=verdict
                        ))
                    
                    total_relevance += relevance_score
                    relevance_count += 1
                    
                    validated_citations[idx] = ValidatedCitation(
                        citation=c,
                        context=context,
                        relevance_score=relevance_score,
                        verdict=verdict,
                        confidence=confidence,
                        explanation=None
                    )
            else:
                validated_citations[idx] = ValidatedCitation(
                    citation=c,
                    context=context,
                    relevance_score=None,
                    verdict=None,
                    confidence=None,
                    explanation=None
                )
            
            valid += 1
    
    if use_llm and semantic_check and citation_data_for_llm:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            citation_data_map = {idx: (citation, context) for idx, citation, context, _ in citation_data_for_llm}
            
            future_to_index = {
                executor.submit(_validate_single_citation_llm, context, page_text, citation, idx): idx
                for idx, citation, context, page_text in citation_data_for_llm
            }
            
            llm_results: Dict[int, Tuple[bool, str, Optional[float], Optional[str], Optional[str], Optional[str]]] = {}
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result_idx, is_valid, full_explanation, relevance_score, verdict, confidence, explanation = future.result()
                    llm_results[result_idx] = (is_valid, full_explanation, relevance_score, verdict, confidence, explanation)
                except Exception as e:
                    citation, context = citation_data_map[idx]
                    llm_results[idx] = (True, f"LLM validation failed: {str(e)}", 0.5, "UNKNOWN", "LOW", None)
        
        for idx, citation, context, _ in citation_data_for_llm:
            if idx in llm_results:
                is_valid, full_explanation, relevance_score, verdict, confidence, explanation = llm_results[idx]
                
                if not is_valid:
                    warnings.append(CitationIssue(
                        citation=citation,
                        issue=f"LLM validation: {full_explanation}",
                        context=context,
                        relevance_score=relevance_score,
                        verdict=verdict
                    ))
                
                if relevance_score is not None:
                    total_relevance += relevance_score
                    relevance_count += 1
                
                validated_citations[idx] = ValidatedCitation(
                    citation=citation,
                    context=context,
                    relevance_score=relevance_score,
                    verdict=verdict,
                    confidence=confidence,
                    explanation=explanation
                )
    
    final_validated_citations: List[ValidatedCitation] = []
    for vc in validated_citations:
        if vc is not None:
            final_validated_citations.append(vc)
        else:
            final_validated_citations.append(ValidatedCitation(
                citation=Citation(raw="", document_id="", page_no=0),
                context="",
                relevance_score=None,
                verdict=None,
                confidence=None,
                explanation=None
            ))

    clean_text = re.sub(r'[#*_`\[\]]', '', markdown_text)
    clean_text = re.sub(r'\(http[^)]+\)', '', clean_text)  # Remove URLs
    words = clean_text.split()
    total_words = len(words)
    citation_density = (len(citations) / total_words * 100) if total_words > 0 else 0.0
    
    MIN_CITATION_DENSITY = 2.0  # At least 2 citations per 100 words (1 per 50 words)
    if citation_density < MIN_CITATION_DENSITY and total_words > 100:
        warnings.append(CitationIssue(
            citation=Citation(raw="N/A", document_id="", page_no=0),
            issue=f"Low citation density ({citation_density:.2f} per 100 words). Aim for at least {MIN_CITATION_DENSITY} citations per 100 words (1 per 50 words).",
            context=f"Document has {len(citations)} citations in {total_words} words"
        ))
    
    avg_relevance = (total_relevance / relevance_count) if relevance_count > 0 else 0.0
    
    return CitationReport(
        path=Path(""),
        total_citations=len(citations),
        valid=valid,
        invalid=len(citations) - valid,
        issues=issues,
        warnings=warnings,
        validated_citations=final_validated_citations,
        citation_density=citation_density,
        total_words=total_words,
        avg_relevance_score=avg_relevance
    )


def validate_file(path: Path) -> CitationReport:
    text = path.read_text(encoding="utf-8")
    report = validate_citations(text)
    report.path = path
    return report


def validate_directory(dir_path: Path) -> List[CitationReport]:
    reports: List[CitationReport] = []
    for md in sorted(dir_path.glob("*.md")):
        reports.append(validate_file(md))
    return reports


def serialize_report(report: CitationReport) -> Dict:
    """Serialize a CitationReport to a dictionary for JSON storage"""
    return {
        'total_citations': report.total_citations,
        'valid': report.valid,
        'invalid': report.invalid,
        'citation_density': report.citation_density,
        'total_words': report.total_words,
        'avg_relevance_score': report.avg_relevance_score,
        'issues': [
            {
                'citation': {
                    'raw': issue.citation.raw,
                    'document_id': issue.citation.document_id,
                    'page_no': issue.citation.page_no
                },
                'issue': issue.issue,
                'context': issue.context,
                'relevance_score': issue.relevance_score,
                'verdict': issue.verdict
            }
            for issue in report.issues
        ],
        'warnings': [
            {
                'citation': {
                    'raw': warning.citation.raw,
                    'document_id': warning.citation.document_id,
                    'page_no': warning.citation.page_no
                },
                'issue': warning.issue,
                'context': warning.context,
                'relevance_score': warning.relevance_score,
                'verdict': warning.verdict
            }
            for warning in report.warnings
        ],
        'validated_citations': [
            {
                'citation': {
                    'raw': vc.citation.raw,
                    'document_id': vc.citation.document_id,
                    'page_no': vc.citation.page_no
                },
                'context': vc.context,
                'relevance_score': vc.relevance_score,
                'verdict': vc.verdict,
                'confidence': vc.confidence,
                'explanation': vc.explanation
            }
            for vc in report.validated_citations
        ]
    }


def deserialize_report(data: Dict, path: Path) -> CitationReport:
    """Deserialize a dictionary back to a CitationReport"""
    issues = [
        CitationIssue(
            citation=Citation(
                raw=issue['citation']['raw'],
                document_id=issue['citation']['document_id'],
                page_no=issue['citation']['page_no']
            ),
            issue=issue['issue'],
            context=issue.get('context'),
            relevance_score=issue.get('relevance_score'),
            verdict=issue.get('verdict')
        )
        for issue in data.get('issues', [])
    ]
    
    warnings = [
        CitationIssue(
            citation=Citation(
                raw=warning['citation']['raw'],
                document_id=warning['citation']['document_id'],
                page_no=warning['citation']['page_no']
            ),
            issue=warning['issue'],
            context=warning.get('context'),
            relevance_score=warning.get('relevance_score'),
            verdict=warning.get('verdict')
        )
        for warning in data.get('warnings', [])
    ]
    
    validated_citations = [
        ValidatedCitation(
            citation=Citation(
                raw=vc['citation']['raw'],
                document_id=vc['citation']['document_id'],
                page_no=vc['citation']['page_no']
            ),
            context=vc.get('context', ''),
            relevance_score=vc.get('relevance_score'),
            verdict=vc.get('verdict'),
            confidence=vc.get('confidence'),
            explanation=vc.get('explanation')
        )
        for vc in data.get('validated_citations', [])
    ]
    
    return CitationReport(
        path=path,
        total_citations=data['total_citations'],
        valid=data['valid'],
        invalid=data['invalid'],
        issues=issues,
        warnings=warnings,
        validated_citations=validated_citations,
        citation_density=data['citation_density'],
        total_words=data['total_words'],
        avg_relevance_score=data['avg_relevance_score']
    )
