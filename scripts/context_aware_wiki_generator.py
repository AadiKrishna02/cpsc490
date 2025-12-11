#!/usr/bin/env python3
"""
Context-Aware Wiki Generator
Uses hybrid search embeddings to find the most relevant documents for wiki generation
"""
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import openai
from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

load_dotenv()

sys.path.insert(0, str(ROOT / "scripts"))

PRODUCTION_SEARCH_AVAILABLE = False
try:
    from production_search import get_production_search
    PRODUCTION_SEARCH_AVAILABLE = True
except ImportError as e:
    print(f"Production search not available: {e}")

LEGACY_SEARCH_AVAILABLE = False
try:
    from federated_search_lazy import LazyFederatedSearch
    LEGACY_SEARCH_AVAILABLE = True
except ImportError as e:
    print(f"Legacy search not available: {e}")

from foia_ai.storage.db import get_session
from foia_ai.storage.models import Document, Page


class ContextAwareWikiGenerator:
    """Generate wiki pages using hybrid search for optimal context retrieval"""
    
    def __init__(self):
        """Initialize wiki generator with best available search system"""
        self.search_system = None
        self.using_production_search = False
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.load_search_system()
    
    def load_search_system(self):
        """Load search system - prefer production (LanceDB/Tantivy), fallback to legacy"""
        
        if PRODUCTION_SEARCH_AVAILABLE:
            try:
                print("Loading production search system (LanceDB + Tantivy)...")
                self.search_system = get_production_search()
                
                has_lancedb = (hasattr(self.search_system, 'table') and 
                              self.search_system.table is not None)
                has_tantivy = (hasattr(self.search_system, 'tantivy_searcher') and 
                              self.search_system.tantivy_searcher is not None)
                
                if has_lancedb or has_tantivy:
                    self.using_production_search = True
                    print("Production search loaded successfully")
                    if not has_lancedb:
                        print("LanceDB missing - using Tantivy only")
                    if not has_tantivy:
                        print("Tantivy missing - using LanceDB only")
                    return
                else:
                    print("Production search has no usable indexes")
                    
            except Exception as e:
                print(f"Production search failed to load: {e}")
        
        if LEGACY_SEARCH_AVAILABLE:
            try:
                print("Falling back to legacy search system (FAISS/BM25)...")
                batch_dir = ROOT / "data" / "search_indexes"
                
                batch_paths = sorted([
                    p for p in batch_dir.iterdir()
                    if p.is_dir() and p.name.startswith("batch_")
                ])
                
                if not batch_paths:
                    print("No batch indices found in data/search_indexes/")
                    print("   Run: python scripts/build_index_batched.py")
                    return
                
                print(f"   Loading {len(batch_paths)} batch indices...")
                self.search_system = LazyFederatedSearch(batch_paths)
                self.using_production_search = False
                print("Legacy search loaded successfully")
                return
                
            except Exception as e:
                print(f"Legacy search failed to load: {e}")
        
        print("No search system available!")
        print("   Options:")
        print("   1. Download pre-built indexes: ./scripts/download_indexes.sh")
        print("   2. Build from scratch: python scripts/build_index_batched.py")
        self.search_system = None
    
    def _find_page_number_for_chunk(self, doc_id: str, chunk_text: str) -> Optional[int]:
        """
        Find the page number for a chunk by querying the database.
        Uses fuzzy matching to find the best matching page.
        """
        from foia_ai.storage.db import get_session
        from foia_ai.storage.models import Document, Page
        
        try:
            with get_session() as session:
                doc = session.query(Document).filter(Document.external_id == doc_id).first()
                if not doc:
                    return None
                
                pages = session.query(Page).filter_by(document_id=doc.id).order_by(Page.page_no).all()
                
                search_text = chunk_text[:200].lower().strip()
                
                best_match = None
                best_match_score = 0
                
                for page in pages:
                    if not page.text:
                        continue
                    
                    page_text_lower = page.text.lower()
                    
                    if search_text in page_text_lower:
                        match_score = len(search_text)
                        if match_score > best_match_score:
                            best_match = page.page_no
                            best_match_score = match_score
                
                return best_match
                
        except Exception as e:
            print(f"Error finding page number for chunk: {e}")
            return None
    
    def retrieve_relevant_context(self, topic: str, max_chunks: int = 40, 
                                  max_context_length: int = 30000,
                                  diversity_mode: str = 'balanced') -> List[Dict]:
        """
        Retrieve the most relevant document chunks for a topic using hybrid search
        
        Args:
            topic: Topic to search for
            max_chunks: Maximum number of chunks to retrieve
            max_context_length: Maximum total character length of context
            diversity_mode: Document diversity strategy
                - 'strict': Max 1 chunk per document (20-40 unique documents)
                - 'balanced': Max 2 chunks per document (15-30 unique documents)
                - 'relaxed': Max 3 chunks per document (10-20 unique documents)
                - 'best': Best chunks regardless of source
            
        Returns:
            List of relevant document chunks with metadata
        """
        if not self.search_system:
            print("Search system not initialized")
            return []
        
        print(f"Retrieving context for topic: '{topic}' (diversity: {diversity_mode})")
        
        results = self.search_system.search(
            query=topic, 
            top_k=max_chunks,
            search_mode="hybrid",
            semantic_weight=0.6,  # 60-40 split as requested
            diversity=diversity_mode
        )
        
        context_chunks = []
        total_length = 0
        min_relevance_threshold = 0.4
        
        print(f"Resolving page numbers from database...")
        print(f"Filtering chunks: only including chunks with relevance_score > {min_relevance_threshold}")
        
        stopped_by_relevance = False
        stopped_by_chunks = False
        stopped_by_length = False
        
        for i, result in enumerate(results):
            if len(context_chunks) >= max_chunks:
                stopped_by_chunks = True
                break
            
            relevance_score = result.get('score', 0)
            
            if relevance_score <= min_relevance_threshold:
                stopped_by_relevance = True
                print(f"Stopping at chunk {i+1}: relevance_score {relevance_score:.3f} <= {min_relevance_threshold}")
                break
            
            chunk_text = result['chunk_text']
            chunk_length = len(chunk_text)
            
            if total_length + chunk_length > max_context_length:
                remaining = max_context_length - total_length
                if remaining > 500:  # Only add if we have meaningful space
                    chunk_text = chunk_text[:remaining] + "..."
                    chunk_length = len(chunk_text)
                else:
                    stopped_by_length = True
                    break
            
            page_no = result.get('page_no')
            if page_no is None:
                page_no = self._find_page_number_for_chunk(result['doc_id'], chunk_text)
            
            context_chunks.append({
                'text': chunk_text,
                'doc_id': result['doc_id'],
                'title': result['title'],
                'source': result['source'],
                'page_no': page_no,  # Now always tries to have a page number
                'relevance_score': relevance_score,
                'semantic_score': result.get('semantic_score', 0),
                'bm25_score': result.get('bm25_score', 0)
            })
            
            total_length += chunk_length
        
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(results)} chunks... (relevance: {relevance_score:.3f})")
        
        chunks_with_pages = sum(1 for c in context_chunks if c['page_no'] is not None)
        
        search_returned = len(results)
        passed_to_llm = len(context_chunks)
        
        print(f"Search returned {search_returned} chunks (requested up to {max_chunks})")
        print(f"Passing {passed_to_llm} chunks to LLM ({total_length:,} chars, limit: {max_context_length:,})")
        
        if stopped_by_relevance:
            print(f"Stopped early: encountered chunk with relevance_score <= {min_relevance_threshold}")
        elif stopped_by_chunks:
            print(f"Stopped: reached max_chunks limit ({max_chunks})")
        elif stopped_by_length:
            print(f"Stopped: reached max_context_length limit ({max_context_length:,} chars)")
        elif passed_to_llm < max_chunks and total_length < max_context_length:
            print(f"Search returned fewer chunks ({search_returned}) than requested ({max_chunks})")
        
        if search_returned > passed_to_llm:
            if stopped_by_relevance:
                print(f"{search_returned - passed_to_llm} chunk(s) excluded due to relevance threshold (>{min_relevance_threshold})")
            elif stopped_by_length:
                print(f"{search_returned - passed_to_llm} chunk(s) excluded due to context length limit")
            elif stopped_by_chunks:
                print(f"{search_returned - passed_to_llm} chunk(s) excluded due to max_chunks limit")
        
        print(f"Page numbers resolved: {chunks_with_pages}/{passed_to_llm} chunks")
        
        return context_chunks
    
    def _validate_wiki_content(self, content: str, use_llm: bool = False):
        """
        Validate wiki content for citation quality.
        Returns tuple of (summary_dict, CitationReport).
        """
        from foia_ai.synthesis.citation_validator import validate_citations
        
        report = validate_citations(content, semantic_check=True, use_llm=use_llm)
        
        summary = {
            'total_citations': report.total_citations,
            'valid': report.valid,
            'invalid': report.invalid,
            'citation_density': report.citation_density,
            'avg_relevance': report.avg_relevance_score,
            'issues': report.issues,
            'warnings': report.warnings,
            'passes_quality': (
                report.citation_density >= 2.0 and  
                report.avg_relevance_score >= 0.7 and  # At least 70% relevance
                report.invalid == 0 and  
                report.valid >= 5  
            )
        }
        
        return summary, report
    
    def _build_refinement_prompt(self, topic: str, content: str, validation: Dict, 
                                 context_text: str, iteration: int, length: str = "medium",
                                 context_chunks: List[Dict] = None) -> str:
        """
        Build a prompt to refine the wiki page based on validation feedback.
        """
        issues_text = []
        
        if validation['citation_density'] < 2.0:
            issues_text.append(
                f"- LOW CITATION DENSITY: Currently {validation['citation_density']:.2f} per 100 words. "
                f"REQUIRED: At least 2.0 per 100 words (1 citation at the end of each line). "
                f"Add more citations throughout the article, with atleast one citation at the end of each line."
            )
        
        if validation['invalid'] > 0:
            issues_text.append(
                f"- INVALID CITATIONS: {validation['invalid']} citation(s) reference documents or pages that don't exist. "
                f"Only use citation keys and information that appear in the source material below."
            )
        
        
        if validation['avg_relevance'] < 0.7:
            issues_text.append(
                f"- LOW RELEVANCE: Average citation relevance is {validation['avg_relevance']*100:.0f}%. "
                f"REQUIRED: At least 70%. Ensure citations actually support the claims they're attached to, do not reference any external information."
            )
        
        if validation['warnings']:
            unsupported_details = []
            for warning in validation['warnings'][:5]:  
                doc_id = warning.citation.document_id if warning.citation else 'unknown'
                page_no = warning.citation.page_no if warning.citation else 'unknown'
                issue = warning.issue or ''
                unsupported_details.append(f"  • {doc_id}, p.{page_no}: {issue[:100]}...")
            
            issues_text.append(
                f"- UNSUPPORTED CITATIONS: {len(validation['warnings'])} citation(s) don't support their claims.\n"
                f"  Examples of unsupported citations:\n" + "\n".join(unsupported_details) + "\n"
                f"  ACTION REQUIRED: For each unsupported citation, either:\n"
                f"    a) Remove the citation if the claim cannot be supported by the source, OR\n"
                f"    b) Rewrite the claim to accurately reflect what the source actually says"
            )
        
        if context_chunks:
            num_available_chunks = len(context_chunks)
            unique_docs = len(set(c['doc_id'] for c in context_chunks))
            
            import re
            citation_pattern = r'\[([A-Za-z0-9_.-]+_p\d+)\]'
            citations_in_content = re.findall(citation_pattern, content)
            unique_citations = len(set(citations_in_content))
            target_unique = max(20, min(unique_docs, int(num_available_chunks * 0.15)))  # At least 15% of chunks, minimum 20, max unique docs
            
            if unique_citations < target_unique:
                issues_text.append(
                    f"- LOW SOURCE DIVERSITY: Currently citing from only {unique_citations} unique chunks/documents.\n"
                    f"  TARGET: Cite from at least {target_unique} different chunks/documents (you have {num_available_chunks} chunks from {unique_docs} unique documents available).\n"
                    f"  ACTION REQUIRED:\n"
                    f"    - Actively seek out and cite from different chunks throughout the article\n"
                    f"    - Do NOT repeatedly cite the same chunks - spread citations across many sources\n"
                    f"    - Every section should include citations from multiple different documents\n"
                    f"    - Use the diversity of available sources to provide comprehensive coverage\n"
                    f"    - Look through ALL available chunks, not just the first few that seem most relevant"
                )
        
        issues_summary = "\n".join(issues_text)
        
        prompt = f"""You are refining a wiki article to meet quality standards.

ARTICLE TOPIC: {topic}

CURRENT QUALITY ISSUES (Iteration {iteration}):
{issues_summary}

QUALITY REQUIREMENTS (your work will be rejected if you do not meet these criteria):
✓ Citation Density: At least 1 citation at the end of each sentence (currently: {validation['citation_density']:.2f})
✓ Citation Relevance: If most citations support the claim, and the claim is supported by the source material (currently: {validation['avg_relevance']*100:.0f}%)
✓ Invalid Citations: There are no citations that don't exist in the source material (currently: {validation['invalid']})

CURRENT ARTICLE:
{content}

AVAILABLE SOURCE MATERIAL:
{context_text}

INSTRUCTIONS:
1. Fix the citation density, citation relevance and invalid citations quality issues listed above - this is CRITICAL
2. Add MORE citations throughout the article - aim for at least 1 citation at the end of every sentence (2 per 100 words)
3. VERY CRITICAL: Ensure every citation DIRECTLY supports its claim - don't cite loosely. We are checking that every citation directly supports the text that it is citing
4. Only cite information that ACTUALLY appears in the source material below 
5. CRITICAL: Use proper markdown headers (## for main sections, ### for subsections, #### for sub-subsections). DO NOT use numbered lists (1., 2., 3.) for section headings. DO NOT use plain text headings without markdown syntax.
6. If a citation doesn't support a claim, either remove the citation OR rewrite the claim to match the source
7. Maintain the article's structure and content quality
8. Do NOT remove factual information - just add proper citations and fix unsupported claims"""

        if length == "exhaustive":
            prompt += """
7. IMPORTANT FOR EXHAUSTIVE MODE: The article MUST be very long (5000-8000+ words)
   - Add extensive subsections and details
   - Try to include as much information from the source material as possible
   - Expand on every topic with comprehensive coverage
   - Do NOT shorten the article during refinement"""

        prompt += "\n\nReturn the COMPLETE revised article with all issues fixed."

        return prompt
    
    def generate_wiki_page(self, topic: str, context_chunks: List[Dict], 
                          style: str = "comprehensive", length: str = "medium",
                          max_iterations: int = 3, max_chunks: int = None, 
                          diversity_mode: str = None) -> Dict:
        """
        Generate a wiki page using retrieved context
        
        Args:
            topic: Topic for the wiki page
            context_chunks: Relevant document chunks from hybrid search
            style: Generation style (comprehensive, concise, technical)
            length: Target length (short, medium, long, exhaustive)
            
        Returns:
            Generated wiki page with metadata
        """
        print(f"Generating wiki page for: '{topic}' (Length: {length})")
        
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            if chunk.get('page_no') is not None:
                citation_key = f"{chunk['doc_id']}_p{chunk['page_no']}"
            else:
                citation_key = f"{chunk['doc_id']}_chunk{i}"
            
            context_parts.append(
                f"[Citation: {citation_key}]\n"
                f"Title: {chunk['title']}\n"
                f"{chunk['text']}"
            )
        
        context_text = "\n\n---\n\n".join(context_parts)
        
        length_instructions = {
            "short": {
                "desc": "a concise summary (approx. 500-800 words)",
                "detail": "Focus only on the most critical high-level facts. Be brief and direct.",
                "tokens": 1500,
                "max_chunks": 30,
                "max_context": 20000
            },
            "medium": {
                "desc": "a standard wiki article (approx. 1000-1500 words)",
                "detail": "Balance high-level overview with specific details. Cover key sub-topics adequately.",
                "tokens": 3000,
                "max_chunks": 40,
                "max_context": 30000
            },
            "long": {
                "desc": "a detailed deep-dive (approx. 2000-4000 words)",
                "detail": "Provide extensive detail, historical context, and thorough analysis of all available information.",
                "tokens": 8000,
                "max_chunks": 80,
                "max_context": 80000
            },
            "exhaustive": {
                "desc": "an exhaustive comprehensive report (8000+ words, maximum length)",
                "detail": "Generate the LONGEST POSSIBLE article. Leave no detail out. Cover every minor finding, date, name, event, and detail mentioned in the source text. Include extensive subsections, exhaustive historical context, comprehensive analysis, and all relevant information. Aim for at least 8000-12000 words minimum. The article should be as comprehensive and detailed as humanly possible based on all available source material.",
                "tokens": 16384,  # Output tokens (≈12K words). Can be increased up to 128K for GPT-5 Nano if needed.
                "max_chunks": 500,  # Maximum chunks for exhaustive coverage
                "max_context": 300000  # 300K chars ≈ 75K tokens input (allows all 500 chunks, uses ~19% of GPT-5 Nano's 400K context window)
            }
        }
        
        target = length_instructions.get(length, length_instructions["medium"])
        
        num_chunks = len(context_chunks)
        unique_docs = len(set(c['doc_id'] for c in context_chunks))
        target_unique_citations = min(unique_docs, max(20, int(num_chunks * 0.1)))  # Aim for at least 10% of chunks, minimum 20
        
        system_prompt = f"""You are an expert intelligence analyst and technical writer. 
You are an wiki writer whose task is to create {target['desc']} about intelligence, 
military, and government topics based on FOIA documents.

Guidelines:
- Write as an expert intelligence analyst, in an encyclopedic, objective tone
- Organize information logically with clear sections
- Include specific details, dates, names, and facts from the source documents
- CRITICAL: Ensure that every sentence has at least one citation, and that the citation is from the source material. 
- EXAMPLE OF PROPER CITATION DENSITY:

"In 1992, Chávez led a coup attempt against the Venezuelan government [VEN_FileId_12345_p3]. 
The operation involved military units from Maracay [VEN_FileId_12346_p7] and was coordinated 
with civilian opposition groups [VEN_FileId_12347_p2]. U.S. intelligence had been monitoring 
the situation for months [CIA_FileId_98765_p15], noting increasing dissatisfaction within 
the military ranks [DIA_FileId_45678_p9]."

This paragraph: ~50 words, 5 citations from 5 different sources ✓
- CRITICAL: Each citation must directly support its claim, do not cite speculatively, and the source text must actually say what you claim it says
- You will be penalized for every citation that does not directly support the text that it is citing
- We run all the citations through a checker to validate that they exist and they say EXACTLY what you claim they say
- You lose a point for every citation that does not directly support the text that it is citing, and after 5 points are lost, the article will be rejected
- Do not make up any citations, you will be penalized for every citation that doesn't exist in the source material or is not supported by the source material
- CRITICAL: Cite sources using the EXACT citation keys from the source headers (e.g., [DIA_FileId_238677_p5])
- Write with richer, sentence-by-sentence annotations linking each factual claim to its exact source line (e.g., “In 1992 Chávez’s coup attempt occurred; SITREP noted X leaders Y actions,” with precise page references)
- Every factual claim must have a citation that is from the source material, do not make anything up, and ensure you don't use external or existing information so that you rely only on the information provided in the source material
- CRITICAL: Cite from DIVERSE sources - use citations from at least {target_unique_citations} different chunks/documents
- Draw information from the ENTIRE source material provided, not just the most obvious chunks
- Highlight key findings and important revelations from multiple perspectives
- Do not give any suggestions at the end of the article, keep it professional and factual
- CRITICAL: If you do not meet the criteria for having valid citations, the required citation density, and the required relevance score, you will be penalized and your article will be rejected.

CRITICAL MARKDOWN FORMATTING REQUIREMENTS:
- Use proper markdown headers for all major sections: # for title, ## for main sections, ### for subsections, #### for sub-subsections
- DO NOT use numbered lists (1., 2., 3.) for major sections - only use markdown headers (##, ###)
- DO NOT use plain text headings without markdown syntax (e.g., don't write "Introduction" - write "## Introduction")
- Structure example:
- Use bullet points (-) only for lists within sections, NOT for section headings
- {target['detail']}
- Focus on verified information from the documents"""

        length_emphasis = ""
        if length == "exhaustive":
            length_emphasis = "\n\nCRITICAL FOR EXHAUSTIVE MODE:\n- Generate the LONGEST possible article (aim for 5000-8000+ words minimum)\n- Use ALL available source material - do not skip any information\n- Create extensive subsections for every topic mentioned\n- Include comprehensive historical context\n- Add detailed analysis and interpretation\n- Cover every date, name, event, and detail from the sources\n- The article should be as long and comprehensive as possible while maintaining quality"

        user_prompt = f"""Create a comprehensive wiki page about: {topic}{length_emphasis}

Use the following declassified FOIA documents as your source material:

{context_text}

Generate a well-structured wiki page with:
- A title using # (single hash for H1 header)
- ## Introduction or ## Overview section
- ## Main sections using ## (double hash for H2 headers)
- ### Subsections using ### (triple hash for H3 headers)
- #### Sub-subsections using #### (quadruple hash for H4 headers) when needed
- IMPORTANT: List all the citations in a References section at the end of the article, using the exact citation keys from the source headers (e.g., [DIA_FileId_238677_p5])
- Key facts and findings within sections
- Historical context where applicable, but only if it is directly supported by the source material
- Citations to source documents using the EXACT citation keys shown above (e.g., [DIA_FileId_238677_p5])

CRITICAL FORMATTING: Use proper markdown headers (##, ###, ####) for all sections. DO NOT use numbered lists (1., 2., 3.) for section headings. DO NOT use plain text headings without markdown syntax.

CRITICAL CITATION REQUIREMENTS:
- Use the citation keys EXACTLY as they appear in the [Citation: ...] headers above
- Cite from at least {target_unique_citations} different chunks/documents (you have {num_chunks} chunks available from {unique_docs} unique documents)
- Do NOT repeatedly use the same citations - actively seek out and cite from many different sources
- Every section should include citations from multiple different documents
- The goal is to synthesize information from MANY sources, not just the first few that seem most relevant

IMPORTANT: You have access to {num_chunks} chunks from {unique_docs} unique documents. Use this diversity to create a comprehensive, well-sourced article with citations spanning many different sources.

Format the page using proper markdown syntax with headers (##, ###, ####), not numbered lists or plain text headings."""

        try:
            print(f"Generating initial wiki page...")
            response = self.openai_client.chat.completions.create(
                model="gpt-5-nano",  # Fast and cost-effective model with 400K context window
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                # Note: GPT-5 Nano only supports default temperature (1), custom values not supported
                max_completion_tokens=128000  # High safety ceiling for GPT-5 Nano (128K max) - actual length controlled by prompt instructions
            )
            
            print("\n" + "="*60)
            print("FULL API RESPONSE:")
            print("="*60)
            print(f"Response ID: {response.id}")
            print(f"Model: {response.model}")
            print(f"Object: {response.object}")
            print(f"Created: {response.created}")
            print(f"Usage: {response.usage}")
            print(f"\nChoices count: {len(response.choices)}")
            for i, choice in enumerate(response.choices):
                print(f"\nChoice {i}:")
                print(f"  Finish reason: {choice.finish_reason}")
                print(f"  Index: {choice.index}")
                if choice.message:
                    print(f"  Role: {choice.message.role}")
                    content = choice.message.content
                    print(f"  Content length: {len(content) if content else 0} chars")
                    print(f"  Content preview (first 500 chars): {content[:500] if content else 'None'}")
                    if content:
                        print(f"  Content (full):\n{json.dumps(content, indent=2)}")
            print("="*60 + "\n")
            
            wiki_content = response.choices[0].message.content
            if not wiki_content or not wiki_content.strip():
                response_dict = {
                    'id': response.id,
                    'model': response.model,
                    'object': response.object,
                    'created': response.created,
                    'usage': {
                        'prompt_tokens': response.usage.prompt_tokens if response.usage else None,
                        'completion_tokens': response.usage.completion_tokens if response.usage else None,
                        'total_tokens': response.usage.total_tokens if response.usage else None,
                    },
                    'choices': [
                        {
                            'index': choice.index,
                            'finish_reason': choice.finish_reason,
                            'message': {
                                'role': choice.message.role if choice.message else None,
                                'content': choice.message.content if choice.message else None,
                            }
                        } for choice in response.choices
                    ]
                }
                print("\nEMPTY CONTENT ERROR - Full response JSON:")
                print(json.dumps(response_dict, indent=2, default=str))
                raise ValueError("LLM returned empty content. Generation failed.")
            
            print(f"Initial generation complete ({len(wiki_content)} chars)")
            
            best_content = wiki_content
            best_validation = None
            best_report = None
            iteration_history = []
            
            for iteration in range(1, max_iterations + 1):
                print(f"\nValidation iteration {iteration}/{max_iterations}...")
                
                validation, report = self._validate_wiki_content(wiki_content, use_llm=True)
                iteration_history.append({
                    'iteration': iteration,
                    'validation': validation
                })
                
                print(f"   Citations: {validation['total_citations']} total, {validation['valid']} valid, {validation['invalid']} invalid")
                print(f"   Density: {validation['citation_density']:.2f} per 100 words (target: ≥2.0)")
                print(f"   Relevance: {validation['avg_relevance']*100:.0f}% (target: ≥70%)")
                
                should_update_best = False
                if best_validation is None:
                    should_update_best = True
                elif validation['passes_quality'] and not best_validation['passes_quality']:
                    should_update_best = True
                elif validation['passes_quality'] == best_validation['passes_quality']:

                    
                    dens_score_curr = min(validation['citation_density'] / 2.0, 1.0) # Cap at 2x target
                    dens_score_best = min(best_validation['citation_density'] / 2.0, 1.0)
                    
                    rel_score_curr = min(validation['avg_relevance'] / 0.7, 1.0)
                    rel_score_best = min(best_validation['avg_relevance'] / 0.7, 1.0)
                    
                    

                    total_citations = validation['valid'] + validation['invalid']
                    valid_percentage = validation['valid'] / total_citations if total_citations > 0 else 0

                    valid_score_curr = min(valid_percentage / 0.95, 1.0)
                    valid_score_best = min(best_validation['valid_percentage'] / 0.95, 1.0)
                    
                    current_score = (dens_score_curr * 0.35) + (rel_score_curr * 0.45) + (valid_score_curr * 0.20)
                    best_score = (dens_score_best * 0.35) + (rel_score_best * 0.45) + (valid_score_best * 0.20)
                    
                    if current_score > best_score:
                        should_update_best = True
                
                if should_update_best:
                    best_content = wiki_content
                    best_validation = validation
                    best_report = report
                
                if validation['passes_quality']:
                    print(f"Quality standards met on iteration {iteration}!")
                    break
                
                if iteration == max_iterations:
                    if best_validation and best_validation['passes_quality']:
                        print(f"Using best version that meets quality standards!")
                        wiki_content = best_content
                        validation = best_validation
                        report = best_report
                    else:
                        print(f"WARNING: Quality standards NOT met after {max_iterations} iterations!")
                        print(f"   Final metrics: Density={validation['citation_density']:.2f} (need ≥2.0), "
                              f"Relevance={validation['avg_relevance']*100:.0f}% (need ≥70%), "
                              f"Valid={validation['valid']} (need ≥5), "
                              f"Invalid={validation['invalid']} (need 0)")
                        print(f"   Publishing anyway, but quality may be substandard.")
                    break
                
                print(f"Refining content to fix quality issues...")
                refinement_prompt = self._build_refinement_prompt(
                    topic, wiki_content, validation, context_text, iteration, length, context_chunks
                )
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-5-nano",  # Fast and cost-effective model with 400K context window
                    messages=[
                        {"role": "system", "content": "You are an expert editor improving wiki article quality. You MUST fix all citation quality issues AND ensure proper markdown formatting with headers (##, ###, ####), not numbered lists or plain text headings."},
                        {"role": "user", "content": refinement_prompt}
                    ],
                    # Note: GPT-5 Nano only supports default temperature (1), custom values not supported
                    max_completion_tokens=128000  # High safety ceiling for GPT-5 Nano (128K max) - actual length controlled by prompt instructions
                )
                
                wiki_content = response.choices[0].message.content
                if not wiki_content or not wiki_content.strip():
                    print(f"WARNING: Refinement returned empty content on iteration {iteration}. Using previous version.")
                    continue 
                
                print(f"Refinement complete ({len(wiki_content)} chars)")
            
            wiki_content = best_content
            
            if not wiki_content or not wiki_content.strip():
                raise ValueError("Failed to generate wiki content. All generation attempts returned empty content.")
            
            if best_report is None:
                final_validation, final_report = self._validate_wiki_content(wiki_content, use_llm=True)
            else:
                final_validation = best_validation
                final_report = best_report
            
            from foia_ai.synthesis.citation_validator import serialize_report
            validation_cache = serialize_report(final_report)
            
            chunks_data = []
            for i, chunk in enumerate(context_chunks):
                text_preview = chunk['text'][:200] + ('...' if len(chunk['text']) > 200 else '')
                chunks_data.append({
                    'index': i,
                    'doc_id': chunk['doc_id'],
                    'title': chunk['title'],
                    'source': chunk['source'],
                    'page_no': chunk.get('page_no'),
                    'text_preview': text_preview,
                    'text_length': len(chunk['text']),
                    'relevance_score': chunk.get('relevance_score', 0),
                    'semantic_score': chunk.get('semantic_score', 0),
                    'bm25_score': chunk.get('bm25_score', 0),
                    'citation_key': f"{chunk['doc_id']}_p{chunk['page_no']}" if chunk.get('page_no') else f"{chunk['doc_id']}_chunk{i}"
                })
            
            metadata = {
                'topic': topic,
                'generated_at': datetime.now().isoformat(),
                'model': 'gpt-5-nano',  
                'num_sources': len(context_chunks),
                'source_documents': [
                    {
                        'doc_id': chunk['doc_id'],
                        'title': chunk['title'],
                        'source': chunk['source'],
                        'relevance': chunk['relevance_score']
                    }
                    for chunk in context_chunks
                ],
                'chunks': chunks_data,  
                'context_length': len(context_text),
                'style': style,
                'length_mode': length,
                'max_chunks': max_chunks,
                'diversity_mode': diversity_mode,
                'quality_iterations': len(iteration_history),
                'iteration_history': iteration_history,  
                'final_quality': {
                    'citation_density': final_validation['citation_density'],
                    'avg_relevance': final_validation['avg_relevance'],
                    'total_citations': final_validation['total_citations'],
                    'valid_citations': final_validation['valid'],
                    'invalid_citations': final_validation['invalid'],
                    'passes_standards': final_validation['passes_quality']
                },
                'validation_cache': validation_cache  
            }
            
            print(f"\nFinal wiki page ready:")
            print(f"   Citations: {final_validation['total_citations']} ({final_validation['valid']} valid)")
            print(f"   Density: {final_validation['citation_density']:.2f} per 100 words")
            print(f"   Relevance: {final_validation['avg_relevance']*100:.0f}%")
            print(f"   Quality: {'PASSED' if final_validation['passes_quality'] else 'NEEDS REVIEW'}")
            
            return {
                'content': wiki_content,
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"Error generating wiki page: {e}")
            raise
    
    def _normalize_markdown_formatting(self, content: str) -> str:
        """
        Normalize markdown formatting to ensure consistent structure.
        """
        lines = content.split('\n')
        normalized_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if not stripped:
                normalized_lines.append(line)
                continue
            
            numbered_heading = re.match(r'^(\d+)\.\s+([A-Z][^:]+):?\s*$', stripped)
            if numbered_heading:
                is_heading = False
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if not next_line or next_line.startswith('-') or next_line.startswith('*'):
                        is_heading = True
                elif i == 0 or (normalized_lines and not normalized_lines[-1].strip()):
                    is_heading = True
                
                if is_heading:
                    heading_text = numbered_heading.group(2).strip()
                    normalized_lines.append(f"## {heading_text}")
                    continue
            
            normalized_lines.append(line)
        
        return '\n'.join(normalized_lines)
    
    def save_wiki_page(self, topic: str, wiki_data: Dict) -> Path:
        """Save generated wiki page to file, creating unique filename if one already exists"""
        from datetime import datetime
        
        safe_topic = topic.replace('/', '_').replace('\\', '_').replace(' ', '_')
        safe_topic = ''.join(c for c in safe_topic if c.isalnum() or c in '_-')
        
        wiki_dir = ROOT / "data" / "wiki"
        wiki_dir.mkdir(exist_ok=True)
        
        base_path = wiki_dir / safe_topic
        md_path = base_path.with_suffix('.md')
        
        if md_path.exists():
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_topic_unique = f"{safe_topic}_{timestamp}"
            md_path = wiki_dir / f"{safe_topic_unique}.md"
            meta_path = wiki_dir / f"{safe_topic_unique}.meta.json"
            print(f"File {safe_topic}.md already exists. Saving as {safe_topic_unique}.md")
        else:
            meta_path = wiki_dir / f"{safe_topic}.meta.json"
        
        print("Formatting citations with hyperlinks...")
        
        def convert_single_citation(cite_text):
            """Convert a single citation to clickable link"""
            cite_text = cite_text.strip()
            
            if '_p' in cite_text:
                parts = cite_text.rsplit('_p', 1)
                if len(parts) == 2:
                    doc_id, page_str = parts
                    
                    doc_number = doc_id.split('_')[-1] if '_' in doc_id else doc_id
                    
                    clean_text = f"Doc #{doc_number}, p.{page_str}"
                    
                    link = f"/pdf/{doc_id}#page={page_str}"
                    
                    return f'<a href="{link}" class="citation" target="_blank">({clean_text})</a>'
            
            return f'<a href="/documents/{cite_text}" class="citation" target="_blank">[{cite_text}]</a>'
        
        def replace_citation(match):
            """Convert [DocID_pPageNo] or [DocID1_pPageNo1; DocID2_pPageNo2] to clickable links"""
            citations_text = match.group(1)  # e.g., "DIA_FileId_238677_p5" or "CIA_comments_1994-2005u_p162; NARAjfkAssassinationRecordsCollWithheld_2016_p57"
            
            if ';' in citations_text:
                citations = [convert_single_citation(c) for c in citations_text.split(';')]
                return ' '.join(citations)
            else:
                return convert_single_citation(citations_text)
        
        content = wiki_data['content']
        bracket_pattern = r'\[([A-Za-z0-9_.-]+(?:_p\d+|_chunk\d+)(?:\s*;\s*[A-Za-z0-9_.-]+(?:_p\d+|_chunk\d+))*)\]'
        paren_pattern = r'\(([A-Za-z0-9_.-]+(?:_p\d+|_chunk\d+)(?:\s*;\s*[A-Za-z0-9_.-]+(?:_p\d+|_chunk\d+))*)\)'
        content = re.sub(bracket_pattern, replace_citation, content)
        content = re.sub(paren_pattern, replace_citation, content)
        
        content = re.sub(r'<a href="([^"]+)"[^>]*><a href="\1"[^>]*>([^<]+)</a></a>', r'<a href="\1" class="citation" target="_blank">\2</a>', content)
        content = re.sub(r'\(#\)', '', content)
        
        if not content or not content.strip():
            raise ValueError(f"Cannot save empty wiki page for topic: {topic}. Generation must have failed silently.")
        
        content = self._normalize_markdown_formatting(content)
        
        try:
            from foia_ai.synthesis.wiki_service import enhance_references_section
            content = enhance_references_section(content)
        except Exception:
            pass
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(wiki_data['metadata'], f, indent=2)
        
        print(f"Wiki page saved to: {md_path}")
        return md_path
    
    def generate_and_save(self, topic: str, max_chunks: int = 40, diversity_mode: str = 'balanced', 
                         length: str = 'medium', max_quality_iterations: int = 3) -> Path:
        """
        Complete workflow: retrieve context, generate, and save wiki page
        
        Args:
            topic: Topic for the wiki page
            max_chunks: Maximum number of context chunks to use
            diversity_mode: Document diversity strategy
            length: Target length (short, medium, long, exhaustive)
            max_quality_iterations: Maximum iterations for quality refinement (default: 3)
            
        Returns:
            Path to saved wiki page
        """
        print("="*60)
        print(f"Generating Wiki Page: {topic} (Length: {length})")
        print("="*60)
        
        length_config = {
            "short": {"max_chunks": 30, "max_context": 20000},
            "medium": {"max_chunks": 40, "max_context": 30000},
            "long": {"max_chunks": 80, "max_context": 80000},
            "exhaustive": {"max_chunks": 500, "max_context": 300000}  # Maximum chunks for exhaustive coverage with GPT-5 Nano
        }
        config = length_config.get(length, length_config["medium"])
        
        effective_max_chunks = max(max_chunks, config["max_chunks"])
        effective_max_context = config["max_context"]
        
        print(f"Retrieving context: up to {effective_max_chunks} chunks, {effective_max_context:,} chars")
        
        context_chunks = self.retrieve_relevant_context(
            topic, 
            max_chunks=effective_max_chunks,
            max_context_length=effective_max_context,
            diversity_mode=diversity_mode
        )
        
        if not context_chunks:
            raise ValueError(f"No relevant documents found for topic: {topic}")
        
        wiki_data = self.generate_wiki_page(topic, context_chunks, length=length, 
                                           max_iterations=max_quality_iterations,
                                           max_chunks=effective_max_chunks,
                                           diversity_mode=diversity_mode)
        
        wiki_path = self.save_wiki_page(topic, wiki_data)
        
        print("Re-validating final file content with HTML citations...")
        final_content = wiki_path.read_text(encoding='utf-8')
        final_validation, final_report = self._validate_wiki_content(final_content, use_llm=True)
        
        from foia_ai.synthesis.citation_validator import serialize_report
        final_validation_cache = serialize_report(final_report)
        
        meta_path = wiki_path.parent / f"{wiki_path.stem}.meta.json"
        if meta_path.exists():
            import json
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            metadata['validation_cache'] = final_validation_cache
            metadata['final_quality'] = {
                'citation_density': final_validation['citation_density'],
                'avg_relevance': final_validation['avg_relevance'],
                'total_citations': final_validation['total_citations'],
                'valid_citations': final_validation['valid'],
                'invalid_citations': final_validation['invalid'],
                'passes_standards': final_validation['passes_quality']
            }
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            print(f"Updated validation cache with final citation count: {final_validation['total_citations']}")
        
        print("="*60)
        print("Wiki Page Generation Complete!")
        print("="*60)
        print(f"Topic: {topic}")
        print(f"Sources: {len(context_chunks)} documents")
        print(f"Content: {len(final_content):,} characters")
        print(f"Citations: {final_validation['total_citations']} total ({final_validation['valid']} valid)")
        print(f"Saved to: {wiki_path}")
        print("="*60)
        
        return wiki_path
    
    def batch_generate(self, topics: List[str], max_chunks: int = 40):
        """Generate wiki pages for multiple topics"""
        print(f"Batch generating {len(topics)} wiki pages...")
        
        results = []
        for i, topic in enumerate(topics, 1):
            print(f"\n[{i}/{len(topics)}] Processing: {topic}")
            try:
                wiki_path = self.generate_and_save(topic, max_chunks)
                results.append({
                    'topic': topic,
                    'status': 'success',
                    'path': str(wiki_path)
                })
            except Exception as e:
                print(f"Failed to generate wiki for '{topic}': {e}")
                results.append({
                    'topic': topic,
                    'status': 'failed',
                    'error': str(e)
                })
        
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = len(results) - successful
        
        print("\n" + "="*60)
        print("Batch Generation Summary")
        print("="*60)
        print(f"Total: {len(topics)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print("="*60)
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Context-Aware Wiki Generator")
    parser.add_argument("topic", nargs="?", help="Topic for wiki page")
    parser.add_argument("--topics", nargs="+", help="Multiple topics to generate")
    parser.add_argument("--topics-file", help="File with topics (one per line)")
    parser.add_argument("--index", help="Path to search index")
    parser.add_argument("--max-chunks", type=int, default=40, help="Max context chunks")
    parser.add_argument("--diversity", choices=['strict', 'balanced', 'relaxed', 'best'], 
                       default='balanced', help="Document diversity mode")
    
    args = parser.parse_args()
    
    generator = ContextAwareWikiGenerator(search_index_path=args.index)
    
    topics = []
    if args.topic:
        topics = [args.topic]
    elif args.topics:
        topics = args.topics
    elif args.topics_file:
        with open(args.topics_file, 'r') as f:
            topics = [line.strip() for line in f if line.strip()]
    else:
        print("No topic specified. Use --topic, --topics, or --topics-file")
        return
    
    if len(topics) == 1:
        generator.generate_and_save(topics[0], max_chunks=args.max_chunks, diversity_mode=args.diversity)
    else:
        generator.batch_generate(topics, max_chunks=args.max_chunks)


if __name__ == "__main__":
    main()
