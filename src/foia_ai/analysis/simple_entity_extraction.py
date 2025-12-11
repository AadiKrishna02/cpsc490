from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict, Counter

import dateparser

from ..storage.db import get_session
from ..storage.models import Document, Page, Source

LOGGER = logging.getLogger(__name__)


class SimpleEntityExtractor:
    """
    Pattern-based entity extraction for FOIA documents.
    
    Uses regex patterns to identify:
    - Organizations (agencies, military units)
    - Locations (countries, cities)
    - Dates (various formats)
    - Classifications and redactions
    - Operations and codenames
    - Key intelligence topics
    """
    
    def __init__(self):
        self.patterns = {
            'classifications': [
                r'\b(SECRET|TOP SECRET|CONFIDENTIAL|UNCLASSIFIED|CLASSIFIED)\b',
                r'\b(NOFORN|FOUO|FOR OFFICIAL USE ONLY)\b',
                r'\(b\)\([0-9]+\)',  # Redaction codes like (b)(3)
                r'\b(USC \d+)\b',    # US Code references
            ],
            
            'agencies': [
                r'\b(CIA|DIA|FBI|NSA|DEA|ATF|DOD|DOJ|STATE|USAID)\b',
                r'\b(Defense Intelligence Agency|Central Intelligence Agency)\b',
                r'\b(Federal Bureau of Investigation|National Security Agency)\b',
                r'\b(Department of Defense|Department of Justice)\b',
                r'\b(USCINCSO|SOUTHCOM|CENTCOM|EUCOM|PACOM)\b',
            ],
            
            'military': [
                r'\b(\d+(?:st|nd|rd|th)?\s+(?:Infantry|Airborne|Armored|Marine))\b',
                r'\b(Special Forces|Delta Force|Navy SEALs?)\b',
                r'\b(Joint Chiefs|Joint Staff)\b',
            ],
            
            'operations': [
                r'\b(Operation\s+[A-Z][a-zA-Z\s]+)\b',
                r'\b(Plan\s+[A-Z][a-zA-Z\s]+)\b',
                r'\b([A-Z]{2,}\s+(?:OPERATION|PLAN|MISSION))\b',
            ],
            
            'countries': [
                r'\b(Colombia|Mexico|Afghanistan|Iraq|Iran|Syria|Libya|Somalia)\b',
                r'\b(United States|Russia|China|North Korea|Venezuela)\b',
                r'\b(Pakistan|India|Israel|Egypt|Saudi Arabia)\b',
            ],
            
            'organizations': [
                r'\b(FARC|ELN|Taliban|Al-Qaeda|ISIS|Hezbollah)\b',
                r'\b(Cartel|Mafia|Syndicate)\b',
                r'\b(Revolutionary Armed Forces|National Liberation Army)\b',
            ],
            
            'topics': [
                r'\b(counterinsurgency|insurgency|terrorism|counterterrorism)\b',
                r'\b(intelligence|surveillance|reconnaissance)\b',
                r'\b(drug trafficking|narcotics|cocaine|heroin)\b',
                r'\b(security|threat|assessment|analysis)\b',
                r'\b(weapons|arms|ammunition|explosives)\b',
                r'\b(training|operations|missions|deployment)\b',
            ],
            
            'dates': [
                r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',  # MM/DD/YYYY
                r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b',    # YYYY/MM/DD
                r'\b(\d{6}Z)\b',                          # DDHHMMZ
                r'\b(\d{2}\d{4}Z\s+[A-Z]{3}\s+\d{2})\b', # DTG format
                r'\b([A-Z]{3}\s+\d{4})\b',               # JAN 1998
                r'\b(\d{1,2}\s+[A-Z]{3}\s+\d{4})\b',     # 15 JAN 1998
            ]
        }
    
    def extract_from_text(self, text: str, doc_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Extract entities and topics from a single text using pattern matching.
        
        Args:
            text: Text to analyze
            doc_context: Optional document context
        
        Returns:
            Dictionary with extracted entities, dates, and topics
        """
        results = {}
        
        for category, patterns in self.patterns.items():
            matches = []
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    matches.append({
                        'text': match.group(0),
                        'start': match.start(),
                        'end': match.end(),
                        'pattern': pattern
                    })
            results[category] = matches
        
        parsed_dates = []
        for date_match in results['dates']:
            date_str = date_match['text']
            try:
                parsed_date = dateparser.parse(date_str)
                if parsed_date:
                    date_match['parsed'] = parsed_date.isoformat()
                    date_match['year'] = parsed_date.year
                else:
                    date_match['parsed'] = None
            except Exception:
                date_match['parsed'] = None
            parsed_dates.append(date_match)
        
        results['dates'] = parsed_dates
        
        frequencies = {}
        for category, matches in results.items():
            freq_counter = Counter(match['text'].lower() for match in matches)
            frequencies[category] = dict(freq_counter.most_common(10))
        
        stats = {
            'total_entities': sum(len(matches) for matches in results.values()),
            'categories_found': len([cat for cat, matches in results.items() if matches]),
            'text_length': len(text),
            'word_count': len(text.split())
        }
        
        return {
            'entities': results,
            'frequencies': frequencies,
            'statistics': stats
        }
    
    def extract_from_corpus(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Extract entities and topics from the entire corpus.
        
        Args:
            limit: Optional limit on number of pages to process
        
        Returns:
            Aggregated extraction results across all documents
        """
        LOGGER.info("Starting pattern-based entity extraction...")
        
        with get_session() as session:
            query = session.query(Page).join(Document).join(Source).filter(Page.text.isnot(None))
            if limit:
                query = query.limit(limit)
            
            pages = query.all()
            LOGGER.info("Processing %d pages for entity extraction", len(pages))
        
        corpus_entities = defaultdict(lambda: defaultdict(list))
        corpus_frequencies = defaultdict(Counter)
        document_stats = {}
        all_dates = []
        
        with get_session() as session:
            for i, page in enumerate(pages):
                if i % 10 == 0:
                    LOGGER.info("Processed %d/%d pages", i, len(pages))
                
                page = session.merge(page)
                doc = page.document
                doc_context = {
                    'document_id': doc.id,
                    'external_id': doc.external_id,
                    'source': doc.source.name,
                    'title': doc.title
                }
                
                extraction = self.extract_from_text(page.text, doc_context)
                
                corpus_entities[doc.external_id][page.page_no] = extraction
                
                for category, freq_dict in extraction['frequencies'].items():
                    for entity, count in freq_dict.items():
                        corpus_frequencies[category][entity] += count
                
                for date_info in extraction['entities']['dates']:
                    if date_info['parsed']:
                        all_dates.append({
                            'text': date_info['text'],
                            'parsed': date_info['parsed'],
                            'year': date_info.get('year'),
                            'document': doc.external_id,
                            'page': page.page_no,
                            'source': doc.source.name
                        })
                
                if doc.external_id not in document_stats:
                    document_stats[doc.external_id] = {
                        'title': doc.title,
                        'source': doc.source.name,
                        'pages': 0,
                        'total_entities': 0,
                        'total_words': 0,
                        'categories': set()
                    }
                
                document_stats[doc.external_id]['pages'] += 1
                document_stats[doc.external_id]['total_entities'] += extraction['statistics']['total_entities']
                document_stats[doc.external_id]['total_words'] += extraction['statistics']['word_count']
                document_stats[doc.external_id]['categories'].update(
                    cat for cat, matches in extraction['entities'].items() if matches
                )
        
        for doc_id, stats in document_stats.items():
            stats['categories'] = list(stats['categories'])
        
        temporal_analysis = self._analyze_temporal_patterns(all_dates)
        
        cross_doc_analysis = self._analyze_cross_document_patterns(corpus_entities)
        
        summary = {
            'corpus_stats': {
                'total_pages': len(pages),
                'total_documents': len(document_stats),
                'total_dates': len(all_dates),
                'categories_found': list(corpus_frequencies.keys())
            },
            'top_entities': {
                category: dict(counter.most_common(15))
                for category, counter in corpus_frequencies.items()
            },
            'temporal_analysis': temporal_analysis,
            'cross_document_patterns': cross_doc_analysis,
            'document_stats': document_stats,
            'detailed_extractions': dict(corpus_entities)
        }
        
        LOGGER.info("Pattern-based extraction complete: %d pages processed", len(pages))
        
        return summary
    
    def _analyze_temporal_patterns(self, dates: List[Dict]) -> Dict:
        """Analyze temporal patterns in the extracted dates."""
        if not dates:
            return {'min_date': None, 'max_date': None, 'span_years': 0, 'year_distribution': {}}
        
        parsed_dates = []
        years = []
        
        for date_info in dates:
            if date_info['parsed']:
                try:
                    parsed_date = dateparser.parse(date_info['parsed'])
                    if parsed_date:
                        parsed_dates.append(parsed_date)
                        years.append(parsed_date.year)
                except Exception:
                    continue
        
        if not parsed_dates:
            return {'min_date': None, 'max_date': None, 'span_years': 0, 'year_distribution': {}}
        
        min_date = min(parsed_dates)
        max_date = max(parsed_dates)
        span_years = (max_date - min_date).days / 365.25
        
        year_counts = Counter(years)
        
        return {
            'min_date': min_date.isoformat(),
            'max_date': max_date.isoformat(),
            'span_years': round(span_years, 1),
            'total_dates': len(parsed_dates),
            'year_distribution': dict(year_counts.most_common(10))
        }
    
    def _analyze_cross_document_patterns(self, corpus_entities: Dict) -> Dict:
        """Analyze patterns that appear across multiple documents."""
        entity_docs = defaultdict(set)
        
        for doc_id, doc_data in corpus_entities.items():
            for page_no, page_data in doc_data.items():
                for category, entities in page_data['entities'].items():
                    for entity in entities:
                        entity_text = entity['text'].lower()
                        entity_docs[f"{category}:{entity_text}"].add(doc_id)
        
        cross_doc_entities = {
            entity: docs for entity, docs in entity_docs.items()
            if len(docs) > 1
        }
        
        cross_doc_by_category = defaultdict(list)
        for entity, docs in cross_doc_entities.items():
            category, text = entity.split(':', 1)
            cross_doc_by_category[category].append({
                'entity': text,
                'documents': list(docs),
                'document_count': len(docs)
            })
        
        for category in cross_doc_by_category:
            cross_doc_by_category[category].sort(
                key=lambda x: x['document_count'], reverse=True
            )
        
        return {
            'total_cross_doc_entities': len(cross_doc_entities),
            'by_category': dict(cross_doc_by_category)
        }


def create_simple_entity_extractor() -> SimpleEntityExtractor:
    """Factory function to create a simple entity extractor."""
    return SimpleEntityExtractor()
