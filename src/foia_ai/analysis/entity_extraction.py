from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict, Counter

import dateparser
import spacy
from spacy import displacy

from ..storage.db import get_session
from ..storage.models import Document, Page

LOGGER = logging.getLogger(__name__)


class EntityExtractor:
    """
    Extract entities, dates, and topics from FOIA documents.
    
    Focuses on intelligence-specific entities:
    - Organizations (agencies, military units, terrorist groups)
    - People (officials, operatives, subjects)
    - Locations (countries, cities, facilities)
    - Dates (operations, reports, events)
    - Operations/Codenames
    - Classifications and document types
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self.nlp = None
        
        self.classification_patterns = [
            r'\b(SECRET|TOP SECRET|CONFIDENTIAL|UNCLASSIFIED|CLASSIFIED)\b',
            r'\b(NOFORN|FOUO|FOR OFFICIAL USE ONLY)\b',
            r'\(b\)\([0-9]+\)',  # Redaction codes like (b)(3)
            r'\b(USC \d+)\b',    # US Code references
        ]
        
        self.agency_patterns = [
            r'\b(CIA|DIA|FBI|NSA|DEA|ATF|DOD|DOJ|STATE|USAID)\b',
            r'\b(Defense Intelligence Agency|Central Intelligence Agency)\b',
            r'\b(Federal Bureau of Investigation|National Security Agency)\b',
            r'\b(Department of Defense|Department of Justice)\b',
        ]
        
        self.military_patterns = [
            r'\b(USCINCSO|SOUTHCOM|CENTCOM|EUCOM|PACOM)\b',
            r'\b(\d+(?:st|nd|rd|th)?\s+(?:Infantry|Airborne|Armored|Marine))\b',
            r'\b(Special Forces|Delta Force|Navy SEALs?)\b',
        ]
        
        self.operation_patterns = [
            r'\b(Operation\s+[A-Z][a-zA-Z\s]+)\b',
            r'\b(Plan\s+[A-Z][a-zA-Z\s]+)\b',
            r'\b([A-Z]{2,}\s+(?:OPERATION|PLAN|MISSION))\b',
        ]
    
    def _load_model(self):
        """Load spaCy model with error handling."""
        if self.nlp is None:
            try:
                self.nlp = spacy.load(self.model_name)
                LOGGER.info("Loaded spaCy model: %s", self.model_name)
            except OSError:
                LOGGER.error("spaCy model '%s' not found. Install with: python -m spacy download %s", 
                           self.model_name, self.model_name)
                raise
    
    def extract_from_text(self, text: str, doc_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Extract entities and topics from a single text.
        
        Args:
            text: Text to analyze
            doc_context: Optional document context (title, source, etc.)
        
        Returns:
            Dictionary with extracted entities, dates, and topics
        """
        self._load_model()
        
        doc = self.nlp(text)
        
        entities = self._extract_ner_entities(doc)
        
        dates = self._extract_dates(text)
        
        classifications = self._extract_patterns(text, self.classification_patterns, "CLASSIFICATION")
        agencies = self._extract_patterns(text, self.agency_patterns, "AGENCY")
        military = self._extract_patterns(text, self.military_patterns, "MILITARY")
        operations = self._extract_patterns(text, self.operation_patterns, "OPERATION")
        
        topics = self._extract_topics(doc)
        
        result = {
            'entities': entities,
            'dates': dates,
            'classifications': classifications,
            'agencies': agencies,
            'military_units': military,
            'operations': operations,
            'topics': topics,
            'statistics': {
                'total_entities': sum(len(v) for v in entities.values()),
                'total_dates': len(dates),
                'total_topics': len(topics),
                'text_length': len(text),
                'word_count': len(text.split())
            }
        }
        
        return result
    
    def _extract_ner_entities(self, doc) -> Dict[str, List[Dict]]:
        """Extract named entities using spaCy NER."""
        entities = defaultdict(list)
        
        for ent in doc.ents:
            entity_data = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': getattr(ent, 'confidence', 1.0)
            }
            entities[ent.label_].append(entity_data)
        
        return dict(entities)
    
    def _extract_dates(self, text: str) -> List[Dict]:
        """Extract dates using multiple approaches."""
        dates = []
        
        date_patterns = [
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',  
            r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b',    
            r'\b(\d{6}Z)\b',                   
            r'\b(\d{2}\d{4}Z\s+[A-Z]{3}\s+\d{2})\b', 
            r'\b([A-Z]{3}\s+\d{4})\b',               
            r'\b(\d{1,2}\s+[A-Z]{3}\s+\d{4})\b',     
        ]
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_str = match.group(1)
                
                try:
                    parsed_date = dateparser.parse(date_str)
                    if parsed_date:
                        dates.append({
                            'text': date_str,
                            'parsed': parsed_date.isoformat(),
                            'start': match.start(),
                            'end': match.end(),
                            'type': 'extracted'
                        })
                except Exception:
                    dates.append({
                        'text': date_str,
                        'parsed': None,
                        'start': match.start(),
                        'end': match.end(),
                        'type': 'unparsed'
                    })
        
        return dates
    
    def _extract_patterns(self, text: str, patterns: List[str], category: str) -> List[Dict]:
        """Extract entities matching specific regex patterns."""
        results = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                results.append({
                    'text': match.group(0),
                    'category': category,
                    'start': match.start(),
                    'end': match.end(),
                    'pattern': pattern
                })
        
        return results
    
    def _extract_topics(self, doc) -> List[Dict]:
        """Extract topics using noun phrases and key terms."""
        topics = []
        
        noun_phrases = []
        for chunk in doc.noun_chunks:
            if 2 <= len(chunk.text.split()) <= 4:
                if any(token.is_alpha and not token.is_stop for token in chunk):
                    noun_phrases.append({
                        'text': chunk.text.lower(),
                        'start': chunk.start_char,
                        'end': chunk.end_char,
                        'type': 'noun_phrase'
                    })
        
        intel_keywords = [
            'counterinsurgency', 'intelligence', 'surveillance', 'reconnaissance',
            'terrorism', 'insurgency', 'guerrilla', 'paramilitary',
            'drug trafficking', 'narcotics', 'cartel', 'smuggling',
            'security', 'threat', 'assessment', 'analysis',
            'operation', 'mission', 'deployment', 'strategy',
            'classified', 'secret', 'confidential', 'sensitive'
        ]
        
        text_lower = doc.text.lower()
        for keyword in intel_keywords:
            if keyword in text_lower:
                start = 0
                while True:
                    pos = text_lower.find(keyword, start)
                    if pos == -1:
                        break
                    topics.append({
                        'text': keyword,
                        'start': pos,
                        'end': pos + len(keyword),
                        'type': 'keyword'
                    })
                    start = pos + 1
        
        all_topics = noun_phrases + topics
        
        topic_counts = Counter(t['text'] for t in all_topics)
        
        return [
            {
                'text': topic,
                'frequency': count,
                'type': 'topic'
            }
            for topic, count in topic_counts.most_common(20)
        ]
    
    def extract_from_corpus(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Extract entities and topics from the entire corpus.
        
        Args:
            limit: Optional limit on number of pages to process
        
        Returns:
            Aggregated extraction results across all documents
        """
        LOGGER.info("Starting corpus-wide entity extraction...")
        
        with get_session() as session:
            query = session.query(Page).filter(Page.text.isnot(None))
            if limit:
                query = query.limit(limit)
            
            pages = query.all()
            LOGGER.info("Processing %d pages for entity extraction", len(pages))
        
        corpus_entities = defaultdict(lambda: defaultdict(list))
        corpus_dates = []
        corpus_topics = defaultdict(int)
        corpus_agencies = defaultdict(int)
        corpus_operations = defaultdict(int)
        document_stats = {}
        
        for i, page in enumerate(pages):
            if i % 10 == 0:
                LOGGER.info("Processed %d/%d pages", i, len(pages))
            
            doc = page.document
            doc_context = {
                'document_id': doc.id,
                'external_id': doc.external_id,
                'source': doc.source.name,
                'title': doc.title
            }
            
            extraction = self.extract_from_text(page.text, doc_context)
            
            doc_key = f"{doc.external_id}_page_{page.page_no}"
            
            corpus_entities[doc.external_id][page.page_no] = extraction
            
            for date_info in extraction['dates']:
                date_info['document'] = doc.external_id
                date_info['page'] = page.page_no
                corpus_dates.append(date_info)
            
            for topic in extraction['topics']:
                corpus_topics[topic['text']] += topic['frequency']
            
            for agency in extraction['agencies']:
                corpus_agencies[agency['text']] += 1
            
            for op in extraction['operations']:
                corpus_operations[op['text']] += 1
            
            if doc.external_id not in document_stats:
                document_stats[doc.external_id] = {
                    'title': doc.title,
                    'source': doc.source.name,
                    'pages': 0,
                    'total_entities': 0,
                    'total_words': 0
                }
            
            document_stats[doc.external_id]['pages'] += 1
            document_stats[doc.external_id]['total_entities'] += extraction['statistics']['total_entities']
            document_stats[doc.external_id]['total_words'] += extraction['statistics']['word_count']

        summary = {
            'corpus_stats': {
                'total_pages': len(pages),
                'total_documents': len(document_stats),
                'total_dates': len(corpus_dates),
                'unique_topics': len(corpus_topics),
                'unique_agencies': len(corpus_agencies),
                'unique_operations': len(corpus_operations)
            },
            'top_topics': dict(Counter(corpus_topics).most_common(20)),
            'top_agencies': dict(Counter(corpus_agencies).most_common(10)),
            'top_operations': dict(Counter(corpus_operations).most_common(10)),
            'date_range': self._analyze_date_range(corpus_dates),
            'document_stats': document_stats,
            'detailed_extractions': dict(corpus_entities)
        }
        
        LOGGER.info("Corpus extraction complete: %d pages, %d unique topics", 
                   len(pages), len(corpus_topics))
        
        return summary
    
    def _analyze_date_range(self, dates: List[Dict]) -> Dict:
        parsed_dates = []
        
        for date_info in dates:
            if date_info['parsed']:
                try:
                    parsed_dates.append(dateparser.parse(date_info['parsed']))
                except Exception:
                    continue
        
        if not parsed_dates:
            return {'min_date': None, 'max_date': None, 'span_years': 0}
        
        min_date = min(parsed_dates)
        max_date = max(parsed_dates)
        span_years = (max_date - min_date).days / 365.25
        
        return {
            'min_date': min_date.isoformat(),
            'max_date': max_date.isoformat(),
            'span_years': round(span_years, 1),
            'total_dates': len(parsed_dates)
        }


def create_entity_extractor() -> EntityExtractor:
    """Factory function to create an entity extractor."""
    return EntityExtractor()
