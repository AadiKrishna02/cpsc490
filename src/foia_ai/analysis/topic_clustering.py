from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
import networkx as nx

from ..storage.db import get_session
from ..storage.models import Document, Page
from .simple_entity_extraction import create_simple_entity_extractor

LOGGER = logging.getLogger(__name__)


class TopicClusterer:
    """
    Create topic clusters for Wiki-style summary generation.
    
    This builds hierarchical topic structures from:
    - Extracted entities (organizations, locations, operations)
    - Document similarity patterns
    - Cross-document entity relationships
    - Temporal patterns
    """
    
    def __init__(self, entity_results_path: Optional[Path] = None):
        self.entity_results_path = entity_results_path or Path("data/simple_entity_extraction_results.json")
        self.entity_data = None
        self.topic_hierarchy = {}
        self.document_clusters = {}
        
    def load_entity_data(self) -> Dict:
        """Load entity extraction results."""
        if not self.entity_results_path.exists():
            LOGGER.error("Entity extraction results not found. Run entity extraction first.")
            raise FileNotFoundError(f"No entity data at {self.entity_results_path}")
        
        with open(self.entity_results_path, 'r') as f:
            self.entity_data = json.load(f)
        
        LOGGER.info("Loaded entity data: %d documents, %d cross-doc entities", 
                   self.entity_data['corpus_stats']['total_documents'],
                   self.entity_data['cross_document_patterns']['total_cross_doc_entities'])
        
        return self.entity_data
    
    def build_topic_hierarchy(self) -> Dict[str, Any]:
        """
        Build hierarchical topic structure for Wiki organization.
        
        Creates a hierarchy like:
        - Intelligence Operations
          - Counterinsurgency
            - Colombia Operations
              - FARC Analysis
              - ELN Activities
          - Drug Trafficking
            - Cartel Operations
        """
        if not self.entity_data:
            self.load_entity_data()
        
        hierarchy = {
            "Intelligence Operations": {
                "description": "Intelligence assessments, operations, and analysis",
                "keywords": ["intelligence", "assessment", "analysis", "operations"],
                "subtopics": {
                    "Counterinsurgency": {
                        "description": "Counter-insurgency operations and analysis",
                        "keywords": ["counterinsurgency", "insurgency", "guerrilla", "paramilitary"],
                        "entities": ["FARC", "ELN"],
                        "countries": ["Colombia"]
                    },
                    "Drug Trafficking": {
                        "description": "Narcotics trafficking and related operations",
                        "keywords": ["drug trafficking", "narcotics", "cartel", "cocaine"],
                        "entities": ["cartel"],
                        "countries": ["Colombia", "Mexico"]
                    },
                    "Security Threats": {
                        "description": "Security assessments and threat analysis",
                        "keywords": ["security", "threat", "assessment", "surveillance"],
                        "entities": [],
                        "countries": []
                    }
                }
            },
            "Organizations": {
                "description": "Key organizations mentioned in intelligence documents",
                "keywords": [],
                "subtopics": {
                    "Government Agencies": {
                        "description": "US and foreign government agencies",
                        "keywords": ["agency", "government", "department"],
                        "entities": ["DIA", "CIA", "FBI", "DOD", "State"],
                        "countries": ["United States"]
                    },
                    "Insurgent Groups": {
                        "description": "Insurgent and terrorist organizations",
                        "keywords": ["insurgent", "terrorist", "revolutionary"],
                        "entities": ["FARC", "ELN", "Taliban", "Al-Qaeda"],
                        "countries": ["Colombia", "Afghanistan"]
                    },
                    "Criminal Organizations": {
                        "description": "Drug cartels and criminal syndicates",
                        "keywords": ["cartel", "criminal", "trafficking"],
                        "entities": ["cartel"],
                        "countries": ["Colombia", "Mexico"]
                    }
                }
            },
            "Geographic Regions": {
                "description": "Regional intelligence and operations by location",
                "keywords": [],
                "subtopics": {
                    "Latin America": {
                        "description": "Intelligence operations in Latin America",
                        "keywords": ["latin america", "south america"],
                        "entities": ["FARC", "ELN"],
                        "countries": ["Colombia", "Mexico", "Venezuela"]
                    },
                    "Middle East": {
                        "description": "Middle East intelligence operations",
                        "keywords": ["middle east"],
                        "entities": ["Al-Qaeda", "Hezbollah"],
                        "countries": ["Iraq", "Iran", "Syria"]
                    }
                }
            }
        }
        
        self._map_documents_to_topics(hierarchy)
        
        self.topic_hierarchy = hierarchy
        return hierarchy
    
    def _map_documents_to_topics(self, hierarchy: Dict) -> None:
        """Map documents to topics based on entity and keyword matches."""
        top_entities = self.entity_data['top_entities']
        document_stats = self.entity_data['document_stats']
        
        for doc_id, doc_data in document_stats.items():
            doc_topic_scores = {}
            
            doc_extractions = self.entity_data['detailed_extractions'].get(doc_id, {})
            doc_entities = set()
            doc_text_lower = ""
            
            for page_data in doc_extractions.values():
                for category, entities in page_data['entities'].items():
                    for entity in entities:
                        doc_entities.add(entity['text'].lower())
                        doc_text_lower += " " + entity['text'].lower()
            
            for main_topic, main_data in hierarchy.items():
                for subtopic, sub_data in main_data.get('subtopics', {}).items():
                    score = 0
                    
                    for keyword in sub_data.get('keywords', []):
                        if keyword in doc_text_lower:
                            score += 2
                    
                    for entity in sub_data.get('entities', []):
                        if entity.lower() in doc_entities:
                            score += 3
                    
                    for country in sub_data.get('countries', []):
                        if country.lower() in doc_entities:
                            score += 1
                    
                    if score > 0:
                        topic_key = f"{main_topic} > {subtopic}"
                        doc_topic_scores[topic_key] = score
            
            if doc_topic_scores:
                sorted_topics = sorted(doc_topic_scores.items(), key=lambda x: x[1], reverse=True)
                doc_data['topic_scores'] = dict(sorted_topics[:3])  # Top 3 topics
                doc_data['primary_topic'] = sorted_topics[0][0]
            else:
                doc_data['topic_scores'] = {}
                doc_data['primary_topic'] = "Uncategorized"
    
    def create_document_clusters(self) -> Dict[str, List[str]]:
        """Create document clusters based on topic similarity."""
        if not self.topic_hierarchy:
            self.build_topic_hierarchy()
        
        clusters = defaultdict(list)
        document_stats = self.entity_data['document_stats']
        
        for doc_id, doc_data in document_stats.items():
            primary_topic = doc_data.get('primary_topic', 'Uncategorized')
            clusters[primary_topic].append(doc_id)
        
        self.document_clusters = dict(clusters)
        return self.document_clusters
    
    def get_topic_summary_data(self, topic_path: str) -> Dict[str, Any]:
        """
        Get structured data for LLM summary generation for a specific topic.
        
        Args:
            topic_path: Topic path like "Intelligence Operations > Counterinsurgency"
        
        Returns:
            Structured data including relevant documents, entities, and context
        """
        if not self.document_clusters:
            self.create_document_clusters()
        
        topic_docs = self.document_clusters.get(topic_path, [])
        
        if not topic_docs:
            return {"error": f"No documents found for topic: {topic_path}"}
        
        topic_entities = defaultdict(int)
        topic_dates = []
        topic_pages = []
        document_summaries = []
        
        with get_session() as session:
            for doc_id in topic_docs:
                doc_stats = self.entity_data['document_stats'][doc_id]
                document_summaries.append({
                    'id': doc_id,
                    'title': doc_stats['title'],
                    'source': doc_stats['source'],
                    'pages': doc_stats['pages'],
                    'words': doc_stats['total_words'],
                    'entities': doc_stats['total_entities']
                })
                
                doc_pages = session.query(Page).join(Document).filter(
                    Document.external_id == doc_id
                ).all()
                
                for page in doc_pages:
                    if page.text and len(page.text.strip()) > 100:  # Only substantial pages
                        topic_pages.append({
                            'document_id': doc_id,
                            'page_no': page.page_no,
                            'text': page.text,
                            'word_count': len(page.text.split())
                        })
                
                doc_extractions = self.entity_data['detailed_extractions'].get(doc_id, {})
                for page_data in doc_extractions.values():
                    for category, entities in page_data['entities'].items():
                        for entity in entities:
                            topic_entities[f"{category}:{entity['text']}"] += 1
        
        topic_pages.sort(key=lambda x: x['word_count'], reverse=True)
        
        top_entities = dict(Counter(topic_entities).most_common(20))
        
        return {
            'topic': topic_path,
            'document_count': len(topic_docs),
            'total_pages': len(topic_pages),
            'documents': document_summaries,
            'top_entities': top_entities,
            'pages': topic_pages[:10],  # Top 10 most substantial pages
            'all_pages': topic_pages,   # For LLM context if needed
            'metadata': {
                'generated_at': str(Path().cwd()),
                'entity_extraction_file': str(self.entity_results_path)
            }
        }
    
    def get_all_topics(self) -> List[str]:
        """Get list of all available topics."""
        if not self.document_clusters:
            self.create_document_clusters()
        
        return list(self.document_clusters.keys())
    
    def save_topic_structure(self, output_path: Path) -> None:
        """Save topic hierarchy and clusters to file."""
        if not self.topic_hierarchy:
            self.build_topic_hierarchy()
        
        if not self.document_clusters:
            self.create_document_clusters()
        
        output_data = {
            'topic_hierarchy': self.topic_hierarchy,
            'document_clusters': self.document_clusters,
            'available_topics': self.get_all_topics(),
            'metadata': {
                'total_topics': len(self.document_clusters),
                'total_documents': sum(len(docs) for docs in self.document_clusters.values()),
                'entity_source': str(self.entity_results_path)
            }
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        LOGGER.info("Topic structure saved to: %s", output_path)


def create_topic_clusterer() -> TopicClusterer:
    """Factory function to create a topic clusterer."""
    return TopicClusterer()
