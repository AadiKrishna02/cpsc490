#!/usr/bin/env python3
"""Civis Wiki Web UI

Run:
  OPENAI_API_KEY=... python scripts/wiki_web.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List
import html

def safe_html(text):
    """Escape HTML special characters to prevent injection"""
    if text is None:
        return ""
    return html.escape(str(text))

from flask import Flask, render_template_string, request, redirect, url_for, flash, jsonify
import threading
import time
import markdown as md
from urllib.parse import quote_plus

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from foia_ai.synthesis.wiki_service import (
    load_entity_data,
    create_topic_clusters,
    list_generated_pages,
    generate_topic,
)
from foia_ai.synthesis.citation_validator import validate_file
from foia_ai.storage.db import get_session
from foia_ai.storage.models import Document, Page, Source
from sqlalchemy import or_

try:
    from search_integration import search_documents, get_search_manager
    HYBRID_SEARCH_AVAILABLE = True
except ImportError:
    HYBRID_SEARCH_AVAILABLE = False
    print("Hybrid search not available - using basic search")

try:
    from context_aware_wiki_generator import ContextAwareWikiGenerator
    wiki_generator = None
    CONTEXT_AWARE_GENERATION = True
except ImportError:
    CONTEXT_AWARE_GENERATION = False
    print("Context-aware generation not available - using basic generation")

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = "foia-ai-wiki-ui"

generation_status = {}
generation_lock = threading.Lock()

_index_cache = {
    'db_stats': None,
    'db_stats_time': None,
    'topics': None,
    'topics_time': None,
    'pages': None,
    'pages_time': None
}
_cache_lock = threading.Lock()
CACHE_TTL = 0

def clear_index_cache():
    """Clear the index page cache"""
    with _cache_lock:
        _index_cache['pages'] = None
        _index_cache['pages_time'] = None

@app.after_request
def add_header(response):
    """Disable browser caching"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["X-Content-Type-Options"] = "nosniff"
    import time
    response.headers["X-Version"] = str(int(time.time()))
    return response


    

BASE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Civis - Intelligence Wiki</title>
  <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">
  <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
  <link href=\"https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:wght@400;500;600;700;800&display=swap\" rel=\"stylesheet\">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg: #FAF9F7;          
      --bg-dark: #F5F2ED;     
      --text: #3E2B22;        
      --text-light: #6D5B52;  
      --muted: #9C8C84;       
      
      --border: #D6CDB8;      
      --border-light: #E5E0D0;
      
      --card: #FDFBF7;        
      --surface: #F8F4EB;     
      
      --accent: #D98C21;      
      --accent-hover: #B87318;
      --accent-light: #FCEFDC;
      
      --secondary: #8B5E3C;   
      
      --success: #5F7144;     
      --error: #A64444;       
      --warning: #D98C21;     
      
      --shadow-xs: 0 1px 2px 0 rgba(62, 43, 34, 0.05);
      --shadow-sm: 0 1px 3px 0 rgba(62, 43, 34, 0.1);
      --shadow-md: 0 4px 12px -2px rgba(62, 43, 34, 0.08);
      --shadow-lg: 0 10px 24px -4px rgba(62, 43, 34, 0.12);
      --shadow-xl: 0 20px 40px -8px rgba(62, 43, 34, 0.12);
      
      --radius-sm: 6px;
      --radius-md: 10px;
      --radius-lg: 14px;
      --radius-xl: 18px;
    }
    
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.65;
      -webkit-font-smoothing: antialiased;
    }
    
    h1, h2, h3, h4, h5, h6 {
      font-family: 'Playfair Display', serif;
      color: var(--text);
    }
    
    a { color: var(--accent); text-decoration: none; transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1); }
    a:hover { color: var(--accent-hover); }
    
    .page-wrapper {
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    
    main {
      flex: 1;
    }
    
    .container {
      max-width: 1280px;
      margin: 0 auto;
      padding: 0 32px;
    }
    
    @media (max-width: 768px) {
      .container {
        padding: 0 20px;
      }
    }
    
    .header {
      background: rgba(239, 235, 224, 0.95); 
      backdrop-filter: saturate(180%) blur(20px);
      border-bottom: 1px solid var(--border);
      padding: 16px 0;
      position: sticky;
      top: 0;
      z-index: 1000;
      box-shadow: var(--shadow-sm);
    }
    
    .header-content {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 48px;
    }
    
    .brand {
      display: flex;
      align-items: center;
      gap: 16px;
      cursor: pointer;
      text-decoration: none;
    }
    
    .brand img.logo-img {
      height: 48px;
      width: auto;
      display: block;
    }
    
    .brand .logo-fallback {
      width: 42px;
      height: 42px;
      border-radius: 50%;
      background: var(--text);
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: 'Playfair Display', serif;
      font-weight: 700;
      color: var(--accent);
      font-size: 20px;
      border: 2px solid var(--accent);
    }
    
    .brand h1 {
      font-size: 28px;
      font-weight: 700;
      color: var(--text);
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }
    
    .nav {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    
    .nav-link {
      padding: 8px 18px;
      border-radius: var(--radius-md);
      font-weight: 500;
      font-size: 15px;
      color: var(--text-light);
      transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
      font-family: 'Inter', sans-serif;
    }
    
    .nav-link:hover {
      background: rgba(62, 43, 34, 0.05);
      color: var(--text);
    }
    
    .nav-link.active {
      background: var(--text);
      color: var(--bg);
      box-shadow: var(--shadow-md);
    }

    .card-horizontal {
      display: flex;
      flex-direction: row;
      background: white;
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      overflow: hidden;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      box-shadow: var(--shadow-sm);
      margin-bottom: 24px;
      text-decoration: none;
      color: inherit;
    }

    .card-horizontal:hover {
      box-shadow: var(--shadow-md);
      border-color: var(--accent);
      transform: translateY(-2px);
    }

    .card-horizontal .content {
      flex: 1;
      padding: 24px;
      display: flex;
      flex-direction: column;
      justify-content: center;
    }

    .card-horizontal .actions {
      padding: 24px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: var(--surface);
      border-left: 1px solid var(--border-light);
    }

    .card-horizontal h3 {
      font-family: 'Playfair Display', serif;
      font-size: 24px;
      font-weight: 700;
      color: var(--text);
      margin-bottom: 8px;
    }

    .card-horizontal .meta {
      font-size: 13px;
      color: var(--text-light);
      margin-bottom: 12px;
      font-family: 'Inter', sans-serif;
    }

    .card-horizontal .excerpt {
      font-size: 15px;
      color: var(--text-light);
      line-height: 1.6;
      font-family: 'Inter', sans-serif;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }

    @media (max-width: 768px) {
      .card-horizontal {
        flex-direction: column;
      }
      .card-horizontal .actions {
        width: 100%;
        border-left: none;
        border-top: 1px solid var(--border-light);
        padding: 16px;
      }
    }
    
    .hero {
      padding: 140px 0 100px 0; 
      text-align: center;
      background: transparent !important;
      box-shadow: none !important;
      border: none !important;
      position: relative;
      overflow: hidden;
    }
    
    .hero::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: radial-gradient(circle at 50% 50%, rgba(217, 140, 33, 0.08) 0%, transparent 70%);
      pointer-events: none;
    }
    
    .hero > * {
      position: relative;
      z-index: 1;
    }
    
    .hero h1 {
      font-family: 'Playfair Display', serif;
      font-size: 64px;
      font-weight: 800;
      color: var(--text);
      margin-bottom: 24px;
      letter-spacing: -0.01em;
      line-height: 1.1;
    }
    
    .hero .subtitle {
      font-family: 'Inter', sans-serif;
      font-size: 20px;
      color: var(--text-light);
      margin-bottom: 48px;
      max-width: 720px;
      margin-left: auto;
      margin-right: auto;
      font-weight: 400;
      line-height: 1.6;
    }
    
    .search-hero {
      max-width: 720px;
      margin: 0 auto 48px auto;
      position: relative;
      display: flex;
      flex-direction: column;
      align-items: flex-start; 
      gap: 16px;
    }
    
    .search-input-wrapper {
      width: 100%;
      position: relative;
    }
    
    .search-input {
      width: 100%;
      padding: 22px 160px 22px 36px; 
      font-size: 18px;
      border: 2px solid var(--border);
      border-radius: 99px;
      background: white;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
      font-family: 'Inter', sans-serif;
      color: var(--text);
    }
    
    .search-input:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 4px 24px rgba(217, 140, 33, 0.15);
      transform: translateY(-1px);
    }
    
    .search-btn {
      position: absolute;
      right: 8px;
      top: 50%;
      transform: translateY(-50%);
      background: var(--text);
      color: var(--bg);
      border: none;
      padding: 14px 32px;
      border-radius: 99px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
      font-size: 15px;
      font-family: 'Inter', sans-serif;
      letter-spacing: 0.01em;
    }
    
    .search-btn:hover {
      background: var(--accent);
      color: white;
      transform: translateY(-50%) scale(1.02);
      box-shadow: 0 4px 12px rgba(217, 140, 33, 0.3);
    }
    
    .search-mode-toggle {
      display: inline-flex;
      align-items: center;
      gap: 4px;             
      background: white;
      padding: 4px;         
      border-radius: 8px;
      border: 1px solid var(--border);
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
      margin-top: 12px;
    }

    .toggle-option {
      padding: 6px 12px;    
      font-size: 13px;
      font-weight: 500;
      color: var(--text-light);
      cursor: pointer;
      border-radius: 6px;
      transition: all 0.2s ease;
      user-select: none;
      display: flex;
      align-items: center;
      overflow: hidden;
      white-space: nowrap;
    }

    .info-icon {
      width: 18px;
      height: 18px;
      border-radius: 50%;
      border: 1.5px solid var(--border);
      background: transparent;
      color: var(--text);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 11px;
      font-weight: 600;
      cursor: help;
      transition: all 0.2s ease;
      flex-shrink: 0;
    }

    
    .toggle-option input[type="radio"] {
      display: none;
    }
    
    .toggle-option:has(input:checked) {
      background: var(--text);
      color: white;
      font-weight: 600;
      box-shadow: none;
    }
    
    .toggle-option:hover:not(:has(input:checked)) {
      color: var(--text);
      background: var(--surface);
    }
    
    .info-icon:hover {
      border-color: var(--text);
      background: var(--surface);
      color: var(--text);
    }
    
    .info-icon .tooltip {
      visibility: hidden;
      opacity: 0;
      position: absolute;
      bottom: calc(100% + 12px);
      left: 50%;
      transform: translateX(-50%);
      width: 300px;
      background: var(--text);
      color: white;
      padding: 14px 16px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 400;
      line-height: 1.6;
      z-index: 1000;
      box-shadow: 0 8px 24px rgba(0,0,0,0.25);
      text-align: left;
      transition: all 0.2s ease;
      pointer-events: none;
      font-family: 'Inter', sans-serif;
    }
    
    .info-icon .tooltip::after {
      content: '';
      position: absolute;
      top: 100%;
      left: 50%;
      transform: translateX(-50%);
      border-width: 6px;
      border-style: solid;
      border-color: var(--text) transparent transparent transparent;
    }
    
    .info-icon:hover .tooltip {
      visibility: visible;
      opacity: 1;
    }
    
    @media (max-width: 768px) {
      .search-hero {
        align-items: center; 
      }
      
      .search-input {
        padding: 18px 120px 18px 24px;
        font-size: 16px;
      }
      
      .search-btn {
        padding: 10px 20px;
        font-size: 14px;
      }
    }
    
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 32px;
      margin-bottom: 28px;
      box-shadow: var(--shadow-sm);
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .card:hover {
      box-shadow: var(--shadow-md);
      border-color: var(--accent);
      transform: translateY(-2px);
      background: white;
    }
    
    .card h2 {
      font-family: 'Playfair Display', serif;
      font-size: 28px;
      font-weight: 700;
      color: var(--text);
      margin-bottom: 16px;
      border-bottom: 2px solid var(--accent-light);
      padding-bottom: 12px;
      display: inline-block;
    }
    
    .card h3 {
      font-family: 'Playfair Display', serif;
      font-size: 22px;
      font-weight: 600;
      color: var(--text);
      margin-bottom: 14px;
    }
    
    .card .card-title {
      font-family: 'Playfair Display', serif;
      font-size: 20px;
      font-weight: 700;
      color: var(--text);
      margin-bottom: 8px;
    }
    
    .card-hover {
      cursor: pointer;
    }
    
    .card-hover:hover {
      border-color: var(--accent);
      box-shadow: var(--shadow-lg);
    }
    
    /* Buttons */
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      background: var(--text);
      color: var(--bg);
      padding: 12px 24px;
      border-radius: var(--radius-md);
      border: none;
      cursor: pointer;
      font-weight: 600;
      font-size: 15px;
      transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
      text-decoration: none;
      box-shadow: var(--shadow-sm);
      position: relative;
      font-family: 'Inter', sans-serif;
    }
    
    .btn:hover {
      background: var(--accent);
      color: white;
      box-shadow: var(--shadow-md);
      transform: translateY(-2px);
    }
    
    .btn:active {
      transform: translateY(0px);
    }
    
    .btn.secondary {
      background: white;
      color: var(--text);
      border: 1px solid var(--border);
      box-shadow: var(--shadow-xs);
    }
    
    .btn.secondary:hover {
      background: var(--surface);
      border-color: var(--text);
      color: var(--text);
      box-shadow: var(--shadow-sm);
    }
    
    .btn.large {
      padding: 16px 36px;
      font-size: 17px;
      border-radius: 99px; 
    }
    
    .btn.small {
      padding: 8px 16px;
      font-size: 14px;
      border-radius: var(--radius-sm);
    }
    
    .topics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      gap: 24px;
      margin: 40px 0;
    }
    
    .topic-icon {
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 28px 20px;
      background: white;
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      text-decoration: none;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      box-shadow: var(--shadow-xs);
      position: relative;
      overflow: hidden;
    }
    
    .topic-icon::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 4px;
      background: var(--accent);
      transform: scaleX(0);
      transition: transform 0.3s ease;
    }
    
    .topic-icon:hover::before {
      transform: scaleX(1);
    }
    
    .topic-icon:hover {
      border-color: var(--accent);
      box-shadow: var(--shadow-md);
      transform: translateY(-4px);
    }
    
    .topic-icon .icon {
      width: 52px;
      height: 52px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      margin-bottom: 16px;
      font-size: 22px;
      font-weight: 700;
      color: var(--accent);
    }
    
    .topic-icon .title {
      font-size: 15px;
      font-weight: 600;
      color: var(--text);
      text-align: center;
      line-height: 1.4;
    }
    
    .topic-icon .count {
      font-size: 12px;
      color: var(--muted);
      margin-top: 6px;
      font-weight: 500;
    }
    
    /* Content sections */
    .section {
      padding: 80px 0;
    }
    
    .section-title {
      font-family: 'Playfair Display', serif;
      font-size: 42px;
      font-weight: 700;
      color: var(--text);
      margin-bottom: 24px;
      text-align: center;
      letter-spacing: -0.01em;
    }
    
    .section-subtitle {
      font-family: 'Inter', sans-serif;
      font-size: 18px;
      color: var(--text-light);
      text-align: center;
      margin-top: -16px;
      margin-bottom: 48px;
      font-weight: 400;
      max-width: 600px;
      margin-left: auto;
      margin-right: auto;
    }
    
    .muted {
      color: var(--muted);
      font-size: 15px;
      font-weight: 450;
    }
    
    .tag {
      display: inline-block;
      padding: 6px 14px;
      background: var(--surface);
      border-radius: 999px;
      font-size: 13px;
      font-weight: 600;
      color: var(--text);
      border: 1px solid var(--border);
      font-family: 'Inter', sans-serif;
    }
    
    /* Quick actions */
    .quick-actions {
      display: flex;
      gap: 20px;
      justify-content: center;
      flex-wrap: wrap;
    }
    
    .content {
      line-height: 1.9;
      font-size: 18px;
      color: var(--text-light);
      font-family: 'Inter', sans-serif;
    }
    
    .content h1 {
      font-family: 'Playfair Display', serif;
      font-size: 38px;
      font-weight: 800;
      color: var(--text);
      margin-top: 56px;
      margin-bottom: 24px;
      letter-spacing: -0.02em;
      line-height: 1.2;
    }
    
    .content h2 {
      font-family: 'Playfair Display', serif;
      font-size: 30px;
      font-weight: 700;
      color: var(--text);
      margin-top: 48px;
      margin-bottom: 20px;
      letter-spacing: -0.01em;
      border-bottom: 1px solid var(--border);
      padding-bottom: 12px;
    }
    
    .content h3 {
      font-family: 'Playfair Display', serif;
      font-size: 24px;
      font-weight: 600;
      color: var(--text);
      margin-top: 36px;
      margin-bottom: 16px;
    }
    
    .content p {
      margin-bottom: 24px;
    }
    
    .content ul, .content ol {
      margin-left: 24px;
      margin-bottom: 24px;
    }
    
    .content li {
      margin-bottom: 12px;
    }
    
    .content code {
      background: var(--surface);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 15px;
      border: 1px solid var(--border);
      font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
      color: var(--text);
      font-weight: 500;
    }
    
    .content strong {
      font-weight: 600;
      color: var(--text);
    }
    
    /* Citation styling */
    .content a.citation {
      color: var(--accent);
      text-decoration: none;
      font-size: 0.9em;
      font-weight: 500;
      transition: all 0.2s ease;
      border-bottom: 1px dotted var(--accent);
    }
    
    .content a.citation:hover {
      color: var(--accent-hover);
      border-bottom: 1px solid var(--accent-hover);
    }
    
    /* Reference link styling */
    .content a.reference-link {
      color: var(--accent);
      text-decoration: none;
      font-weight: 600;
      transition: all 0.2s ease;
      border-bottom: 1px solid var(--accent);
    }
    
    .content a.reference-link:hover {
      color: var(--accent-hover);
      border-bottom: 2px solid var(--accent-hover);
    }
    
    .content pre {
      background: var(--surface);
      padding: 16px;
      border-radius: 10px;
      overflow-x: auto;
      border: 1px solid var(--border);
    }
    
    .flash {
      background: rgba(95, 113, 68, 0.1);
      border: 1px solid rgba(95, 113, 68, 0.3);
      padding: 16px 24px;
      color: var(--success);
      border-radius: var(--radius-md);
      margin-bottom: 24px;
      font-weight: 600;
      box-shadow: var(--shadow-sm);
      display: flex;
      align-items: center;
      gap: 12px;
      font-family: 'Inter', sans-serif;
    }
    
    .flash::before {
      content: '';
      display: none;
    }
    
    .error {
      background: rgba(166, 68, 68, 0.1);
      border: 1px solid rgba(166, 68, 68, 0.3);
      padding: 16px 24px;
      color: var(--error);
      border-radius: var(--radius-md);
      margin-bottom: 24px;
      font-weight: 600;
      box-shadow: var(--shadow-sm);
      display: flex;
      align-items: center;
      gap: 12px;
      font-family: 'Inter', sans-serif;
    }
    
      .error::before {
        content: '';
        display: none;
      }
      
      /* Search fallback warning banner */
      .search-fallback-warning {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        color: white;
        padding: 16px 24px;
        border-bottom: 3px solid #e65100;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
        position: sticky;
        top: 0;
        z-index: 999;
        font-weight: 600;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        font-family: 'Inter', sans-serif;
        font-size: 15px;
        text-align: center;
      }
      
      .search-fallback-warning strong {
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }
      
      .search-fallback-warning::before {
        content: 'WARNING:';
        font-size: 20px;
      }
      
      @media (max-width: 768px) {
        .search-fallback-warning {
          font-size: 14px;
          padding: 12px 16px;
          flex-direction: column;
          gap: 8px;
        }
    }
    
    .issue {
      background: var(--surface);
      border: 1px solid var(--border);
      padding: 12px;
      border-radius: 10px;
      margin: 10px 0;
    }
    
    form .row {
      display: flex;
      gap: 20px;
      flex-wrap: wrap;
    }
    
    form label {
      display: block;
      font-weight: 600;
      font-size: 14px;
      color: var(--text);
      margin-bottom: 8px;
    }
    
    form input, form select, form textarea {
      background: white;
      border: 1.5px solid var(--border);
      color: var(--text);
      padding: 12px 16px;
      border-radius: var(--radius-md);
      font-size: 15px;
      transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
      font-family: inherit;
      font-weight: 450;
    }
    
    form input:hover, form select:hover, form textarea:hover {
      border-color: var(--text-light);
    }
    
    form input:focus, form select:focus, form textarea:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 4px rgba(217, 140, 33, 0.15);
      transform: translateY(-1px);
    }
    
    form input::placeholder, form textarea::placeholder {
      color: var(--muted);
      font-style: italic;
    }
    
    .form-section {
      background: var(--surface);
      padding: 32px;
      border-radius: var(--radius-lg);
      border: 1px solid var(--border);
      margin-bottom: 24px;
    }
    
    .form-section h3 {
      margin-top: 0;
      margin-bottom: 24px;
      font-size: 20px;
      font-weight: 700;
      color: var(--text);
      font-family: 'Playfair Display', serif;
    }
    
    .hero {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
      padding: 32px;
      background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(96, 165, 250, 0.1));
      border: 1px solid rgba(59, 130, 246, 0.2);
      border-radius: 16px;
      margin-bottom: 32px;
      box-shadow: var(--shadow-md);
    }
    
    .hero h2 {
      margin: 0 0 8px 0;
      font-size: 32px;
      font-weight: 800;
      color: var(--text);
    }
    
    .pill {
      display: inline-block;
      padding: 4px 12px;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      border-radius: 999px;
      color: white;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      box-shadow: var(--shadow-sm);
    }
    
    table {
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      border-radius: var(--radius-md);
      overflow: hidden;
      border: 1px solid var(--border);
    }
    
    table th {
      background: var(--surface);
      font-family: 'Playfair Display', serif;
      font-weight: 700;
      text-align: left;
      padding: 16px 20px;
      border-bottom: 2px solid var(--border);
      color: var(--text);
      font-size: 15px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    
    table td {
      padding: 16px 20px;
      border-bottom: 1px solid var(--border-light);
      font-size: 15px;
      font-family: 'Inter', sans-serif;
      color: var(--text-light);
    }
    
    table tbody tr {
      transition: all 0.2s ease;
      background: white;
    }
    
    table tbody tr:hover {
      background: var(--surface);
    }
    
    table tbody tr:last-child td {
      border-bottom: none;
    }
    
    @media (max-width: 1024px) {
      .hero h1 {
        font-size: 44px;
      }
      
      .hero .subtitle {
        font-size: 19px;
      }
    }
    
    @media (max-width: 768px) {
      .hero {
        padding: 60px 0 48px 0;
      }
      
      .hero h1 {
        font-size: 36px;
      }
      
      .hero .subtitle {
        font-size: 17px;
      }
      
      .header {
        padding: 16px 0;
      }
      
      .header-content {
        flex-direction: column;
        gap: 16px;
      }
      
      .nav {
        width: 100%;
        justify-content: center;
        flex-wrap: wrap;
      }
      
      .topics-grid {
        grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
        gap: 14px;
      }
      
      .quick-actions {
        flex-direction: column;
        width: 100%;
      }
      
      .quick-actions .btn {
        width: 100%;
      }
      
      .card {
        padding: 24px 20px;
      }
      
      .section {
        padding: 48px 0;
      }
      
      .section-title {
        font-size: 28px;
      }
      
      .content h1 {
        font-size: 28px;
      }
      
      .content h2 {
        font-size: 22px;
      }
      
      form .row {
        flex-direction: column;
      }
      
      form .row > * {
        width: 100% !important;
      }
      
      table {
        font-size: 14px;
      }
      
      table th, table td {
        padding: 12px 10px;
      }
    }
    
    @media (max-width: 480px) {
      .hero h1 {
        font-size: 30px;
      }
      
      .brand h1 {
        font-size: 18px;
      }
      
      .nav-link {
        font-size: 14px;
        padding: 8px 14px;
      }
    }
  </style>
</head>
<body>
  <div class=\"page-wrapper\">
    <header class=\"header\">
      <div class=\"container\">
        <div class=\"header-content\">
          <a href=\"{{ url_for('index') }}\" class=\"brand\">
            <img src=\"{{ url_for('static', filename='FOIA logo.png') }}\" alt=\"Civis Logo\" class=\"logo-img\">
            <h1>CIVIS</h1>
          </a>
          <nav class=\"nav\">
            <a class=\"nav-link\" href=\"{{ url_for('wiki_hub') }}\">Wiki Hub</a>
            <a class=\"nav-link\" href=\"{{ url_for('community_wikis') }}\">Community Wikis</a>
            <a class=\"nav-link\" href=\"{{ url_for('generate') }}\">Generate</a>
            <a class=\"nav-link\" href=\"{{ url_for('documents') }}\">Documents</a>
          </nav>
        </div>
      </div>
    </header>
    
    <main>
      {% if show_search_warning %}
        <div class=\"search-fallback-warning\">
          <strong>Search System Notice:</strong>
          <span>Production search unavailable - using legacy federated search. Results may be slower.</span>
          {% if search_fallback_reason %}
            <span style=\"font-size: 13px; opacity: 0.9; font-weight: 400;\">({{ search_fallback_reason }})</span>
          {% endif %}
        </div>
      {% endif %}
      {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
          <div class=\"container\">
            {% for category, m in messages %}
              {% if category == 'error' %}
                <div class=\"error\">{{ m }}</div>
              {% else %}
                <div class=\"flash\">{{ m }}</div>
              {% endif %}
            {% endfor %}
          </div>
        {% endif %}
      {% endwith %}
      {{ body|safe }}
    </main>
  </div>
</body>
</html>
"""



def render(body_html: str):
    """Render page with optional search warning banner"""
    show_search_warning = False
    search_fallback_reason = None
    
    if HYBRID_SEARCH_AVAILABLE:
        try:
            manager = get_search_manager()
            if not manager.using_production_search:
                show_search_warning = True
                search_fallback_reason = manager.fallback_reason
        except Exception:
            pass
    
    return render_template_string(
        BASE_HTML, 
        body=body_html,
        show_search_warning=show_search_warning,
        search_fallback_reason=search_fallback_reason
    )


def get_topic_icon(topic: str) -> str:
    """Return icon for topic based on keywords."""

    return ''
@app.route("/")
def index():
    import time
    from datetime import datetime, timedelta
    
    current_time = time.time()
    
    with _cache_lock:
        if (_index_cache['db_stats'] is None or 
            _index_cache['db_stats_time'] is None or 
            current_time - _index_cache['db_stats_time'] > CACHE_TTL):
            with get_session() as session:
                total_docs = session.query(Document).count()
                total_pages = session.query(Page).count()
            _index_cache['db_stats'] = (total_docs, total_pages)
            _index_cache['db_stats_time'] = current_time
        else:
            total_docs, total_pages = _index_cache['db_stats']
    
    with _cache_lock:
        if (_index_cache['topics'] is None or 
            _index_cache['topics_time'] is None or 
            current_time - _index_cache['topics_time'] > CACHE_TTL):
            topics = []
            try:
                import json
                
                dynamic_topics_file = ROOT / "data" / "dynamic_topics_cache.json"
                if dynamic_topics_file.exists():
                    with open(dynamic_topics_file) as f:
                        cache_data = json.load(f)
                    
                    generated_at = datetime.fromisoformat(cache_data['generated_at'])
                    if datetime.now() - generated_at < timedelta(days=7):
                        topics = [(t['name'], t['count']) for t in cache_data['topics'][:12]]
                        print(f"Loaded {len(topics)} topics from dynamic cache")
            except Exception as e:
                print(f"Dynamic topics unavailable: {e}")
            
            if not topics:
                try:
                    entity = load_entity_data()
                    clusters, defs = create_topic_clusters(entity)
                    topics = [(t, len(docs)) for t, docs in clusters.items()]
                    topics.sort(key=lambda x: -x[1]) 
                    topics = topics[:12] 
                    print(f"Loaded {len(topics)} topics from entity extraction (may be outdated)")
                except Exception as e:
                    print(f"Entity extraction unavailable: {e}")
                    topics = []

            _index_cache['topics'] = topics
            _index_cache['topics_time'] = current_time
        else:
            topics = _index_cache['topics']

    with _cache_lock:
        if (_index_cache['pages'] is None or 
            _index_cache['pages_time'] is None or 
            current_time - _index_cache['pages_time'] > CACHE_TTL):
            all_pages = list_generated_pages()
            pages_with_mtime = [(p, p.stat().st_mtime) for p in all_pages]
            pages_with_mtime.sort(key=lambda x: x[1], reverse=True)
            pages_by_time = [p for p, _ in pages_with_mtime]
            
            _index_cache['pages'] = pages_by_time
            _index_cache['pages_time'] = current_time
        else:
            pages_by_time = _index_cache['pages']
    
    featured_config_path = ROOT / "data" / "wiki" / "featured_pages.json"
    featured_slugs = []
    if featured_config_path.exists():
        try:
            import json
            with open(featured_config_path, 'r') as f:
                config = json.load(f)
                featured_slugs = config.get('featured', [])
        except Exception:
            pass
    
    if featured_slugs:
        pages = [p for p in all_pages if p.stem in featured_slugs]
        pages.sort(key=lambda p: featured_slugs.index(p.stem) if p.stem in featured_slugs else 999)
    else:
        pages = pages_by_time

    body = [
        '<section class="hero">',
        '<div class="container">',
        '<h1>Civis Intelligence</h1>',
        '<p class="subtitle">Generate comprehensive intelligence reports from 6,000+ FOIA documents</p>',
        
        '<form class="search-hero" method="get" action="/search" id="search-form">',
        '<div class="search-input-wrapper">',
        '<input class="search-input" type="text" name="q" placeholder="Search topics, documents, and reports..." id="search-query" />',
        '<button class="search-btn" type="submit" id="search-submit">Search</button>',
        '</div>',
        '<div style="display: flex; justify-content: center; align-items: center; margin-top: 12px; width: 100%;">',
        '<div class="search-mode-toggle" style="position: relative;">',
        '<label class="toggle-option"><input type="radio" name="mode" value="quick" checked> Quick Semantic</label>',
        '<label class="toggle-option"><input type="radio" name="mode" value="keyword"> Quick Keyword</label>',
        '<label class="toggle-option"><input type="radio" name="mode" value="advanced"> Advanced Hybrid</label>',
        '<div class="info-icon" style="position: absolute; right: -30px; top: 50%; transform: translateY(-50%);">i',
        '<div class="tooltip"><strong>Quick Semantic:</strong> Fast concept search using embeddings.<br/><br/><strong>Keyword Search:</strong> Fast phrase matching powered by BM25.<br/><br/><strong>Advanced Hybrid:</strong> Combines semantic + keyword for the most comprehensive results, takes longer to generate.</div>',
        '</div>',
        '</div>',
        '</div>',
        '</form>',
        
        '<section class="section" style="background:white;border-bottom:1px solid var(--border-light);padding:80px 0;">',
        '<div class="container">',
        '<div style="text-align:center;margin-bottom:48px;">',
        '<h2 class="section-title" style="font-family:\'Playfair Display\', serif;font-size:36px;margin-bottom:16px;">Intelligence Landscape</h2>',
        '<p class="section-subtitle" style="font-size:18px;color:var(--text-light);max-width:600px;margin:0 auto;">Explore the breadth of our FOIA collection spanning decades of intelligence history.</p>',
        '</div>',
        '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:24px;">',
    ]

    if topics:
        for topic, count in topics[:12]:
            display_name = topic if len(topic) <= 25 else topic[:22] + "..."
            body.append(
                f'<div style="background:var(--surface);padding:24px;border-radius:12px;border:1px solid var(--border);text-align:center;">'
                f'<div style="font-family:\'Playfair Display\', serif;font-size:20px;font-weight:600;color:var(--text);margin-bottom:8px;">{safe_html(display_name)}</div>'
                f'<div style="font-size:14px;color:var(--text-light);font-weight:500;">{count:,} documents</div>'
                f'</div>'
            )
    else:
        body.append('<div style="grid-column:1/-1;text-align:center;"><div class="muted">Topics loading...</div></div>')
    
    body.extend([
        '</div>',
        '</div>',
        '</section>',
        
        '<section class="section" style="background:var(--text);color:white;padding:80px 0;">',
        '<div class="container">',
        '<div style="display:flex;flex-direction:column;align-items:center;text-align:center;max-width:800px;margin:0 auto;">',
        '<div class="tag" style="background:rgba(255,255,255,0.1);color:white;border:1px solid rgba(255,255,255,0.2);margin-bottom:24px;font-weight:600;letter-spacing:0.5px;padding:6px 16px;font-size:12px;">OFFICIAL WIKIS</div>',
        '<h2 style="font-family:\'Playfair Display\', serif;font-size:48px;font-weight:700;margin-bottom:24px;line-height:1.2;color:white;">The Wiki Hub</h2>',
        '<p style="font-size:20px;color:rgba(255,255,255,0.8);line-height:1.6;margin-bottom:40px;">Curated, deep-dive wikis produced by site analysts. Exhaustive reports covering major social, political, and economics themes with maximum depth and verification.</p>',
        f'<a href="{url_for("wiki_hub")}" class="btn large" style="background:white;color:var(--text);border:none;">Enter Wiki Hub</a>',
        '</div>',
        '</div>',
        '</section>',

        '<section class="section" style="background:var(--surface);padding:80px 0;">',
        '<div class="container">',
        '<div style="display:flex;flex-direction:column;align-items:center;text-align:center;max-width:800px;margin:0 auto;">',
        '<h2 class="section-title" style="font-family:\'Playfair Display\', serif;font-size:36px;margin-bottom:16px;">Community Research</h2>',
        '<p class="section-subtitle" style="font-size:18px;color:var(--text-light);margin-bottom:40px;">Explore specific inquiries from the community or launch your own investigation.</p>',
        '<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;">',
        f'<a href="{url_for("community_wikis")}" class="btn secondary large" style="background:white;">Browse Community Wikis</a>',
        f'<a href="{url_for("generate")}" class="btn large">Generate New Report</a>',
        '</div>',
        '</div>',
        '</div>',
        '</section>',
    ])
    
    return render("".join(body))


@app.route("/generating/<job_id>")
def generating(job_id: str):
    """Loading page while wiki is being generated"""
    with generation_lock:
        status = generation_status.get(job_id, {
            'status': 'unknown',
            'progress': 0,
            'message': 'Initializing...',
            'topic': 'Unknown'
        })
    
    topic = status.get('topic', 'Unknown')
    
    body = f'''
    <div class="container" style="padding-top:60px;">
        <div class="card" style="max-width:800px;margin:0 auto;text-align:center;padding:60px 40px;">
            <div style="margin-bottom:32px;">
                <div class="spinner" style="margin:0 auto 24px auto;"></div>
                <h2 style="margin-bottom:16px;font-size:32px;">Generating Wiki Page</h2>
                <p class="muted" style="font-size:18px;margin-bottom:32px;">"{topic}"</p>
            </div>
            
            <div style="margin-bottom:32px;">
                <div class="progress-bar" style="width:100%;height:8px;background:var(--border-light);border-radius:999px;overflow:hidden;margin-bottom:16px;">
                    <div id="progress-fill" style="width:10%;height:100%;background:linear-gradient(90deg,var(--accent),var(--accent-hover));transition:width 0.5s ease;"></div>
                </div>
                <p id="status-message" class="muted" style="font-size:15px;">Initializing federated search...</p>
            </div>
            
            <div style="background:var(--surface);padding:24px;border-radius:12px;border:1px solid var(--border);">
                <h3 style="font-size:16px;margin-bottom:12px;color:var(--text-light);">What's Happening?</h3>
                <ul style="text-align:left;color:var(--muted);font-size:14px;line-height:2;list-style:none;padding:0;">
                    <li id="step1" style="opacity:0.4;">Searching 6,095 documents across 7 batch indices</li>
                    <li id="step2" style="opacity:0.4;">Ranking chunks and filtering by relevance score (> 0.4)</li>
                    <li id="step3" style="opacity:0.4;">Generating comprehensive wiki article with GPT-5 Nano</li>
                    <li id="step4" style="opacity:0.4;">Formatting citations and references</li>
                    <li id="step5" style="opacity:0.4;">Finalizing and saving wiki page</li>
                </ul>
            </div>
            
            <p class="muted" style="margin-top:24px;font-size:13px;">
                This typically takes 2-5 minutes, depending on page size. Please keep this page open.
            </p>
        </div>
    </div>
    
    <style>
        .spinner {{
            width: 60px;
            height: 60px;
            border: 4px solid var(--border-light);
            border-top: 4px solid var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0.4; }}
            to {{ opacity: 1; }}
        }}
        
        .step-active {{
            opacity: 1 !important;
            font-weight: 600;
            color: var(--accent) !important;
            animation: fadeIn 0.5s ease;
        }}
    </style>
    
    <script>
        const jobId = "{job_id}";
        const statusUrl = "/api/generation-status/" + jobId;
        
        function updateStatus() {{
            fetch(statusUrl)
                .then(response => response.json())
                .then(data => {{
                    const progressFill = document.getElementById('progress-fill');
                    progressFill.style.width = data.progress + '%';
                    
                    const statusMessage = document.getElementById('status-message');
                    statusMessage.textContent = data.message;
                    
                    // Update step indicators
                    const steps = ['step1', 'step2', 'step3', 'step4', 'step5'];
                    const currentStep = Math.floor(data.progress / 20);
                    steps.forEach((stepId, index) => {{
                        const element = document.getElementById(stepId);
                        if (index <= currentStep) {{
                            element.classList.add('step-active');
                        }}
                    }});
                    
                    // Check if complete
                    if (data.status === 'complete') {{
                        progressFill.style.width = '100%';
                        statusMessage.textContent = 'Complete! Redirecting...';
                        setTimeout(() => {{
                            window.location.href = "/wiki/" + data.slug;
                        }}, 1000);
                    }} else if (data.status === 'error') {{
                        // Show error state
                        progressFill.style.background = 'var(--error)';
                        progressFill.style.width = '100%';
                        statusMessage.textContent = 'Error: ' + data.message;
                        statusMessage.style.color = 'var(--error)';
                        statusMessage.style.fontWeight = '600';
                        
                        // Hide spinner and show error icon
                        const spinner = document.querySelector('.spinner');
                        if (spinner) {{
                            spinner.style.display = 'none';
                        }}
                        
                        // Show error message prominently
                        const errorDiv = document.createElement('div');
                        errorDiv.style.cssText = 'background: rgba(166, 68, 68, 0.1); border: 2px solid var(--error); padding: 20px; border-radius: 12px; margin-top: 24px; color: var(--error);';
                        errorDiv.innerHTML = '<strong style="display:block;margin-bottom:12px;font-size:16px;">Generation Failed</strong><div style="font-size:14px;line-height:1.6;">' + data.message + '</div>';
                        document.querySelector('.card').appendChild(errorDiv);
                        
                        // Add a "Return to Generate Page" button
                        const returnBtn = document.createElement('a');
                        returnBtn.href = '/generate?error=' + encodeURIComponent(data.message);
                        returnBtn.className = 'btn';
                        returnBtn.style.cssText = 'margin-top: 24px; display: inline-block; background: var(--error); color: white;';
                        returnBtn.textContent = 'Return to Generate Page';
                        returnBtn.onclick = (e) => {{
                            e.preventDefault();
                            window.location.href = '/generate?error=' + encodeURIComponent(data.message);
                        }};
                        document.querySelector('.card').appendChild(returnBtn);
                        
                        // Auto-redirect back to generate page after 5 seconds
                        setTimeout(() => {{
                            const errorMsg = encodeURIComponent(data.message);
                            window.location.href = "/generate?error=" + errorMsg;
                        }}, 5000);
                    }} else {{
                        // Continue polling
                        setTimeout(updateStatus, 2000);
                    }}
                }})
                .catch(error => {{
                    console.error('Error fetching status:', error);
                    setTimeout(updateStatus, 3000);
                }});
        }}
        
        // Start polling
        setTimeout(updateStatus, 1000);
    </script>
    '''
    
    return render(body)


@app.route("/api/generation-status/<job_id>")
def generation_status_api(job_id: str):
    """API endpoint to check generation status"""
    with generation_lock:
        status = generation_status.get(job_id, {
            'status': 'unknown',
            'progress': 0,
            'message': 'Status not found',
            'topic': 'Unknown'
        })
    
    return jsonify(status)


@app.route("/documents")
def documents():
    q = (request.args.get("q") or "").strip()
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1
    try:
        per_page = max(1, min(100, int(request.args.get("per_page", 50))))
    except ValueError:
        per_page = 50

    with get_session() as session:
        query = session.query(Document)
        if q:
            like = f"%{q}%"
            query = query.filter(or_(Document.title.ilike(like), Document.external_id.ilike(like)))
        total = query.count()
        docs = query.order_by(Document.created_at.desc()).offset((page - 1) * per_page).limit(per_page).all()

        start_index = (page - 1) * per_page + 1
        rows = []
        for idx, d in enumerate(docs, start=start_index):
            page_count = session.query(Page).filter_by(document_id=d.id).count()
            total_words = 0
            if page_count:
                pages = session.query(Page).filter_by(document_id=d.id).all()
                total_words = sum(len(p.text.split()) if p.text else 0 for p in pages)
            has_pdf = bool((d.file_path and Path(d.file_path).exists()) or d.url)
            pdf_link = url_for('serve_pdf', external_id=d.external_id) if d.external_id else None
            rows.append({
                'index': idx,
                'external_id': d.external_id or '(none)',
                'title': d.title or '(untitled)',
                'source': d.source.name if d.source else '(unknown)',
                'pages': page_count,
                'words': total_words,
                'has_pdf': has_pdf,
                'pdf_link': pdf_link,
            })

    body = [
        '<div class="container" style="padding-top:48px;padding-bottom:48px;">',
        '<div class="card">',
        '<div style="margin-bottom:32px;">',
        '<h2 style="margin-bottom:8px;">Document Library</h2>',
        '<p class="muted" style="font-size:16px;">Browse and search through <strong>{:,}</strong> declassified FOIA documents</p>'.format(total),
        '</div>',
        '<form method="get" style="display:flex;gap:12px;margin-bottom:32px;">',
        f'<input class="search-input" type="text" name="q" placeholder="Search by title, external ID, or content..." value="{request.args.get("q","")}" style="flex:1;padding:14px 20px;">',
        '<button class="btn" type="submit">Search</button>',
        '</form>',
        '<div style="overflow-x:auto;margin:-32px;padding:32px;margin-top:0;">',
        '<table>',
        '<thead><tr>',
        '<th style="text-align:right;">#</th>',
        '<th style="text-align:left;">External ID</th>',
        '<th style="text-align:left;">Title</th>',
        '<th style="text-align:left;">Source</th>',
        '<th style="text-align:right;">Pages</th>',
        '<th style="text-align:right;">Words</th>',
        '<th style="text-align:left;">PDF</th>',
        '<th style="text-align:left;">Text</th>',
        '</tr></thead><tbody>'
    ]
    if rows:
        for r in rows:
            pdf_cell = (f'<a class="btn secondary small" href="{r["pdf_link"]}">PDF</a>' if r['has_pdf'] and r['pdf_link'] else '<span class="muted">(none)</span>')
            text_link = f'<a class="btn secondary small" href="{url_for("document_text", external_id=r["external_id"])}">Text</a>'
            body.append(
                '<tr>'
                f'<td style="text-align:right;"><strong>{r["index"]}</strong></td>'
                f'<td><code style="font-size:13px;">{r["external_id"]}</code></td>'
                f'<td><strong>{r["title"]}</strong></td>'
                f'<td><span class="tag">{r["source"]}</span></td>'
                f'<td style="text-align:right;">{r["pages"]}</td>'
                f'<td style="text-align:right;"><strong>{r["words"]:,}</strong></td>'
                f'<td>{pdf_cell}</td>'
                f'<td>{text_link}</td>'
                '</tr>'
            )
    else:
        body.append('<tr><td colspan="8" class="muted" style="padding:40px;text-align:center;font-size:16px;">No documents found</td></tr>')
    last_index = (rows[-1]['index'] if rows else 0)
    showing_from = rows[0]['index'] if rows else 0
    showing_to = last_index
    total_pages = (total + per_page - 1) // per_page if per_page else 1
    prev_link = None
    next_link = None
    if page > 1:
        prev_link = url_for('documents', q=q, page=page-1, per_page=per_page)
    if page < total_pages:
        next_link = url_for('documents', q=q, page=page+1, per_page=per_page)

    body.append('</tbody></table>')
    body.append('</div>')  
    
    body.append('<div style="margin-top:32px; padding-top:24px; border-top:2px solid var(--border-light);">')
    
    body.append('<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;flex-wrap:wrap;gap:16px;">')
    body.append(f'<div style="font-size:15px;font-weight:600;color:var(--text);">Showing <span style="color:var(--accent);">{showing_from:,}-{showing_to:,}</span> of <span style="color:var(--accent);">{total:,}</span> documents</div>')
    
    body.append('<div style="display:flex; align-items:center; gap:10px;">')
    body.append('<span style="font-size:14px;font-weight:600;color:var(--muted);">Per page:</span>')
    for size in [25, 50, 100]:
        if size == per_page:
            body.append(f'<span class="btn small" style="cursor:default;">{size}</span>')
        else:
            body.append(f'<a class="btn secondary small" href="{url_for("documents", q=q, page=1, per_page=size)}">{size}</a>')
    body.append('</div>')
    body.append('</div>')
    
    body.append('<div style="display:flex; justify-content:center; align-items:center; gap:8px;flex-wrap:wrap;">')
    
    if page > 1:
        body.append(f'<a class="btn secondary small" href="{url_for("documents", q=q, page=1, per_page=per_page)}">«« First</a>')
    
    if page > 1:
        body.append(f'<a class="btn secondary small" href="{url_for("documents", q=q, page=page-1, per_page=per_page)}">‹ Prev</a>')
    
    start_page = max(1, page - 2)
    end_page = min(total_pages, page + 2)
    
    if start_page > 1:
        body.append('<span class="muted" style="padding:0 4px;">...</span>')
    
    for p in range(start_page, end_page + 1):
        if p == page:
            body.append(f'<span class="btn small" style="cursor:default;min-width:40px;">{p}</span>')
        else:
            body.append(f'<a class="btn secondary small" href="{url_for("documents", q=q, page=p, per_page=per_page)}" style="min-width:40px;">{p}</a>')
    
    if end_page < total_pages:
        body.append('<span class="muted" style="padding:0 4px;">...</span>')
    
    if page < total_pages:
        body.append(f'<a class="btn secondary small" href="{url_for("documents", q=q, page=page+1, per_page=per_page)}">Next ›</a>')
    
    if page < total_pages:
        body.append(f'<a class="btn secondary small" href="{url_for("documents", q=q, page=total_pages, per_page=per_page)}">Last »»</a>')
    
    body.append('</div>')
    
    body.append(f'<div style="text-align:center; margin-top:16px;"><span class="muted" style="font-size:14px;">Page <strong>{page}</strong> of <strong>{total_pages:,}</strong></span></div>')
    
    body.append('</div></div></div>')

    return render("".join(body))

@app.route("/documents/<external_id>")
def document_text(external_id: str):
    with get_session() as session:
        doc = session.query(Document).filter(Document.external_id == external_id).first()
        if not doc:
            return render('<div class="error">Document not found</div>')
        pages = session.query(Page).filter_by(document_id=doc.id).order_by(Page.page_no.asc()).all()

        body = [
            '<div class="card">',
            f'<h2>Document: {external_id}</h2>',
            f'<div class="muted">Source: {doc.source.name if doc.source else "(unknown)"} • Pages: {len(pages)}</div>',
            '<div style="margin-top:10px;">',
            f'<a class="btn secondary" href="{url_for("serve_pdf", external_id=external_id)}">Open PDF</a> ',
            f'<a class="btn secondary" href="{url_for("documents")}">Back to Documents</a>',
            '</div>',
            '</div>'
        ]

        for p in pages:
            method = "OCR" if p.ocr_confidence is not None else "Text"
            word_count = len(p.text.split()) if p.text else 0
            preview = (p.text or "")
            body += [
                '<div class="card">',
                f'<h3>Page {p.page_no} <span class="tag">{method}</span> <span class="tag">{word_count} words</span></h3>',
                f'<div class="content" style="white-space:pre-wrap;">{(preview or "(no text)")}</div>',
                '</div>'
            ]

        if not pages:
            body.append('<div class="card"><div class="muted">No extracted pages for this document.</div></div>')

        return render("".join(body))

@app.route("/pdf/<external_id>")
def serve_pdf(external_id: str):
    """
    Serve PDF with page navigation support.
    Uses JavaScript to extract #page=X from URL fragment and display PDF at that page.
    """
    from flask import send_file, redirect
    
    with get_session() as session:
        doc = session.query(Document).filter(Document.external_id == external_id).first()
        if not doc:
            return render('<div class="error">Document not found</div>')
        
        viewer_html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>{doc.title or external_id}</title>
            <style>
                body {{ 
                    margin: 0; 
                    padding: 0; 
                    overflow: hidden; 
                    background: #525659;
                }}
                iframe {{ 
                    width: 100vw; 
                    height: 100vh; 
                    border: none; 
                }}
                #loading {{
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    color: white;
                    font-family: Arial, sans-serif;
                    font-size: 18px;
                }}
            </style>
        </head>
        <body>
            <div id="loading">Loading PDF...</div>
            <iframe id="pdf-viewer" style="display:none;"></iframe>
            <script>
                // Extract page number from URL fragment (#page=5)
                function getPageFromHash() {{
                    const hash = window.location.hash;
                    const match = hash.match(/page=(\\d+)/);
                    return match ? match[1] : null;
                }}
                
                // Build PDF URL with page parameter
                function buildPdfUrl() {{
                    const pageNo = getPageFromHash();
                    let url = '/pdf-file/{external_id}';
                    
                    if (pageNo) {{
                        // Use #page=X format (standard PDF viewer format)
                        url += '#page=' + pageNo;
                    }}
                    
                    return url;
                }}
                
                // Load the PDF
                const iframe = document.getElementById('pdf-viewer');
                const loading = document.getElementById('loading');
                
                iframe.onload = function() {{
                    loading.style.display = 'none';
                    iframe.style.display = 'block';
                }};
                
                iframe.src = buildPdfUrl();
                
                // Handle hash changes (if user navigates)
                window.addEventListener('hashchange', function() {{
                    iframe.src = buildPdfUrl();
                }});
            </script>
        </body>
        </html>
        '''
        return viewer_html


@app.route("/pdf-file/<external_id>")
def serve_pdf_file(external_id: str):
    """Direct PDF file serving (used by iframe viewer)"""
    from flask import send_file, redirect
    with get_session() as session:
        doc = session.query(Document).filter(Document.external_id == external_id).first()
        if not doc:
            return "PDF not found", 404
        
        if doc.file_path:
            p = Path(doc.file_path)
            if p.exists():
                    return send_file(str(p), mimetype='application/pdf', as_attachment=False)
        
        if doc.url:
            return redirect(doc.url)
        
        return "PDF unavailable", 404


def generate_wiki_background(job_id: str, topic: str, max_chunks: int, diversity_mode: str, length: str):
    """Background task to generate wiki page"""
    try:
        with generation_lock:
            generation_status[job_id] = {
                'status': 'searching',
                'progress': 5,
                'message': 'Initializing federated search system...',
                'topic': topic
            }
        
        global wiki_generator
        if wiki_generator is None:
            wiki_generator = ContextAwareWikiGenerator()
        
        with generation_lock:
            generation_status[job_id] = {
                'status': 'searching',
                'progress': 15,
                'message': f'Searching across {len(wiki_generator.search_system.batch_info)} batch indices...',
                'topic': topic
            }
        
        time.sleep(0.5)
        
        with generation_lock:
            generation_status[job_id] = {
                'status': 'searching',
                'progress': 30,
                'message': 'Querying semantic and keyword indices...',
                'topic': topic
            }
        
        with generation_lock:
            generation_status[job_id] = {
                'status': 'ranking',
                'progress': 45,
                'message': 'Ranking chunks and filtering by relevance score (> 0.4)...',
                'topic': topic
            }
        
        with generation_lock:
            generation_status[job_id] = {
                'status': 'generating',
                'progress': 55,
                'message': f'Generating {length} wiki article with GPT-5 Nano...',
                'topic': topic
            }
        
        out_path = wiki_generator.generate_and_save(topic, max_chunks=max_chunks, diversity_mode=diversity_mode, length=length)
        
        clear_index_cache()
        
        with generation_lock:
            generation_status[job_id] = {
                'status': 'finalizing',
                'progress': 85,
                'message': 'Formatting citations and references...',
                'topic': topic
            }
        
        time.sleep(0.3)
        
        with generation_lock:
            generation_status[job_id] = {
                'status': 'complete',
                'progress': 100,
                'message': 'Wiki page generated successfully!',
                'topic': topic,
                'slug': out_path.stem
            }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        traceback.print_exc()
        
        error_msg = str(e)
        if not error_msg:
            error_msg = "An unexpected error occurred during wiki generation"
        
        if len(error_msg) > 200:
            error_msg = error_msg[:197] + "..."
        
        with generation_lock:
            generation_status[job_id] = {
                'status': 'error',
                'progress': 0,
                'message': error_msg,
                'topic': topic,
                'error_detail': error_trace
            }

@app.route("/generate", methods=["GET", "POST"]) 
def generate():
    error_msg = request.args.get("error")
    if error_msg:
        flash(f"Generation failed: {error_msg}", "error")
    
    topics = []
    topics_with_counts = []
    
    try:
        import json
        from datetime import datetime, timedelta
        
        dynamic_topics_file = ROOT / "data" / "dynamic_topics_cache.json"
        if dynamic_topics_file.exists():
            with open(dynamic_topics_file) as f:
                cache_data = json.load(f)
            
            generated_at = datetime.fromisoformat(cache_data['generated_at'])
            if datetime.now() - generated_at < timedelta(days=7):
                topics_with_counts = [(t['name'], t['count']) for t in cache_data['topics']]
                topics = [t['name'] for t in cache_data['topics']]
                print(f"Loaded {len(topics)} discovered topics for dropdown")
    except Exception as e:
        print(f"Dynamic topics unavailable for dropdown: {e}")
    
    if not topics:
        try:
            entity = load_entity_data()
            clusters, defs = create_topic_clusters(entity)
            topics = sorted(clusters.keys())
            topics_with_counts = [(t, len(clusters[t])) for t in topics]
            print(f"Fallback: Loaded {len(topics)} topics from entity extraction")
        except Exception:
            topics = []
            topics_with_counts = []
            print("No topics available for dropdown")

    if request.method == "POST":
        custom_topic = (request.form.get("custom_topic") or "").strip()
        dropdown_topic = request.form.get("topic") or ""
        
        topic = custom_topic if custom_topic else dropdown_topic
        
        if not topic:
            return render('<div class="container" style="padding-top:40px;"><div class="card error">Please provide a topic</div></div>')
        
        max_chunks = int(request.form.get("max_chunks") or 40)
        diversity_mode = request.form.get("diversity_mode") or "balanced"
        length = request.form.get("length") or "medium"
        
        job_id = f"{topic}_{int(time.time())}"
        
        if CONTEXT_AWARE_GENERATION:
            thread = threading.Thread(
                target=generate_wiki_background,
                args=(job_id, topic, max_chunks, diversity_mode, length)
            )
            thread.daemon = True
            thread.start()
            
            return redirect(url_for("generating", job_id=job_id))
        else:
            try:
                max_chars = int(request.form.get("max_chars") or 12000)
                temperature = float(request.form.get("temperature") or 0.3)
                out_path = generate_topic(topic, max_chars=max_chars, temperature=temperature)
                flash(f"Generated wiki page for: {topic}")
                return redirect(url_for("view_page", slug=out_path.stem))
            except Exception as e:
                import traceback
                traceback.print_exc()
                flash(f"Error generating wiki page: {str(e)}", "error")
                return redirect(url_for("generate"))


    
    
        
            
    

        
        
        
        
        
            

    topic = request.args.get("topic", "")
    body = [
        '<div class="container" style="padding-top:48px;padding-bottom:48px;">',
        '<div style="max-width:900px;margin:0 auto;">',
        '<div class="card">',
        '<div style="text-align:center;margin-bottom:32px;">',
        '<h2 style="margin-bottom:12px;">Generate Wiki Page</h2>',
        '<p class="muted" style="font-size:16px;">Choose from discovered topics or enter your own custom topic to generate a comprehensive wiki article</p>',
        '</div>',
        '<form method="post">',
        
        '<div class="form-section">',
        '<h3>Custom Topic</h3>',
        '<input type="text" name="custom_topic" placeholder="e.g., \'CIA Operations in Afghanistan\', \'Nuclear Weapons Programs\', \'Drug Trafficking Colombia\'" ',
        'style="width: 100%; font-size: 16px;" />',
        '<p class="muted" style="margin: 12px 0 0 0; font-size: 14px;">',
        '<strong>Tip:</strong> Be specific for better results. The AI will perform federated search across all 6,000+ documents.',
        '</p>',
        '</div>',
        
        '<div style="text-align: center; margin: 28px 0; position:relative;">',
        '<div style="position:absolute;top:50%;left:0;right:0;height:1px;background:var(--border);"></div>',
        '<span style="position:relative;background:white;padding:0 16px;color:var(--muted);font-weight:700;font-size:13px;text-transform:uppercase;letter-spacing:1px;">OR</span>',
        '</div>',
        
        '<div class="form-section">',
        '<h3>Intelligence Topics <span class="tag" style="margin-left:8px;">{} available</span></h3>'.format(len(topics)),
        '<div class="row">',
        '<div style="flex: 1;"><label>Select a Pre-Discovered Topic</label>',
        '<select name="topic" style="width: 100%;">'
    ]
    if topics_with_counts:
        body.append('<option value="">-- Select an intelligence topic --</option>')
        for topic_name, count in topics_with_counts:
            sel = 'selected' if topic_name == topic else ''
            display_text = f"{topic_name} ({count:,} docs)"
            body.append(f'<option value="{topic_name}" {sel}>{display_text}</option>')
    elif topics:
        body.append('<option value="">-- Select a topic --</option>')
        for t in topics:
            sel = 'selected' if t == topic else ''
            body.append(f'<option value="{t}" {sel}>{t}</option>')
    else:
        body.append('<option value="">No topics available</option>')
    body.append('</select></div>')
    body.append('</div>')
    body.append('</div>')
    
    body.append('<div class="form-section">')
    body.append('<h3>Advanced Options</h3>')
    
    if CONTEXT_AWARE_GENERATION:
        body.append('<div class="muted" style="background:var(--surface);padding:16px;border-radius:var(--radius-md);border:1px solid var(--border);margin-bottom:24px;font-size:14px;line-height:1.6;">')
        body.append('<div class="row">')
        body.append('<div style="flex:1;"><label>Max Context Chunks</label>')
        body.append('<select name="max_chunks" style="width:100%;">')
        body.append('<option value="40" selected>40</option>')
        body.append('<option value="60">60</option>')
        body.append('<option value="80">80</option>')
        body.append('</select></div>')
        body.append('<div style="flex:1;"><label>Document Diversity</label>')
        body.append('<select name="diversity_mode" style="width:100%;">')
        body.append('<option value="strict">Strict (1 chunk/doc - max diversity)</option>')
        body.append('<option value="balanced" selected>Balanced (2 chunks/doc - recommended)</option>')
        body.append('<option value="relaxed">Relaxed (3 chunks/doc - more depth)</option>')
        body.append('<option value="best">Best (unlimited - quality over diversity)</option>')
        body.append('</select></div>')
        body.append('<div style="flex:1;"><label>Target Length</label>')
        body.append('<select name="length" style="width:100%;">')
        body.append('<option value="short">Short (Brief Summary)</option>')
        body.append('<option value="medium" selected>Medium (Standard Wiki)</option>')
        body.append('<option value="long">Long (Deep Dive)</option>')
        body.append('<option value="exhaustive">Exhaustive (Max Detail)</option>')
        body.append('</select></div>')
        body.append('</div>')
    else:
        body.append('<div class="row">')
        body.append('<div><label>Max Characters</label><input type="number" name="max_chars" value="12000"/></div>')
        body.append('<div><label>Temperature (0-1)</label><input type="number" step="0.1" name="temperature" value="0.3"/></div>')
        body.append('</div>')
        body.append('<p class="muted" style="margin: 12px 0 0 0; font-size: 14px;">Higher temperature = more creative, lower = more focused</p>')
    
    body.append('</div>')
    
    body.append('<button class="btn large" type="submit" style="width: 100%; margin-top:8px;">Generate Wiki Page</button>')
    body.append('</form></div></div></div>')

    return render("".join(body))


@app.route("/wiki/<slug>")
def view_page(slug: str):
    md_path = ROOT / "data/wiki" / f"{slug}.md"
    if not md_path.exists():
        return render('<div class="container" style="padding-top:48px;"><div class="error">Wiki page not found</div></div>')

    raw = md_path.read_text(encoding="utf-8")

    
    html = md.markdown(raw, extensions=["fenced_code", "tables", "toc"])  # basic rendering

    body = [
        '<div class="container" style="padding-top:48px;padding-bottom:64px;">',
        '<div style="max-width:900px;margin:0 auto;">',
        '<div class="card">',
        '<div style="border-bottom:2px solid var(--border-light);padding-bottom:24px;margin-bottom:32px;">',
        f'<h1 style="font-size:36px;font-weight:800;color:var(--text);margin-bottom:12px;letter-spacing:-0.02em;">{slug.replace("_", " ")}</h1>',
        '<div style="display:flex;gap:12px;align-items:center;">',
        '<span class="tag">Generated Wiki</span>',
        f'<a class="btn secondary small" href="{url_for("validate_page", slug=slug)}">Validate Citations</a>',
        f'<a class="btn secondary small" href="{url_for("chunks_page", slug=slug)}">View Chunks</a>',
        f'<a class="btn secondary small" href="{url_for("index")}">← Back to Home</a>',
        '</div>',
        '</div>',
        '<div class="content" style="font-size:17px;line-height:1.8;">',
        html,
        '</div>',
        '</div>',
        '</div>',
        '</div>'
    ]
    return render("".join(body))


@app.route("/search")
def search():
    q = (request.args.get("q") or "").strip()
    search_mode = request.args.get("mode", "quick").lower()
    valid_modes = {"quick", "keyword", "advanced"}
    if search_mode not in valid_modes:
        search_mode = "quick"
    
    if not q:
        return render('<div class="container" style="padding-top:40px;"><div class="card"><h2>Search</h2><div class="muted">No query provided</div></div></div>')
    
    if HYBRID_SEARCH_AVAILABLE:
        try:
            if search_mode == "quick":
                print(f"Web UI quick search (semantic-only) for: {q}")
                manager = get_search_manager()
                search_results = manager.search(
                    query=q,
                    top_k=15,  
                    search_mode="semantic"  
                )
                enhanced_results = []
                for result in search_results:
                    enhanced_result = {
                        'doc_id': result.get('doc_id', ''),
                        'title': result.get('title', 'Untitled'),
                        'source': result.get('source', 'Unknown'),
                        'url': result.get('url', ''),
                        'snippet': result.get('chunk_text', '')[:300] + "..." if len(result.get('chunk_text', '')) > 300 else result.get('chunk_text', ''),
                        'relevance_score': result.get('score', 0),
                        'semantic_score': result.get('semantic_score', result.get('score', 0)),
                        'bm25_score': 0,  
                        'batch_name': result.get('batch_name', 'unknown'),
                        'search_type': 'quick'
                    }
                    enhanced_results.append(enhanced_result)
                search_results = enhanced_results
            elif search_mode == "keyword":
                print(f"Web UI keyword search (bm25) for: {q}")
                manager = get_search_manager()
                search_results = manager.search(
                    query=q,
                    top_k=20,
                    search_mode="bm25"
                )
                enhanced_results = []
                for result in search_results:
                    chunk_text = result.get('chunk_text', result.get('text', ''))
                    enhanced_result = {
                        'doc_id': result.get('doc_id', ''),
                        'title': result.get('title', 'Untitled'),
                        'source': result.get('source', 'Unknown'),
                        'url': result.get('url', ''),
                        'snippet': chunk_text[:300] + "..." if len(chunk_text) > 300 else chunk_text,
                        'relevance_score': result.get('score', result.get('bm25_score', 0)),
                        'semantic_score': 0,
                        'bm25_score': result.get('bm25_score', result.get('score', 0)),
                        'batch_name': result.get('batch_name', 'unknown'),
                        'search_type': 'keyword'
                    }
                    enhanced_results.append(enhanced_result)
                search_results = enhanced_results
            else:
                print(f"Web UI advanced search (hybrid) for: {q}")
                search_results = search_documents(q, top_k=20)
            
            print(f"Web UI got {len(search_results)} results")
            if search_results:
                print(f"   First result keys: {search_results[0].keys()}")
            
            if search_results:
                unique_docs = len(set(r['doc_id'] for r in search_results))
                avg_chunks = len(search_results) / unique_docs if unique_docs > 0 else 0
            else:
                unique_docs = 0
                avg_chunks = 0
            
            mode_meta = {
                "quick": ("Quick Search (Semantic)", "Fast semantic search for conceptual matches"),
                "keyword": ("Keyword Search (BM25)", "Keyword-only BM25 ranking for exact term matches"),
                "advanced": ("Advanced Search (Hybrid)", "Hybrid search combining semantic + keyword matching"),
            }
            mode_label, mode_desc = mode_meta.get(search_mode, mode_meta["quick"])
            
            topic_hits = []
            page_hits = []
            
            if search_mode == "advanced":
                try:
                    entity = load_entity_data()
                    clusters, defs = create_topic_clusters(entity)
                    q_lower = q.lower()
                    for topic, docs in clusters.items():
                        if q_lower in topic.lower():
                            topic_hits.append({"topic": topic, "count": len(docs)})
                except Exception:
                    pass
            
            for p in list_generated_pages():
                title = p.stem.replace('_', ' ')
                if q.lower() in title.lower():
                    page_hits.append({"slug": p.stem, "title": title})
            
            body = [
                '<div class="container" style="padding-top:48px;padding-bottom:64px;">',
                '<div class="card" style="text-align:center;">',
                f'<h2 style="margin-bottom:12px;">Search Results</h2>',
                f'<p style="font-size:18px;color:var(--text-light);margin-bottom:8px;">"{safe_html(q)}"</p>',
                f'<div style="margin-bottom:16px;">',
                f'<span class="tag" style="font-size:13px;padding:6px 12px;background:rgba(217,140,33,0.1);color:var(--accent);border-color:var(--accent);">{mode_label}</span>',
                f'<span style="font-size:13px;color:var(--text-light);margin-left:8px;">{mode_desc}</span>',
                f'</div>',
                f'<div style="display:flex;gap:16px;justify-content:center;margin-top:20px;flex-wrap:wrap;">',
                f'<div class="tag" style="font-size:14px;padding:8px 16px;">{len(search_results)} relevant chunks</div>',
                f'<div class="tag" style="font-size:14px;padding:8px 16px;">{unique_docs} unique documents</div>',
                f'<div class="tag" style="font-size:14px;padding:8px 16px;">{avg_chunks:.1f} avg chunks/doc</div>',
                '</div>',
                '</div>',
            ]
            
            query_param = quote_plus(q)
            mode_switch_labels = {
                "quick": "Switch to Quick Semantic",
                "keyword": "Switch to Keyword Search",
                "advanced": "Switch to Advanced Hybrid",
            }
            switch_links = []
            for mode_key, link_label in mode_switch_labels.items():
                if mode_key == search_mode:
                    continue
                switch_links.append(
                    f'<a href="/search?q={query_param}&mode={mode_key}" class="btn secondary" style="text-decoration:none;">{link_label}</a>'
                )
            if switch_links:
                body.append(
                    '<div style="text-align:center;margin-top:16px;margin-bottom:24px;display:flex;gap:12px;justify-content:center;flex-wrap:wrap;">'
                    + "".join(switch_links) +
                    '</div>'
                )
            
            if search_results:
                body.append('<div style="margin-top:32px;"><h3 style="font-size:24px;font-weight:700;margin-bottom:24px;color:var(--text);">Document Matches</h3></div>')
                
                for i, result in enumerate(search_results, 1):
                    body.append(f'''
                    <div class="card card-hover" style="margin-bottom:20px;">
                        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;gap:20px;">
                            <div style="flex:1;">
                                <div style="font-size:18px;font-weight:700;margin-bottom:6px;">
                                    <a href="{result['url'] or '#'}" style="color:var(--text);text-decoration:none;">
                                        {safe_html(result['title'])}
                                    </a>
                                </div>
                                <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
                                    <span class="tag">{safe_html(result['source'])}</span>
                                    <span style="font-size:13px;color:var(--muted);">Doc ID: <code style="font-size:12px;">{result['doc_id']}</code></span>
                            </div>
                            </div>
                            <div style="text-align:right;padding:12px 20px;background:linear-gradient(135deg,var(--accent-light),#dbeafe);border-radius:var(--radius-md);border:1px solid rgba(37,99,235,0.15);">
                                <div style="font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:0.5px;">Score</div>
                                <div style="font-size:24px;font-weight:800;color:var(--accent);margin-top:2px;">{result['relevance_score']:.2f}</div>
                        </div>
                        </div>
                        <div style="background:var(--surface);padding:16px;border-radius:var(--radius-md);font-size:15px;line-height:1.7;color:var(--text-light);border:1px solid var(--border-light);">
                            {safe_html(result['snippet'])}
                        </div>
                        <div style="margin-top:14px;display:flex;gap:16px;font-size:13px;font-weight:600;">
                            <span style="color:var(--muted);">Semantic: <span style="color:var(--accent);">{result['semantic_score']:.3f}</span></span>
                            <span style="color:var(--muted);">Keyword: <span style="color:var(--accent);">{result['bm25_score']:.3f}</span></span>
                        </div>
                    </div>
                    ''')
            
            if topic_hits:
                body.append('<div style="margin-top:48px;"><h3 style="font-size:24px;font-weight:700;margin-bottom:24px;color:var(--text);">Related Topics</h3></div>')
                body.append('<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:20px;">')
                for hit in topic_hits:
                    body.append(
                        f'<div class="card card-hover" style="margin:0;">'
                        f'<div style="margin-bottom:16px;"><strong style="font-size:17px;color:var(--text);">{hit["topic"]}</strong></div>'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                        f'<span class="tag">{hit["count"]} documents</span>'
                        f'<a class="btn small" href="{url_for("generate", topic=hit["topic"])}">Generate Wiki</a>'
                        f'</div>'
                        f'</div>'
                    )
                body.append('</div>')
            
            if page_hits:
                body.append('<div style="margin-top:48px;"><h3 style="font-size:24px;font-weight:700;margin-bottom:24px;color:var(--text);">Wiki Pages</h3></div>')
                body.append('<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px;">')
                for hit in page_hits:
                    body.append(
                        f'<a href="{url_for("view_page", slug=hit["slug"])}" class="card card-hover" style="margin:0;text-decoration:none;">'
                        f'<div style="font-weight:600;font-size:16px;color:var(--text);">{hit["title"]}</div>'
                        f'<div style="margin-top:8px;font-size:14px;color:var(--accent);">View article →</div>'
                        f'</a>'
                    )
                body.append('</div>')
            
            if not search_results and not topic_hits and not page_hits:
                body.append('<div class="card" style="text-align:center;padding:60px 40px;"><div style="font-size:48px;margin-bottom:16px;"></div><div style="font-size:20px;font-weight:600;color:var(--text);margin-bottom:8px;">No results found</div><div class="muted" style="font-size:16px;">Try different keywords or check spelling</div></div>')
            
            body.append('</div>')
            return render("".join(body))
            
        except Exception as e:
            print(f"Hybrid search error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("HYBRID_SEARCH_AVAILABLE is False")

    q_lower = q.lower()
    topic_hits = []
    page_hits = []

    try:
        entity = load_entity_data()
        clusters, defs = create_topic_clusters(entity)
        for topic, docs in clusters.items():
            if q_lower in topic.lower():
                topic_hits.append({"topic": topic, "count": len(docs)})
    except Exception:
        pass

    for p in list_generated_pages():
        title = p.stem.replace('_', ' ')
        match_title = q_lower in title.lower()
        snippet = ""
        if not match_title:
            try:
                txt = p.read_text(encoding="utf-8")
                idx = txt.lower().find(q_lower) if q else -1
                if idx >= 0:
                    start = max(0, idx - 120)
                    end = min(len(txt), idx + len(q) + 120)
                    snippet = txt[start:end].replace('\n', ' ')
            except Exception:
                snippet = ""
        if match_title or snippet:
            page_hits.append({"slug": p.stem, "title": title, "snippet": snippet})

    body = [
        '<div class="container" style="padding-top:40px;">',
        '<div class="card">',
        f'<h2>Search: {q}</h2>',
        '<p class="muted">Basic search (hybrid search not available)</p>',
        '</div>',

        '<div class="card">',
        '<h3>Topic Matches</h3>'
    ]
    if topic_hits:
        for hit in topic_hits:
            body.append(
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin:6px 0;">'
                f'<div><strong>{hit["topic"]}</strong> <span class="tag">{hit["count"]} docs</span></div>'
                f'<div><a class="btn" href="{url_for("generate", topic=hit["topic"])}">Generate</a></div>'
                f'</div>'
            )
    else:
        body.append('<div class="muted">No topic matches</div>')
    body.append('</div>')

    body.append('<div class="card">')
    body.append('<h3>Wiki Page Matches</h3>')
    if page_hits:
        for hit in page_hits:
            preview = hit["snippet"][:220] + ('...' if len(hit["snippet"]) > 220 else '')
            body.append(
                f'<div style="margin:10px 0;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<div><strong>{hit["title"]}</strong></div>'
                f'<div><a class="btn secondary" href="{url_for("view_page", slug=hit["slug"])}">Open</a></div>'
                f'</div>'
                f'<div class="muted" style="margin-top:6px;">{preview}</div>'
                f'</div>'
            )
    else:
        body.append('<div class="muted">No page matches</div>')
    body.append('</div></div>')

    return render("".join(body))


@app.route("/wiki/<slug>/validate")
def validate_page(slug: str):
    md_path = ROOT / "data/wiki" / f"{slug}.md"
    if not md_path.exists():
        return render('<div class="error">Page not found</div>')

    meta_path = ROOT / "data/wiki" / f"{slug}.meta.json"
    report = None
    
    if meta_path.exists():
        try:
            import json
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            if 'validation_cache' in metadata:
                from foia_ai.synthesis.citation_validator import deserialize_report
                report = deserialize_report(metadata['validation_cache'], md_path)
                print(f"Using cached validation data for {slug}")
        except Exception as e:
            print(f"Error loading cached validation: {e}, falling back to fresh validation")
    
    if report is None:
        report = validate_file(md_path)

    doc_info = {}
    if report.validated_citations:
        doc_ids = set(v.citation.document_id for v in report.validated_citations)
        with get_session() as session:
            docs = session.query(Document).filter(Document.external_id.in_(doc_ids)).all()
            for d in docs:
                doc_info[d.external_id] = {
                    'title': d.title or d.external_id,
                    'source': d.source.name if d.source else "Unknown Source"
                }

    density_color = "#d4edda" if report.citation_density >= 2.0 else "#fff3cd" if report.citation_density >= 1.0 else "#f8d7da"
    density_text_color = "#155724" if report.citation_density >= 2.0 else "#856404" if report.citation_density >= 1.0 else "#721c24"
    
    relevance_color = "#d4edda" if report.avg_relevance_score >= 0.8 else "#fff3cd" if report.avg_relevance_score >= 0.5 else "#f8d7da"
    relevance_text_color = "#155724" if report.avg_relevance_score >= 0.8 else "#856404" if report.avg_relevance_score >= 0.5 else "#721c24"
    
    body = [
        '<div class="card">',
        f'<h2>Citation Validation Report: {slug.replace("_", " ")}</h2>',
        
        '<h3 style="margin-top: 24px; margin-bottom: 16px;">Overall Statistics</h3>',
        '<div style="display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 24px;">',
        
        f'<div style="flex: 1; min-width: 200px; padding: 16px; background: var(--surface); border-radius: 8px; border: 1px solid var(--border);">',
        f'<div style="font-size: 14px; color: var(--muted); margin-bottom: 4px;">Total Citations</div>',
        f'<div style="font-size: 28px; font-weight: 700; color: var(--text);">{report.total_citations}</div>',
        f'</div>',
        
        f'<div style="flex: 1; min-width: 200px; padding: 16px; background: #d4edda; border-radius: 8px; border: 1px solid #c3e6cb;">',
        f'<div style="font-size: 14px; color: #155724; margin-bottom: 4px;">Valid Citations</div>',
        f'<div style="font-size: 28px; font-weight: 700; color: #155724;">{report.valid}</div>',
        f'<div style="font-size: 12px; color: #155724; margin-top: 4px;">{(report.valid/report.total_citations*100) if report.total_citations > 0 else 0:.1f}% of total</div>',
        f'</div>',
        
        f'<div style="flex: 1; min-width: 200px; padding: 16px; background: {density_color}; border-radius: 8px; border: 1px solid var(--border);">',
        f'<div style="font-size: 14px; color: {density_text_color}; margin-bottom: 4px;">Citation Density</div>',
        f'<div style="font-size: 28px; font-weight: 700; color: {density_text_color};">{report.citation_density:.2f}</div>',
        f'<div style="font-size: 12px; color: {density_text_color}; margin-top: 4px;">per 100 words (target: ≥2.0)</div>',
        f'</div>',
        
        f'<div style="flex: 1; min-width: 200px; padding: 16px; background: {relevance_color}; border-radius: 8px; border: 1px solid var(--border);">',
        f'<div style="font-size: 14px; color: {relevance_text_color}; margin-bottom: 4px;">Avg Relevance</div>',
        f'<div style="font-size: 28px; font-weight: 700; color: {relevance_text_color};">{report.avg_relevance_score*100:.0f}%</div>',
        f'<div style="font-size: 12px; color: {relevance_text_color}; margin-top: 4px;">semantic support score</div>',
        f'</div>',
        
        '</div>',
        
        f'<div class="muted" style="margin-bottom: 24px;">',
        f'Total words: {report.total_words:,} | ',
        f'Citations with semantic validation: {len([c for c in report.validated_citations if c.relevance_score is not None])}',
        f'</div>',
    ]

    if report.validated_citations:
        body.append('<h3 style="margin-top: 32px; margin-bottom: 16px;">Individual Citation Scores</h3>')
        body.append('<div style="background: var(--surface); padding: 24px; border-radius: 8px; border: 1px solid var(--border); margin-bottom: 24px;">')
        body.append('<p class="muted" style="margin-bottom: 20px;">Semantic relevance score indicates how well each citation supports its claim (0-100%).</p>')
        body.append('<div style="display:flex;flex-direction:column;gap:16px;">')
        
        for i, validated in enumerate(report.validated_citations, 1):
            if validated.relevance_score is not None:
                score_pct = validated.relevance_score * 100
                if score_pct >= 80:
                    score_color = "#d4edda"
                    score_text_color = "#155724"
                    score_icon = "✓"
                elif score_pct >= 50:
                    score_color = "#fff3cd"
                    score_text_color = "#856404"
                    score_icon = "~"
                else:
                    score_color = "#f8d7da"
                    score_text_color = "#721c24"
                    score_icon = "✗"
            else:
                score_color = "#e9ecef"
                score_text_color = "#6c757d"
                score_icon = "?"
                score_pct = 0
            
            doc_id = validated.citation.document_id
            page_no = validated.citation.page_no
            info = doc_info.get(doc_id, {'title': doc_id, 'source': 'Unknown'})
            pdf_link = f'/pdf/{doc_id}#page={page_no}'

            body.append(f'<div style="background: white; padding: 20px; border-radius: 8px; border: 1px solid var(--border); border-left: 4px solid {score_text_color}; box-shadow: var(--shadow-sm);">')
            body.append(f'<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; gap: 16px;">')
            body.append(f'<div style="flex: 1; min-width: 0;">')
            
            body.append(f'<div style="font-size: 15px; font-weight: 600; color: var(--text); margin-bottom: 4px;">Citation #{i}: {safe_html(info["title"])}</div>')
            
            body.append(f'<div style="font-size: 13px; color: var(--text-light);">')
            body.append(f'{safe_html(info["source"])} • Doc ID: <code style="font-size: 12px; background: var(--surface); padding: 2px 4px; border-radius: 3px;">{safe_html(doc_id)}</code>')
            body.append(f' • Page {page_no}')
            body.append(f' • <a href="{pdf_link}" target="_blank" style="color: var(--accent); font-weight: 600;">View PDF</a>')
            body.append(f'</div>')
            
            
            body.append(f'</div>')
            
            if validated.relevance_score is not None:
                body.append(f'<div style="text-align: right; flex-shrink: 0;">')
                body.append(f'<div style="display: inline-block; padding: 8px 14px; background: {score_color}; color: {score_text_color}; border-radius: 6px; font-weight: 600; font-size: 15px; white-space: nowrap;">')
                body.append(f'{score_icon} {score_pct:.0f}%')
                body.append(f'</div>')
                if validated.verdict:
                    body.append(f'<div style="font-size: 11px; color: var(--muted); margin-top: 6px;">{validated.verdict}</div>')
                body.append(f'</div>')
            else:
                body.append(f'<div style="padding: 8px 14px; background: {score_color}; color: {score_text_color}; border-radius: 6px; font-size: 13px; white-space: nowrap;">No score</div>')
            
            body.append(f'</div>')
            
            if validated.context:
                context_preview = validated.context[:150] + "..." if len(validated.context) > 150 else validated.context
                body.append(f'<div style="font-size: 13px; color: var(--text-light); margin-top: 12px; padding: 12px; background: var(--surface); border-radius: 6px; border: 1px solid var(--border-light); line-height: 1.6;">')
                body.append(f'<strong style="color: var(--text);">Claim:</strong> {safe_html(context_preview)}')
                body.append(f'</div>')
            
            if validated.explanation and validated.relevance_score is not None and validated.relevance_score < 0.8:
                body.append(f'<div style="font-size: 12px; color: {score_text_color}; margin-top: 12px; padding: 12px; background: {score_color}; border-radius: 6px; line-height: 1.5;">')
                body.append(f'<strong>Analysis:</strong> {safe_html(validated.explanation)}')
                body.append(f'</div>')
            
            body.append(f'</div>')
        
        body.append('</div>')
        body.append('</div>')

    # Critical issues (document/page not found)
    if report.issues:
        body.append('<h3 style="color: #721c24; margin-top: 24px; margin-bottom: 16px;">Critical Issues</h3>')
        body.append('<div style="background: #f8d7da; padding: 20px; border-radius: 8px; border-left: 4px solid #721c24; margin-bottom: 24px;">')
        body.append('<div style="display:flex;flex-direction:column;gap:16px;">')
        for issue in report.issues:
            body.append(f'<div style="background: white; padding: 16px; border-radius: 6px; border: 1px solid rgba(114, 28, 36, 0.2);">')
            body.append(f'<div style="margin-bottom: 8px;"><strong style="color: #721c24;">Citation:</strong> <code style="background: #f8d7da; padding: 3px 6px; border-radius: 3px;">{safe_html(issue.citation.raw)}</code></div>')
            body.append(f'<div style="color: #721c24; font-weight: 600; margin-bottom: 8px;"><strong>Issue:</strong> {safe_html(issue.issue)}</div>')
            if issue.context:
                body.append(f'<div style="margin-top: 12px; padding: 12px; background: #f8d7da; border-radius: 4px; font-size: 13px; line-height: 1.6;">')
                body.append(f'<strong>Context:</strong> ...{safe_html(issue.context)}...')
                body.append('</div>')
            body.append('</div>')
        body.append('</div>')
    

    

    if not report.issues and not report.warnings:
        body.append('<div style="background: #d4edda; padding: 16px; border-radius: 8px; border-left: 4px solid #155724; margin-top: 16px;">')
        body.append('<strong style="color: #155724;">All citations validated successfully!</strong>')
        body.append('<p class="muted" style="margin: 8px 0 0 0;">All citations reference existing documents and pages, and appear semantically relevant.</p>')
        body.append('</div>')

    body.append('<div style="margin-top: 16px;">')
    body.append(f'<a class="btn secondary" href="{url_for("view_page", slug=slug)}">← Back to Page</a>')
    body.append(f'<a class="btn secondary" href="{url_for("chunks_page", slug=slug)}" style="margin-left:12px;">View Chunks</a>')
    body.append('</div>')
    body.append('</div>')
    return render("".join(body))



    
            
    

    
    
        
        
        
        
        
        
        

        
            
            
            
            
            
            
        

    
    



@app.route("/wiki/<slug>/chunks")
def chunks_page(slug: str):
    """Visualize the chunks used to generate a wiki page"""
    md_path = ROOT / "data/wiki" / f"{slug}.md"
    if not md_path.exists():
        return render('<div class="error">Page not found</div>')

    meta_path = ROOT / "data/wiki" / f"{slug}.meta.json"
    if not meta_path.exists():
        return render('<div class="error">Metadata not found. This page may have been generated before chunk visualization was added.</div>')
    
    import json
    with open(meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    chunks = metadata.get('chunks', [])
    if not chunks:
        return render('<div class="error">No chunk data found in metadata. Please regenerate this wiki page to enable chunk visualization.</div>')
    
    topic = metadata.get('topic', slug.replace('_', ' '))
    num_sources = metadata.get('num_sources', len(chunks))
    context_length = metadata.get('context_length', 0)
    
    from collections import defaultdict
    doc_chunks = defaultdict(list)
    for chunk in chunks:
        doc_chunks[chunk['doc_id']].append(chunk)
    
    total_chunks = len(chunks)
    unique_docs = len(doc_chunks)
    avg_chunks_per_doc = total_chunks / unique_docs if unique_docs > 0 else 0
    
    body = [
        '<div class="container" style="padding-top:48px;padding-bottom:64px;">',
        '<div class="card">',
        f'<h1 style="font-size:32px;font-weight:700;color:var(--text);margin-bottom:12px;">Chunks Used: {topic}</h1>',
        '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:24px;">',
        f'<a class="btn secondary small" href="{url_for("view_page", slug=slug)}">← Back to Page</a>',
        f'<a class="btn secondary small" href="{url_for("validate_page", slug=slug)}">Validate Citations</a>',
        '</div>',
        
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:32px;">',
        f'<div style="background:var(--surface);padding:20px;border-radius:var(--radius-md);border:1px solid var(--border);">',
        f'<div style="font-size:24px;font-weight:700;color:var(--text);">{total_chunks}</div>',
        '<div style="font-size:14px;color:var(--text-light);margin-top:4px;">Total Chunks</div>',
        '</div>',
        f'<div style="background:var(--surface);padding:20px;border-radius:var(--radius-md);border:1px solid var(--border);">',
        f'<div style="font-size:24px;font-weight:700;color:var(--text);">{unique_docs}</div>',
        '<div style="font-size:14px;color:var(--text-light);margin-top:4px;">Unique Documents</div>',
        '</div>',
        f'<div style="background:var(--surface);padding:20px;border-radius:var(--radius-md);border:1px solid var(--border);">',
        f'<div style="font-size:24px;font-weight:700;color:var(--text);">{avg_chunks_per_doc:.1f}</div>',
        '<div style="font-size:14px;color:var(--text-light);margin-top:4px;">Avg Chunks/Doc</div>',
        '</div>',
        f'<div style="background:var(--surface);padding:20px;border-radius:var(--radius-md);border:1px solid var(--border);">',
        f'<div style="font-size:24px;font-weight:700;color:var(--text);">{context_length:,}</div>',
        '<div style="font-size:14px;color:var(--text-light);margin-top:4px;">Total Characters</div>',
        '</div>',
        '</div>',
        
        '<h2 style="font-size:24px;font-weight:600;margin-top:32px;margin-bottom:16px;">Chunks per Document</h2>',
        '<div style="background:var(--surface);padding:24px;border-radius:var(--radius-md);border:1px solid var(--border);margin-bottom:32px;">',
    ]
    
    sorted_docs = sorted(doc_chunks.items(), key=lambda x: len(x[1]), reverse=True)
    
    for doc_id, doc_chunk_list in sorted_docs[:20]:  # Show top 20 documents
        chunk_count = len(doc_chunk_list)
        max_chunks = max(len(chunks) for chunks in doc_chunks.values())
        bar_width = (chunk_count / max_chunks) * 100 if max_chunks > 0 else 0
        
        first_chunk = doc_chunk_list[0]
        doc_title = first_chunk.get('title', doc_id)
        doc_source = first_chunk.get('source', 'Unknown')
        
        body.append('<div style="margin-bottom:16px;">')
        body.append(f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">')
        body.append(f'<div style="flex:1;min-width:0;">')
        body.append(f'<div style="font-weight:600;color:var(--text);font-size:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{safe_html(doc_title)}</div>')
        body.append(f'<div style="font-size:12px;color:var(--text-light);">{safe_html(doc_source)}</div>')
        body.append('</div>')
        body.append(f'<div style="font-weight:600;color:var(--text);margin-left:16px;min-width:60px;text-align:right;">{chunk_count}</div>')
        body.append('</div>')
        body.append(f'<div style="background:var(--border-light);height:8px;border-radius:4px;overflow:hidden;">')
        body.append(f'<div style="background:var(--accent);height:100%;width:{bar_width}%;transition:width 0.3s;"></div>')
        body.append('</div>')
        body.append('</div>')
    
    if len(sorted_docs) > 20:
        body.append(f'<div style="text-align:center;color:var(--text-light);font-size:14px;margin-top:16px;">...and {len(sorted_docs) - 20} more documents</div>')
    
    body.append('</div>')
    
    body.append('<h2 style="font-size:24px;font-weight:600;margin-top:32px;margin-bottom:16px;">All Chunks</h2>')
    body.append('<div style="display:flex;flex-direction:column;gap:20px;margin-bottom:32px;">')
    
    for i, chunk in enumerate(chunks):
        doc_id = chunk['doc_id']
        doc_title = chunk.get('title', doc_id)
        doc_source = chunk.get('source', 'Unknown')
        page_no = chunk.get('page_no')
        text_preview = chunk.get('text_preview', '')
        relevance = chunk.get('relevance_score', 0)
        semantic = chunk.get('semantic_score', 0)
        bm25 = chunk.get('bm25_score', 0)
        citation_key = chunk.get('citation_key', '')
        
        if relevance >= 0.7:
            rel_color = '#d4edda'
            rel_text_color = '#155724'
        elif relevance >= 0.4:
            rel_color = '#fff3cd'
            rel_text_color = '#856404'
        else:
            rel_color = '#f8d7da'
            rel_text_color = '#721c24'
        
        body.append('<div style="background:white;padding:24px;border-radius:var(--radius-md);border:1px solid var(--border);box-shadow:var(--shadow-sm);margin-bottom:0;">')
        
        body.append(f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px;gap:16px;">')
        body.append(f'<div style="flex:1;min-width:0;">')
        body.append(f'<div style="font-weight:600;color:var(--text);font-size:16px;margin-bottom:6px;">Chunk #{i+1}: {safe_html(doc_title)}</div>')
        body.append(f'<div style="font-size:13px;color:var(--text-light);margin-bottom:8px;">{safe_html(doc_source)} • Doc ID: <code style="font-size:12px;">{doc_id}</code>')
        if page_no:
            body.append(f' • Page {page_no}')
            pdf_link = f'/pdf/{doc_id}#page={page_no}'
            body.append(f' • <a href="{pdf_link}" target="_blank" style="color:var(--accent);font-weight:600;">View PDF</a>')
        body.append('</div>')
        body.append('</div>')
        
        body.append(f'<div style="display:flex;gap:8px;flex-wrap:wrap;flex-shrink:0;">')
        body.append(f'<div style="padding:6px 12px;background:{rel_color};color:{rel_text_color};border-radius:4px;font-size:12px;font-weight:600;white-space:nowrap;">Relevance: {relevance:.2f}</div>')
        body.append(f'<div style="padding:6px 12px;background:var(--surface);color:var(--text-light);border-radius:4px;font-size:12px;white-space:nowrap;">Semantic: {semantic:.2f}</div>')
        body.append(f'<div style="padding:6px 12px;background:var(--surface);color:var(--text-light);border-radius:4px;font-size:12px;white-space:nowrap;">BM25: {bm25:.2f}</div>')
        body.append('</div>')
        body.append('</div>')
        
        if citation_key:
            body.append(f'<div style="font-size:12px;color:var(--muted);margin-top:12px;margin-bottom:12px;padding-top:12px;border-top:1px solid var(--border-light);">')
            body.append(f'<strong>Citation Key:</strong> <code style="background:var(--surface);padding:2px 6px;border-radius:3px;">{safe_html(citation_key)}</code>')
            body.append('</div>')
        
        if text_preview:
            body.append(f'<div style="background:var(--surface);padding:16px;border-radius:6px;border:1px solid var(--border-light);margin-top:12px;font-size:14px;color:var(--text-light);line-height:1.7;font-family:monospace;">{safe_html(text_preview)}</div>')
        
        body.append('</div>')
    
    body.append('</div>')
    body.append('</div>')
    body.append('</div>')
    return render("".join(body))


@app.route("/manage-featured", methods=["GET", "POST"])
def manage_featured():
    """Manage which wiki pages are featured on the homepage"""
    import json
    
    featured_config_path = ROOT / "data" / "wiki" / "featured_pages.json"
    
    if request.method == "POST":
        selected_slugs = request.form.getlist('featured')
        try:
            config = {
                "_comment": "List wiki page slugs (filenames without .md) to feature on the homepage. If empty, shows most recent 6 pages.",
                "featured": selected_slugs
            }
            with open(featured_config_path, 'w') as f:
                json.dump(config, f, indent=2)
            flash(f"Featured pages updated! {len(selected_slugs)} page(s) selected.")
            return redirect(url_for('manage_featured'))
        except Exception as e:
            flash(f"Error saving configuration: {e}")
    
    featured_slugs = []
    if featured_config_path.exists():
        try:
            with open(featured_config_path, 'r') as f:
                config = json.load(f)
                featured_slugs = config.get('featured', [])
        except Exception:
            pass
    
    all_pages = list_generated_pages()
    pages_by_time = sorted(all_pages, key=lambda p: p.stat().st_mtime, reverse=True)
    
    body = [
        '<div class="container" style="padding-top:40px;">',
        '<div class="card">',
        '<h2>Manage Featured Pages</h2>',
        '<p class="muted" style="margin-bottom:24px;">Select which wiki pages to display on the homepage. If none are selected, the 6 most recently modified pages will be shown automatically.</p>',
        
        '<form method="post">',
        '<div style="background:var(--surface);padding:24px;border-radius:12px;border:1px solid var(--border);margin-bottom:24px;">',
        '<h3 style="margin-top:0;font-size:18px;margin-bottom:16px;">Available Pages</h3>',
        '<div style="max-height:500px;overflow-y:auto;">',
    ]
    
    for page in pages_by_time:
        slug = page.stem
        title = slug.replace('_', ' ')
        checked = 'checked' if slug in featured_slugs else ''
        mod_time = page.stat().st_mtime
        import datetime
        mod_date = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M')
        
        body.append(
            f'<div style="padding:12px;margin-bottom:8px;background:white;border:1px solid var(--border);border-radius:8px;display:flex;align-items:center;gap:12px;">'
            f'<input type="checkbox" name="featured" value="{slug}" id="page_{slug}" {checked} style="width:20px;height:20px;cursor:pointer;">'
            f'<label for="page_{slug}" style="flex:1;cursor:pointer;margin:0;">'
            f'<div style="font-weight:600;color:var(--text);">{title}</div>'
            f'<div class="muted" style="font-size:13px;">Modified: {mod_date}</div>'
            f'</label>'
            f'</div>'
        )
    
    body.extend([
        '</div>',
        '</div>',
        
        '<div style="display:flex;gap:12px;justify-content:space-between;align-items:center;">',
        '<div class="muted" style="font-size:14px;">',
        f'Currently featuring: {len(featured_slugs)} page(s)' if featured_slugs else 'Currently showing most recent 6 pages',
        '</div>',
        '<div style="display:flex;gap:12px;">',
        f'<a class="btn secondary" href="{url_for("index")}">Cancel</a>',
        '<button type="submit" class="btn">Save Featured Pages</button>',
        '</div>',
        '</div>',
        '</form>',
        '</div>',
        '</div>',
    ])
    
    return render("".join(body))


def get_page_metadata(page_path: Path) -> dict:
    """Get metadata for a wiki page"""
    meta_path = page_path.parent / f"{page_path.stem}.meta.json"
    if meta_path.exists():
        try:
            import json
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                return metadata
        except Exception:
            pass
    return {}


def get_page_length_mode(page_path: Path) -> str:
    """Get the length mode (short, medium, long, exhaustive) from page metadata"""
    metadata = get_page_metadata(page_path)
    return metadata.get('length_mode', 'medium')  # Default to medium if missing


def get_page_generation_info(page_path: Path) -> dict:
    """Get generation info (max_chunks, diversity_mode, length) for display"""
    metadata = get_page_metadata(page_path)
    
    length_mode = metadata.get('length_mode', 'medium')
    
    max_chunks = metadata.get('max_chunks')
    if max_chunks is None:
        length_config = {
            "short": 30,
            "medium": 40,
            "long": 80,
            "exhaustive": 500
        }
        max_chunks = length_config.get(length_mode, 40)
    
    diversity_mode = metadata.get('diversity_mode')
    
    return {
        'length': length_mode,
        'max_chunks': max_chunks,
        'diversity': diversity_mode
    }


@app.route("/wiki-hub") 
def wiki_hub():
    """List all exhaustive wiki pages in a horizontal card layout"""
    all_pages = list_generated_pages()
    exhaustive_pages = [p for p in all_pages if get_page_length_mode(p) == 'exhaustive']
    exhaustive_pages.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    body = [
        '<div class="container" style="padding-top:48px;padding-bottom:64px;">',
        '<div style="text-align:center;margin-bottom:48px;">',
        '<h1 style="font-family:\'Playfair Display\', serif;font-size:42px;font-weight:800;color:var(--text);margin-bottom:16px;">Wiki Hub</h1>',
        '<p class="muted" style="font-size:18px;max-width:600px;margin:0 auto;">Comprehensive exhaustive intelligence reports with maximum detail and depth.</p>',
        '</div>',
    ]
    
    if not exhaustive_pages:
        body.append('<div class="card" style="text-align:center;"><div class="muted">No exhaustive wiki pages yet.</div></div>')
    else:
        import datetime
        
        for page in exhaustive_pages:
            slug = page.stem
            title = slug.replace('_', ' ')
            mod_time = page.stat().st_mtime
            date_str = datetime.datetime.fromtimestamp(mod_time).strftime('%B %d, %Y')
            
            try:
                text = page.read_text(encoding='utf-8')
                lines = text.split('\n')
                content_start = 0
                for i, line in enumerate(lines):
                    if line.strip() and not line.startswith('#'):
                        content_start = i
                        break
                
                snippet = " ".join(lines[content_start:content_start+5])[:250] + "..."
            except Exception:
                snippet = "No preview available."

            body.append(f'''
            <a href="{url_for('view_page', slug=slug)}" class="card-horizontal">
                <div class="content">
                    <h3>{safe_html(title)}</h3>
                    <div class="meta">Generated on {date_str}</div>
                    <div class="excerpt">{safe_html(snippet)}</div>
                </div>
                <div class="actions">
                    <span class="btn secondary small">Read Article</span>
                </div>
            </a>
            ''')
            
    body.append('</div>')
    return render("".join(body))


@app.route("/community-wikis")
def community_wikis():
    """Feed of all non-exhaustive wikis, ordered by latest to newest, with search"""
    q = (request.args.get('q') or "").strip().lower()
    
    all_pages = list_generated_pages()
    community_pages = [p for p in all_pages if get_page_length_mode(p) != 'exhaustive']
    community_pages.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    if q:
        community_pages = [p for p in community_pages if q in p.stem.lower().replace('_', ' ')]
    
    body = [
        '<div class="container" style="padding-top:48px;padding-bottom:64px;">',
        '<div style="max-width:800px;margin:0 auto;">',
        
        '<div style="margin-bottom:40px;">',
        '<h1 style="font-family:\'Playfair Display\', serif;font-size:36px;font-weight:800;color:var(--text);margin-bottom:24px;">Community Wikis</h1>',
        
        '<form method="get" action="/community-wikis" style="display:flex;gap:12px;">',
        f'<input class="search-input" type="text" name="q" placeholder="Search wikis generated by the community..." value="{safe_html(q)}" style="flex:1;padding:16px 24px;font-size:16px;">',
        '<button class="btn" type="submit">Search</button>',
        '</form>',
        '</div>',
        
        '<div style="display:flex;flex-direction:column;gap:16px;">',
    ]
    
    if not community_pages:
        body.append('<div class="card" style="text-align:center;padding:40px;"><div class="muted">No community wikis found.</div></div>')
    else:
        import datetime
        
        for page in community_pages:
            slug = page.stem
            title = slug.replace('_', ' ')
            mod_time = page.stat().st_mtime
            date_str = datetime.datetime.fromtimestamp(mod_time).strftime('%b %d, %Y • %H:%M')
            
            gen_info = get_page_generation_info(page)
            length_display = gen_info['length'].capitalize()
            max_chunks_display = f"{gen_info['max_chunks']} chunks"
            diversity_display = gen_info['diversity'].capitalize() if gen_info['diversity'] and gen_info['diversity'] != 'Unknown' else None
            
            try:
                text = page.read_text(encoding='utf-8')
                import re
                clean_text = re.sub(r'^#+ .*$', '', text, flags=re.MULTILINE)
                clean_text = re.sub(r'\[.*?\]', '', clean_text)
                snippet = clean_text.strip()[:200].replace('\n', ' ') + "..."
            except Exception:
                snippet = ""

            body.append(f'''
            <div class="card" style="padding:24px;">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">
                    <div style="flex:1;">
                        <h2 style="font-size:22px;margin:0 0 8px 0;"><a href="{url_for('view_page', slug=slug)}" style="color:var(--text);text-decoration:none;">{safe_html(title)}</a></h2>
                        <div class="muted" style="font-size:13px;margin-bottom:12px;">{date_str}</div>
                        <div style="display:flex;gap:6px;flex-wrap:wrap;">
                            <span class="tag" style="background:var(--accent-light);color:var(--accent);border-color:var(--accent);font-size:11px;padding:4px 10px;">{safe_html(length_display)}</span>
                            <span class="tag" style="background:var(--surface);color:var(--text);border-color:var(--border);font-size:11px;padding:4px 10px;">{safe_html(max_chunks_display)}</span>
                            {f'<span class="tag" style="background:var(--surface);color:var(--text);border-color:var(--border);font-size:11px;padding:4px 10px;">{safe_html(diversity_display)}</span>' if diversity_display else ''}
                        </div>
                    </div>
                    <a href="{url_for('view_page', slug=slug)}" class="btn secondary small" style="margin-left:16px;flex-shrink:0;">View</a>
                </div>
                <div style="font-size:15px;color:var(--text-light);line-height:1.6;margin-top:12px;">
                    {safe_html(snippet)}
                </div>
            </div>
            ''')
            
    body.append('</div>') # End Feed
    body.append('</div>') # End Container
    body.append('</div>')
    
    return render("".join(body))


if __name__ == "__main__":
    port = 5053
    print(f"FOIA AI Wiki UI running at http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=True, use_reloader=False)
