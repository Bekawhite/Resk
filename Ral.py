"""
Advanced Document to Excel Converter - Universal Data Extractor
Handles ANY content type: tables, prose, lists, forms, mixed formats
Intelligently structures unstructured data into meaningful tables
"""

import streamlit as st
import pandas as pd
import tempfile
import os
import re
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from datetime import datetime
import io
import base64
import hashlib
import time
from collections import defaultdict, Counter
import itertools
import json

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Universal Document to Excel Converter",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# ADVANCED NLP IMPORTS (with fallbacks)
# ============================================================================

# Try to import advanced NLP libraries
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    NLTK_AVAILABLE = True
    # Download required NLTK data if not present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    try:
        nltk.data.find('chunkers/maxent_ne_chunker')
    except LookupError:
        nltk.download('maxent_ne_chunker', quiet=True)
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False

# ============================================================================
# EXISTING IMPORTS (from your original code)
# ============================================================================

# Try to import scikit-learn with proper error handling
try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Define fallback DBSCAN-like function
    class DBSCAN:
        def __init__(self, eps=0.5, min_samples=2):
            self.eps = eps
            self.min_samples = min_samples
        
        def fit_predict(self, X):
            # Simple fallback: return all points as one cluster
            return np.zeros(len(X), dtype=int)
    
    class StandardScaler:
        def fit_transform(self, X):
            return X

# Try to import scipy
try:
    from scipy import stats
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Simple fallback for find_peaks
    def find_peaks(x, height=None, distance=None):
        peaks = []
        for i in range(1, len(x)-1):
            if x[i] > x[i-1] and x[i] > x[i+1]:
                if height is None or x[i] > height:
                    peaks.append(i)
        return np.array(peaks), {}

# Computer Vision and OCR imports
try:
    import cv2
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pdf2image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

# PDF processing imports
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

# ============================================================================
# CUSTOM CSS STYLES (keeping your existing styles)
# ============================================================================

def load_css():
    """Load custom CSS styles"""
    st.markdown("""
    <style>
        /* Main container styling */
        .main {
            padding: 0rem 1rem;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: white;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            padding: 10px 20px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e9ecef;
            border-color: #adb5bd;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4CAF50 !important;
            color: white !important;
            border-color: #4CAF50 !important;
        }
        
        /* Box styling */
        .success-box {
            background-color: #d4edda;
            padding: 20px;
            border-radius: 10px;
            border-left: 6px solid #28a745;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .info-box {
            background-color: #d1ecf1;
            padding: 20px;
            border-radius: 10px;
            border-left: 6px solid #17a2b8;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .warning-box {
            background-color: #fff3cd;
            padding: 20px;
            border-radius: 10px;
            border-left: 6px solid #ffc107;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .error-box {
            background-color: #f8d7da;
            padding: 20px;
            border-radius: 10px;
            border-left: 6px solid #dc3545;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Column selector styling */
        .column-selector {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #dee2e6;
            margin: 10px 0;
        }
        
        /* Metric cards */
        .metric-card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #dee2e6;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: transform 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 28px;
            font-weight: bold;
            color: #4CAF50;
        }
        .metric-label {
            font-size: 14px;
            color: #6c757d;
            margin-top: 5px;
        }
        
        /* Progress steps */
        .progress-step {
            display: flex;
            align-items: center;
            margin: 10px 0;
        }
        .step-number {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background-color: #4CAF50;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 10px;
            font-weight: bold;
        }
        .step-text {
            color: #495057;
        }
        
        /* Feature cards */
        .feature-card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #dee2e6;
            margin: 10px 0;
            transition: all 0.3s ease;
        }
        .feature-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-color: #4CAF50;
        }
        .feature-icon {
            font-size: 24px;
            margin-bottom: 10px;
        }
        .feature-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .feature-description {
            font-size: 14px;
            color: #6c757d;
        }
        
        /* Status badges */
        .badge-success {
            background-color: #d4edda;
            color: #155724;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
        }
        .badge-warning {
            background-color: #fff3cd;
            color: #856404;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
        }
        .badge-info {
            background-color: #d1ecf1;
            color: #0c5460;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
        }
        .badge-primary {
            background-color: #cfe2ff;
            color: #084298;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
        }
        
        /* Content type badge */
        .content-type {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin: 10px 0;
        }
        .type-table {
            background-color: #d4edda;
            color: #155724;
        }
        .type-prose {
            background-color: #cfe2ff;
            color: #084298;
        }
        .type-list {
            background-color: #fff3cd;
            color: #856404;
        }
        .type-mixed {
            background-color: #e2d5f1;
            color: #6610f2;
        }
        .type-form {
            background-color: #f8d7da;
            color: #721c24;
        }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE MANAGEMENT (enhanced)
# ============================================================================

def init_session_states():
    """Initialize all session state variables with enhanced tracking"""
    defaults = {
        'pdf_uploaded': False,
        'tables_data': {},
        'pdf_metadata': {},
        'selected_tables': {},
        'column_selections': {},
        'row_selections': {},
        'all_columns': {},
        'extraction_mode': "Universal (Auto-detect)",
        'processing_history': [],
        'current_file_hash': None,
        'extraction_stats': {},
        'ocr_language': 'eng',
        'debug_mode': False,
        'detected_regions': [],
        'pattern_confidence': {},
        'cv_available': CV_AVAILABLE,
        'ml_available': SKLEARN_AVAILABLE and SCIPY_AVAILABLE,
        'pdf_available': PDFPLUMBER_AVAILABLE or CAMELOT_AVAILABLE,
        'tesseract_available': TESSERACT_AVAILABLE,
        'nlp_available': SPACY_AVAILABLE or NLTK_AVAILABLE or TEXTBLOB_AVAILABLE,
        'content_types': {},  # Track detected content type per page
        'entity_extractions': {},  # Store extracted entities
        'relationship_graphs': {},  # Store relationships between entities
        'structured_prose': {},  # Store prose converted to structure
        'confidence_scores': {},  # Confidence scores for extractions
        'extraction_method_used': {},  # Which method was used
        'manual_corrections': {},  # Store user manual corrections
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ============================================================================
# ENHANCED UTILITY FUNCTIONS
# ============================================================================

def get_file_hash(file_content: bytes) -> str:
    """Generate hash for file content"""
    return hashlib.md5(file_content).hexdigest()

def convert_to_proper_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert DataFrame columns to proper data types for Excel compatibility
    Enhanced to handle more data types and patterns
    """
    df_converted = df.copy()
    
    for col in df_converted.columns:
        # Skip if column is empty
        if df_converted[col].isna().all():
            continue
        
        # Try to convert to numeric (with enhanced pattern recognition)
        try:
            # First, clean the data: remove currency symbols and commas
            if df_converted[col].dtype == 'object':
                # Check if column contains currency patterns
                sample = df_converted[col].dropna().iloc[0] if not df_converted[col].dropna().empty else ""
                if isinstance(sample, str):
                    # Enhanced currency detection
                    currency_symbols = ['$', '€', '£', '¥', '₹', '₽', '₿', '₩', '₦', '₱', '﷼', '៛', '₫', '฿', '₲', '₴', '₸', '₺', '₼', '₡', '₭', '₮', '₦', '₱', '₲', '₴', '₸', '₺', '₼', '₡', '₢', '₣', '₤', '₥', '₧', '₨', '₪', '₵', '₷', '₶', '₳', '₲']
                    if any(symbol in sample for symbol in currency_symbols):
                        # Remove currency symbols and commas, then try numeric conversion
                        cleaned = df_converted[col].astype(str).str.replace(r'[$€£¥₹₽₿₩₦₱﷼៛₫฿₲₴₸₺₼₡₭₮₢₣₤₥₧₨₪₵₷₶₳]', '', regex=True)
                        cleaned = cleaned.str.replace(',', '')
                        cleaned = cleaned.str.replace(' ', '')
                        cleaned = cleaned.str.replace(r'\(', '-', regex=True)  # Handle negative in parentheses
                        cleaned = cleaned.str.replace(r'\)', '', regex=True)
                        numeric_col = pd.to_numeric(cleaned, errors='coerce')
                        if not numeric_col.isna().all():
                            df_converted[col] = numeric_col
                            continue
                    
                    # Check for percentage
                    if '%' in sample:
                        cleaned = df_converted[col].astype(str).str.replace('%', '')
                        cleaned = cleaned.str.replace(',', '')
                        numeric_col = pd.to_numeric(cleaned, errors='coerce') / 100
                        if not numeric_col.isna().all():
                            df_converted[col] = numeric_col
                            continue
                    
                    # Check for fractions (e.g., 1/2, 3/4)
                    if '/' in sample and not any(c.isalpha() for c in sample):
                        try:
                            # Try to convert fractions
                            def fraction_to_float(x):
                                if isinstance(x, str) and '/' in x:
                                    parts = x.split('/')
                                    if len(parts) == 2 and all(p.strip().isdigit() for p in parts):
                                        return float(parts[0]) / float(parts[1])
                                return x
                            
                            converted = df_converted[col].apply(fraction_to_float)
                            numeric_col = pd.to_numeric(converted, errors='coerce')
                            if not numeric_col.isna().all():
                                df_converted[col] = numeric_col
                                continue
                        except:
                            pass
            
            # Direct numeric conversion
            numeric_col = pd.to_numeric(df_converted[col], errors='coerce')
            if not numeric_col.isna().all():  # If at least some values converted successfully
                df_converted[col] = numeric_col
                continue
        except:
            pass
        
        # Try to convert to datetime (enhanced)
        try:
            if df_converted[col].dtype == 'object':
                # Try multiple date formats
                date_formats = [
                    '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%m-%d-%Y',
                    '%Y/%m/%d', '%d.%m.%Y', '%m.%d.%Y', '%d %b %Y', '%d %B %Y',
                    '%b %d, %Y', '%B %d, %Y', '%d-%b-%Y', '%d-%B-%Y',
                    '%Y%m%d', '%d%m%Y', '%m%d%Y'
                ]
                
                for fmt in date_formats:
                    try:
                        date_col = pd.to_datetime(df_converted[col], format=fmt, errors='coerce')
                        if not date_col.isna().all():
                            df_converted[col] = date_col
                            break
                    except:
                        continue
                
                # If no format worked, try infer
                if df_converted[col].dtype == 'object':
                    date_col = pd.to_datetime(df_converted[col], errors='coerce', infer_datetime_format=True)
                    if not date_col.isna().all():
                        df_converted[col] = date_col
        except:
            pass
        
        # Try to convert to boolean
        try:
            if df_converted[col].dtype == 'object':
                bool_vals = df_converted[col].astype(str).str.lower().str.strip()
                if bool_vals.isin(['true', 'false', 'yes', 'no', 'y', 'n', '1', '0', 't', 'f']).all():
                    # Map to boolean
                    mapping = {
                        'true': True, 'false': False,
                        'yes': True, 'no': False,
                        'y': True, 'n': False,
                        '1': True, '0': False,
                        't': True, 'f': False
                    }
                    df_converted[col] = bool_vals.map(mapping).astype(bool)
        except:
            pass
        
        # If all else fails, ensure it's string type for Excel compatibility
        if df_converted[col].dtype == 'object':
            df_converted[col] = df_converted[col].astype(str).replace('nan', '').replace('None', '')
    
    return df_converted

def create_download_link(df: pd.DataFrame, filename: str) -> str:
    """Create a download link for dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

def calculate_extraction_stats(tables_data: Dict) -> Dict:
    """Calculate statistics about extracted tables"""
    stats = {
        'total_pages': len(tables_data),
        'total_tables': sum(len(tables) for tables in tables_data.values()),
        'total_rows': 0,
        'total_columns': 0,
        'avg_rows_per_table': 0,
        'avg_columns_per_table': 0,
        'max_rows': 0,
        'max_columns': 0,
        'page_with_most_tables': 0,
        'most_tables_count': 0,
        'content_types': {},
        'extraction_methods': Counter(),
        'avg_confidence': 0
    }
    
    all_rows = []
    all_cols = []
    all_confidences = []
    
    for page_num, tables in tables_data.items():
        page_table_count = len(tables)
        if page_table_count > stats['most_tables_count']:
            stats['most_tables_count'] = page_table_count
            stats['page_with_most_tables'] = page_num
        
        # Track content type
        content_type = st.session_state.content_types.get(page_num, 'unknown')
        stats['content_types'][content_type] = stats['content_types'].get(content_type, 0) + 1
        
        for table in tables:
            if isinstance(table, pd.DataFrame):
                rows, cols = table.shape
                all_rows.append(rows)
                all_cols.append(cols)
                stats['total_rows'] += rows
                stats['total_columns'] += cols
                stats['max_rows'] = max(stats['max_rows'], rows)
                stats['max_columns'] = max(stats['max_columns'], cols)
                
                # Track confidence
                conf_key = f"{page_num}_{len(all_rows)}"
                if conf_key in st.session_state.confidence_scores:
                    all_confidences.append(st.session_state.confidence_scores[conf_key])
                
                # Track method
                method = st.session_state.extraction_method_used.get(conf_key, 'unknown')
                stats['extraction_methods'][method] += 1
    
    if stats['total_tables'] > 0:
        stats['avg_rows_per_table'] = stats['total_rows'] / stats['total_tables']
        stats['avg_columns_per_table'] = stats['total_columns'] / stats['total_tables']
    
    if all_confidences:
        stats['avg_confidence'] = sum(all_confidences) / len(all_confidences)
    
    return stats

# ============================================================================
# CONTENT TYPE DETECTION
# ============================================================================

def detect_content_type(text_lines: List[str]) -> str:
    """
    Detect the type of content in the document
    Returns: 'table', 'prose', 'list', 'form', 'mixed'
    """
    if not text_lines:
        return 'unknown'
    
    # Features for detection
    total_lines = len(text_lines)
    if total_lines == 0:
        return 'unknown'
    
    # Check for table indicators
    table_indicators = 0
    prose_indicators = 0
    list_indicators = 0
    form_indicators = 0
    
    # Analyze line patterns
    line_lengths = [len(line) for line in text_lines if line.strip()]
    if not line_lengths:
        return 'unknown'
    
    avg_line_length = sum(line_lengths) / len(line_lengths)
    
    # Check for multiple spaces (common in tables)
    multi_space_lines = sum(1 for line in text_lines if re.search(r'\s{3,}', line))
    if multi_space_lines / total_lines > 0.3:
        table_indicators += 2
    
    # Check for delimiter patterns
    delimiter_patterns = ['\t', '|', ';', ',']
    for delim in delimiter_patterns:
        if any(delim in line for line in text_lines):
            table_indicators += 1
            break
    
    # Check for consistent column-like structure
    if total_lines >= 3:
        # Check if lines have similar structure (number of words, etc.)
        word_counts = [len(line.split()) for line in text_lines if line.strip()]
        if word_counts:
            # Calculate variance in word counts
            if len(word_counts) > 1:
                variance = np.var(word_counts) if len(word_counts) > 1 else 0
                if variance < 5:  # Low variance suggests tabular structure
                    table_indicators += 1
    
    # Check for prose indicators (sentences, punctuation)
    sentences = 0
    for line in text_lines:
        # Count sentence endings
        sentences += len(re.findall(r'[.!?]', line))
    
    if sentences / total_lines > 0.5:
        prose_indicators += 2
    
    # Check for long lines (prose typically has longer lines)
    if avg_line_length > 50:
        prose_indicators += 1
    
    # Check for list indicators
    list_markers = ['•', '-', '*', '·', '○', '●', '■', '→', '▪', '▫']
    numbered_pattern = r'^\s*\d+[.)]\s+'
    
    list_lines = 0
    for line in text_lines:
        if any(marker in line for marker in list_markers):
            list_lines += 1
        elif re.match(numbered_pattern, line):
            list_lines += 1
    
    if list_lines / total_lines > 0.3:
        list_indicators += 2
    
    # Check for form indicators (field: value pattern)
    field_pattern = r'^\s*[A-Za-z\s]+:\s*\S+'
    field_matches = sum(1 for line in text_lines if re.match(field_pattern, line))
    
    if field_matches / total_lines > 0.3:
        form_indicators += 2
    
    # Check for mixed content
    scores = {
        'table': table_indicators,
        'prose': prose_indicators,
        'list': list_indicators,
        'form': form_indicators
    }
    
    # Determine if mixed
    top_two = sorted(scores.values(), reverse=True)[:2]
    if len(top_two) >= 2 and top_two[1] > 0 and (top_two[0] - top_two[1]) < 2:
        return 'mixed'
    
    # Return the highest scoring type
    content_type = max(scores, key=scores.get)
    
    # If all scores are 0, default to prose
    if all(v == 0 for v in scores.values()):
        return 'prose'
    
    return content_type

# ============================================================================
# ENHANCED PROSE-TO-TABLE CONVERSION
# ============================================================================

def extract_entities_from_prose(text: str) -> List[Dict]:
    """
    Extract named entities and key information from prose text
    """
    entities = []
    
    # Try spaCy first (best results)
    if SPACY_AVAILABLE:
        try:
            nlp = spacy.load('en_core_web_sm')
            doc = nlp(text)
            
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'source': 'spacy'
                })
            
            # Also extract noun phrases as potential entities
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 4:  # Limit length
                    entities.append({
                        'text': chunk.text,
                        'label': 'NP',
                        'start': chunk.start_char,
                        'end': chunk.end_char,
                        'source': 'spacy_np'
                    })
            
            return entities
        except:
            pass
    
    # Fallback to NLTK
    if NLTK_AVAILABLE:
        try:
            sentences = sent_tokenize(text)
            for sent in sentences:
                words = word_tokenize(sent)
                pos_tags = pos_tag(words)
                
                # Extract named entities using simple patterns
                current_entity = []
                current_label = None
                
                for i, (word, tag) in enumerate(pos_tags):
                    # Person names (NNP followed by NNP)
                    if tag == 'NNP' and i < len(pos_tags)-1 and pos_tags[i+1][1] == 'NNP':
                        if current_label != 'PERSON':
                            if current_entity:
                                entities.append({
                                    'text': ' '.join(current_entity),
                                    'label': current_label or 'ENTITY',
                                    'source': 'nltk'
                                })
                            current_entity = [word]
                            current_label = 'PERSON'
                        else:
                            current_entity.append(word)
                    elif tag == 'NNP' and current_label == 'PERSON':
                        current_entity.append(word)
                    else:
                        if current_entity:
                            entities.append({
                                'text': ' '.join(current_entity),
                                'label': current_label or 'ENTITY',
                                'source': 'nltk'
                            })
                            current_entity = []
                            current_label = None
                
                # Add last entity
                if current_entity:
                    entities.append({
                        'text': ' '.join(current_entity),
                        'label': current_label or 'ENTITY',
                        'source': 'nltk'
                    })
                
                # Extract organizations (words with title case in specific contexts)
                org_indicators = ['Inc', 'Corp', 'Company', 'LLC', 'Ltd', 'Group', 'Holdings']
                for word, tag in pos_tags:
                    if word in org_indicators and tag == 'NNP':
                        entities.append({
                            'text': word,
                            'label': 'ORG',
                            'source': 'nltk_rule'
                        })
            
            return entities
        except:
            pass
    
    # Ultimate fallback: regex-based entity extraction
    patterns = {
        'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'PHONE': r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        'DATE': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        'MONEY': r'\b[$€£¥₹]\s*\d+(?:,\d{3})*(?:\.\d{2})?\b',
        'URL': r'https?://\S+',
        'SSN': r'\b\d{3}-\d{2}-\d{4}\b',
        'ZIP': r'\b\d{5}(?:-\d{4})?\b'
    }
    
    for label, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            entities.append({
                'text': match.group(),
                'label': label,
                'start': match.start(),
                'end': match.end(),
                'source': 'regex'
            })
    
    return entities

def extract_relationships(entities: List[Dict], text: str) -> List[Dict]:
    """
    Extract relationships between entities based on proximity and patterns
    """
    relationships = []
    
    # Sort entities by position
    sorted_entities = sorted(entities, key=lambda x: x.get('start', 0))
    
    # Look for entities that appear close to each other
    for i, ent1 in enumerate(sorted_entities):
        for j, ent2 in enumerate(sorted_entities[i+1:], i+1):
            # Check if entities are within reasonable distance (100 chars)
            if abs(ent2.get('start', 0) - ent1.get('end', 0)) < 100:
                # Extract the text between them
                between = text[ent1.get('end', 0):ent2.get('start', 0)]
                
                # Look for relationship indicators
                relation_words = ['of', 'in', 'at', 'for', 'from', 'to', 'with', 'by']
                
                relation_type = 'related_to'
                for word in relation_words:
                    if f' {word} ' in between.lower():
                        relation_type = word
                        break
                
                relationships.append({
                    'entity1': ent1['text'],
                    'entity2': ent2['text'],
                    'type': relation_type,
                    'distance': ent2.get('start', 0) - ent1.get('end', 0),
                    'context': between.strip()
                })
    
    return relationships

def prose_to_structured_table(text: str) -> pd.DataFrame:
    """
    Convert prose text to structured table format
    """
    if not text or not text.strip():
        return pd.DataFrame()
    
    # Extract entities
    entities = extract_entities_from_prose(text)
    
    # Extract relationships
    relationships = extract_relationships(entities, text)
    
    # Create structured data
    structured_data = []
    
    # If we have relationships, use them
    if relationships:
        for rel in relationships:
            structured_data.append({
                'Entity_1': rel['entity1'],
                'Relationship': rel['type'],
                'Entity_2': rel['entity2'],
                'Context': rel['context'][:50] if rel['context'] else ''
            })
    
    # If we have entities but no relationships, create entity list
    elif entities:
        for ent in entities:
            structured_data.append({
                'Entity': ent['text'],
                'Type': ent['label'],
                'Source': ent.get('source', 'unknown')
            })
    
    # If we have neither, try sentence-level extraction
    else:
        # Split into sentences
        if NLTK_AVAILABLE:
            sentences = sent_tokenize(text)
        else:
            # Simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
        
        for i, sent in enumerate(sentences):
            if sent.strip():
                # Extract key phrases (nouns and adjectives)
                words = sent.split()
                if NLTK_AVAILABLE:
                    pos_tags = pos_tag(word_tokenize(sent))
                    key_terms = [word for word, tag in pos_tags if tag.startswith(('NN', 'JJ', 'VB'))]
                else:
                    # Simple heuristic: longer words might be important
                    key_terms = [word for word in words if len(word) > 4]
                
                structured_data.append({
                    'Sentence_ID': i + 1,
                    'Sentence': sent.strip(),
                    'Key_Terms': ', '.join(key_terms[:5]),
                    'Word_Count': len(words)
                })
    
    if structured_data:
        df = pd.DataFrame(structured_data)
        return df
    
    # Ultimate fallback: just create a simple table with text chunks
    chunks = [text[i:i+200] for i in range(0, len(text), 200)]
    df = pd.DataFrame({
        'Chunk_ID': range(1, len(chunks) + 1),
        'Text': chunks
    })
    
    return df

# ============================================================================
# LIST TO TABLE CONVERSION
# ============================================================================

def list_to_structured_table(text_lines: List[str]) -> pd.DataFrame:
    """
    Convert list content to structured table
    """
    structured_data = []
    
    list_items = []
    current_item = []
    list_level = 0
    
    for line in text_lines:
        line = line.strip()
        if not line:
            if current_item:
                list_items.append(' '.join(current_item))
                current_item = []
            continue
        
        # Check for list markers
        if re.match(r'^\s*[-•*·○●■→▪▫]\s+', line):
            if current_item:
                list_items.append(' '.join(current_item))
            current_item = [re.sub(r'^\s*[-•*·○●■→▪▫]\s+', '', line)]
            list_level = len(re.match(r'^\s*', line).group())
        elif re.match(r'^\s*\d+[.)]\s+', line):
            if current_item:
                list_items.append(' '.join(current_item))
            current_item = [re.sub(r'^\s*\d+[.)]\s+', '', line)]
            list_level = len(re.match(r'^\s*', line).group())
        else:
            # Continuation of previous list item
            current_item.append(line)
    
    # Add last item
    if current_item:
        list_items.append(' '.join(current_item))
    
    # Create structured data
    for i, item in enumerate(list_items):
        # Try to split item into parts
        parts = item.split(': ', 1)
        if len(parts) == 2:
            structured_data.append({
                'Item_Number': i + 1,
                'Category': parts[0],
                'Value': parts[1]
            })
        else:
            # Check for other patterns
            if ',' in item:
                # Could be comma-separated values
                values = [v.strip() for v in item.split(',')]
                row = {'Item_Number': i + 1}
                for j, val in enumerate(values):
                    row[f'Value_{j+1}'] = val
                structured_data.append(row)
            else:
                # Simple list item
                structured_data.append({
                    'Item_Number': i + 1,
                    'Content': item
                })
    
    if structured_data:
        return pd.DataFrame(structured_data)
    
    return pd.DataFrame()

# ============================================================================
# FORM TO TABLE CONVERSION
# ============================================================================

def form_to_structured_table(text_lines: List[str]) -> pd.DataFrame:
    """
    Convert form-style content (field: value) to structured table
    """
    structured_data = []
    
    current_section = 'General'
    fields = {}
    
    for line in text_lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for section headers (all caps or underlined)
        if line.isupper() or re.match(r'^[A-Z\s]+$', line):
            if fields:
                fields['Section'] = current_section
                structured_data.append(fields)
                fields = {}
            current_section = line
            continue
        
        # Check for field: value pattern
        field_match = re.match(r'^\s*([A-Za-z][A-Za-z\s]*?)\s*:\s*(.+?)\s*$', line)
        if field_match:
            field = field_match.group(1).strip()
            value = field_match.group(2).strip()
            fields[field] = value
        else:
            # Check for other separators
            for sep in ['=', '-', '—']:
                if sep in line:
                    parts = line.split(sep, 1)
                    field = parts[0].strip()
                    value = parts[1].strip() if len(parts) > 1 else ''
                    fields[field] = value
                    break
    
    # Add last section
    if fields:
        fields['Section'] = current_section
        structured_data.append(fields)
    
    if structured_data:
        # Convert to DataFrame, handling different field sets
        df = pd.DataFrame(structured_data)
        return df
    
    return pd.DataFrame()

# ============================================================================
# ENHANCED TABLE EXTRACTION WITH PATTERN RECOGNITION
# ============================================================================

def detect_table_structure(text_lines: List[str]) -> Dict:
    """
    Detect table structure (columns, headers, etc.) from text
    """
    if not text_lines:
        return {'has_header': False, 'columns': [], 'delimiter': None}
    
    # Try to find consistent delimiters
    delimiters = ['\t', '|', ';', ',']
    delimiter_scores = {}
    
    for delim in delimiters:
        scores = []
        for line in text_lines[:10]:  # Check first 10 lines
            if delim in line:
                parts = line.split(delim)
                if len(parts) > 1:
                    scores.append(len(parts))
        
        if scores:
            avg_parts = sum(scores) / len(scores)
            consistency = len(set(scores)) == 1  # All lines have same number of parts
            delimiter_scores[delim] = (avg_parts, consistency)
    
    # Find best delimiter
    best_delim = None
    best_score = 0
    for delim, (avg_parts, consistency) in delimiter_scores.items():
        score = avg_parts * (2 if consistency else 1)
        if score > best_score:
            best_score = score
            best_delim = delim
    
    # Try space-based detection if no clear delimiter
    if not best_delim:
        # Look for multiple spaces
        space_positions = []
        for line in text_lines[:5]:
            positions = [m.start() for m in re.finditer(r'\s{2,}', line)]
            if positions:
                space_positions.extend(positions)
        
        if space_positions:
            # Cluster space positions to find column boundaries
            from collections import Counter
            pos_counter = Counter(space_positions)
            common_positions = [pos for pos, count in pos_counter.items() 
                               if count >= len(text_lines[:5]) * 0.5]
            
            if common_positions:
                return {
                    'has_header': True,
                    'columns': common_positions,
                    'delimiter': 'spaces',
                    'num_columns': len(common_positions) + 1
                }
    
    # Detect header row
    has_header = False
    header_row = None
    
    if len(text_lines) >= 2:
        # Check if first row looks like header (all caps, key terms)
        first_line = text_lines[0].lower()
        header_terms = ['date', 'name', 'id', 'description', 'amount', 'total', 
                       'quantity', 'price', 'code', 'reference', 'account']
        
        if any(term in first_line for term in header_terms):
            has_header = True
            header_row = text_lines[0]
    
    return {
        'has_header': has_header,
        'header_row': header_row,
        'delimiter': best_delim,
        'num_columns': None
    }

def extract_table_with_structure(text_lines: List[str]) -> pd.DataFrame:
    """
    Extract table from text with intelligent structure detection
    """
    if not text_lines:
        return pd.DataFrame()
    
    # Detect structure
    structure = detect_table_structure(text_lines)
    
    # Try to parse based on detected structure
    if structure['delimiter'] and structure['delimiter'] != 'spaces':
        # Parse with explicit delimiter
        data = []
        for line in text_lines:
            if line.strip():
                parts = [p.strip() for p in line.split(structure['delimiter'])]
                data.append(parts)
        
        if data:
            df = pd.DataFrame(data)
            if structure['has_header'] and len(data) > 1:
                df.columns = df.iloc[0]
                df = df[1:].reset_index(drop=True)
            return df
    
    elif structure['delimiter'] == 'spaces' and structure.get('columns'):
        # Parse based on space positions
        data = []
        for line in text_lines:
            if line.strip():
                row = []
                positions = [0] + structure['columns'] + [len(line)]
                for i in range(len(positions) - 1):
                    start = positions[i]
                    end = positions[i+1]
                    cell = line[start:end].strip()
                    row.append(cell)
                data.append(row)
        
        if data:
            df = pd.DataFrame(data)
            if structure['has_header'] and len(data) > 1:
                df.columns = df.iloc[0]
                df = df[1:].reset_index(drop=True)
            return df
    
    # Fallback: try to detect patterns in text
    return detect_and_parse_patterns(text_lines)

def detect_and_parse_patterns(text_lines: List[str]) -> pd.DataFrame:
    """
    Detect and parse patterns in unstructured text
    """
    patterns = []
    
    # Pattern 1: Date + Description + Amount
    date_amount_pattern = re.compile(
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+(.+?)\s+([\d,]+\.\d{2})\s*([A-Z]{3})?'
    )
    
    # Pattern 2: Key: Value pairs
    kv_pattern = re.compile(r'([A-Za-z][A-Za-z\s]*?)\s*[:=]\s*(.+)')
    
    # Pattern 3: Comma-separated values
    csv_pattern = re.compile(r'([^,]+),([^,]+),?([^,]*)')
    
    # Pattern 4: Numbered list with values
    numbered_pattern = re.compile(r'(\d+)[.)]\s+(.+?)(?:\s+(\d+))?$')
    
    structured_data = []
    
    for line in text_lines:
        line = line.strip()
        if not line:
            continue
        
        # Try date-amount pattern
        match = date_amount_pattern.search(line)
        if match:
            groups = match.groups()
            row = {
                'Date': groups[0],
                'Description': groups[1],
                'Amount': groups[2],
                'Currency': groups[3] if len(groups) > 3 else ''
            }
            structured_data.append(row)
            patterns.append('date_amount')
            continue
        
        # Try key-value pattern
        match = kv_pattern.search(line)
        if match:
            row = {match.group(1).strip(): match.group(2).strip()}
            structured_data.append(row)
            patterns.append('key_value')
            continue
        
        # Try CSV pattern
        match = csv_pattern.search(line)
        if match:
            groups = match.groups()
            row = {
                'Column1': groups[0],
                'Column2': groups[1],
                'Column3': groups[2] if len(groups) > 2 else ''
            }
            structured_data.append(row)
            patterns.append('csv')
            continue
        
        # Try numbered pattern
        match = numbered_pattern.search(line)
        if match:
            groups = match.groups()
            row = {
                'Number': groups[0],
                'Item': groups[1],
                'Value': groups[2] if len(groups) > 2 else ''
            }
            structured_data.append(row)
            patterns.append('numbered')
            continue
    
    if structured_data:
        # If multiple patterns detected, try to unify structure
        if len(set(patterns)) > 1:
            # Convert to unified format
            unified_data = []
            for row in structured_data:
                if 'Date' in row:
                    unified_data.append(row)
                elif 'Number' in row and 'Item' in row:
                    unified_data.append({
                        'Date': '',
                        'Description': row['Item'],
                        'Amount': row.get('Value', ''),
                        'Currency': ''
                    })
                else:
                    # Convert key-value to standard format
                    for key, value in row.items():
                        unified_data.append({
                            'Date': '',
                            'Description': f"{key}: {value}",
                            'Amount': '',
                            'Currency': ''
                        })
            
            if unified_data:
                return pd.DataFrame(unified_data)
        
        # If consistent pattern, just create DataFrame
        return pd.DataFrame(structured_data)
    
    # If no patterns found, try to create a simple table
    rows = [line.split() for line in text_lines if line.strip()]
    if rows:
        # Find max columns
        max_cols = max(len(row) for row in rows)
        # Pad rows
        padded_rows = [row + [''] * (max_cols - len(row)) for row in rows]
        return pd.DataFrame(padded_rows)
    
    return pd.DataFrame()

# ============================================================================
# UNIVERSAL EXTRACTION FUNCTION
# ============================================================================

def universal_extract(text: str, page_num: int) -> List[pd.DataFrame]:
    """
    Universal extraction that works for ANY content type
    Returns list of DataFrames
    """
    results = []
    
    if not text or not text.strip():
        return results
    
    # Split into lines
    lines = [line for line in text.split('\n') if line.strip()]
    
    if not lines:
        return results
    
    # Detect content type
    content_type = detect_content_type(lines)
    st.session_state.content_types[page_num] = content_type
    
    # Apply appropriate extraction method
    if content_type == 'table':
        # Try table extraction
        df = extract_table_with_structure(lines)
        if not df.empty:
            st.session_state.extraction_method_used[f"{page_num}_0"] = 'table_structure'
            st.session_state.confidence_scores[f"{page_num}_0"] = 0.9
            results.append(df)
    
    elif content_type == 'prose':
        # Convert prose to table
        df = prose_to_structured_table(text)
        if not df.empty:
            st.session_state.extraction_method_used[f"{page_num}_0"] = 'prose_to_table'
            st.session_state.confidence_scores[f"{page_num}_0"] = 0.7
            results.append(df)
    
    elif content_type == 'list':
        # Convert list to table
        df = list_to_structured_table(lines)
        if not df.empty:
            st.session_state.extraction_method_used[f"{page_num}_0"] = 'list_to_table'
            st.session_state.confidence_scores[f"{page_num}_0"] = 0.8
            results.append(df)
    
    elif content_type == 'form':
        # Convert form to table
        df = form_to_structured_table(lines)
        if not df.empty:
            st.session_state.extraction_method_used[f"{page_num}_0"] = 'form_to_table'
            st.session_state.confidence_scores[f"{page_num}_0"] = 0.85
            results.append(df)
    
    elif content_type == 'mixed':
        # Try multiple approaches and combine
        # First try table extraction
        df1 = extract_table_with_structure(lines)
        if not df1.empty:
            st.session_state.extraction_method_used[f"{page_num}_0"] = 'mixed_table'
            st.session_state.confidence_scores[f"{page_num}_0"] = 0.6
            results.append(df1)
        
        # Then try prose extraction for remaining content
        df2 = prose_to_structured_table(text)
        if not df2.empty and (df1.empty or len(df2) > len(df1)):
            if results:
                # Try to combine with existing results
                combined = pd.concat([results[0], df2], axis=1) if len(results[0]) == len(df2) else df2
                results[0] = combined
            else:
                results.append(df2)
    
    # If still no results, try generic pattern detection
    if not results:
        df = detect_and_parse_patterns(lines)
        if not df.empty:
            st.session_state.extraction_method_used[f"{page_num}_0"] = 'pattern_detection'
            st.session_state.confidence_scores[f"{page_num}_0"] = 0.5
            results.append(df)
    
    return results

# ============================================================================
# ENHANCED OCR WITH CONTENT TYPE DETECTION
# ============================================================================

def enhanced_ocr_with_detection(image):
    """
    Enhanced OCR with content type detection and appropriate processing
    """
    if not TESSERACT_AVAILABLE:
        return []
    
    try:
        # Preprocess image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Multiple preprocessing techniques for different content types
        preprocessed_versions = []
        
        # Version 1: Standard thresholding
        _, thresh1 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        preprocessed_versions.append(('standard', thresh1))
        
        # Version 2: Adaptive thresholding (good for tables)
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        preprocessed_versions.append(('adaptive', thresh2))
        
        # Version 3: Denoised (good for handwritten)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        _, thresh3 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_versions.append(('denoised', thresh3))
        
        # Version 4: Enhanced contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        _, thresh4 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_versions.append(('enhanced', thresh4))
        
        # Try each preprocessing method
        best_results = []
        best_score = 0
        
        for method_name, preprocessed in preprocessed_versions:
            # OCR with different PSM modes
            for psm in [6, 3, 4, 11]:  # Try different page segmentation modes
                config = f'--oem 3 --psm {psm}'
                try:
                    text = pytesseract.image_to_string(preprocessed, config=config)
                    if text and len(text.strip()) > 50:  # Reasonable amount of text
                        # Score this result
                        lines = text.split('\n')
                        line_count = len([l for l in lines if l.strip()])
                        
                        # Check for table indicators
                        table_score = 0
                        if any(re.search(r'\s{3,}', l) for l in lines):
                            table_score += 10
                        if any('\t' in l for l in lines):
                            table_score += 10
                        
                        total_score = line_count + table_score
                        
                        if total_score > best_score:
                            best_score = total_score
                            best_results = [(method_name, psm, text)]
                        elif total_score == best_score:
                            best_results.append((method_name, psm, text))
                except:
                    continue
        
        # Use the best result
        if best_results:
            # Try to detect content type and process appropriately
            for method_name, psm, text in best_results[:1]:  # Use first best
                # Detect content type
                lines = text.split('\n')
                content_type = detect_content_type([l for l in lines if l.strip()])
                
                # Process based on content type
                if content_type == 'table':
                    df = extract_table_with_structure(lines)
                elif content_type == 'prose':
                    df = prose_to_structured_table(text)
                elif content_type == 'list':
                    df = list_to_structured_table(lines)
                elif content_type == 'form':
                    df = form_to_structured_table(lines)
                else:
                    df = detect_and_parse_patterns(lines)
                
                if not df.empty:
                    return [df]
        
        # Fallback to basic OCR
        return basic_ocr_extraction(image)
        
    except Exception as e:
        if st.session_state.debug_mode:
            st.warning(f"Enhanced OCR error: {e}")
        return []

def basic_ocr_extraction(image):
    """
    Basic OCR extraction without complex dependencies
    """
    if not TESSERACT_AVAILABLE:
        return []
    
    try:
        # Get OCR data
        custom_config = '--psm 6 --oem 3'
        data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
        
        # Group by line
        lines = {}
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 30:
                text = data['text'][i].strip()
                if text:
                    line_num = data['line_num'][i]
                    if line_num not in lines:
                        lines[line_num] = []
                    lines[line_num].append({
                        'text': text,
                        'x': data['left'][i],
                        'conf': data['conf'][i]
                    })
        
        # Convert to text
        sorted_lines = []
        for line_num in sorted(lines.keys()):
            words = lines[line_num]
            words.sort(key=lambda x: x['x'])
            line_text = ' '.join([word['text'] for word in words])
            if line_text.strip():
                sorted_lines.append(line_text)
        
        # Use universal extraction
        if sorted_lines:
            full_text = '\n'.join(sorted_lines)
            return universal_extract(full_text, 1)
        
        return []
        
    except Exception as e:
        return []

# ============================================================================
# ENHANCED PDF EXTRACTION WITH UNIVERSAL HANDLING
# ============================================================================

def extract_pdf_content(pdf_path: str, page_num: int) -> List[pd.DataFrame]:
    """
    Extract content from PDF with universal handling
    """
    results = []
    
    # Try pdfplumber first
    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num <= len(pdf.pages):
                    page = pdf.pages[page_num - 1]
                    
                    # Try table extraction
                    page_tables = page.extract_tables()
                    
                    for table_data in page_tables:
                        if table_data and len(table_data) > 1:
                            df = pd.DataFrame(table_data)
                            df = df.replace('', pd.NA).dropna(how='all').reset_index(drop=True)
                            df = df.dropna(axis=1, how='all')
                            
                            if not df.empty and len(df) > 1:
                                st.session_state.extraction_method_used[f"{page_num}_{len(results)}"] = 'pdfplumber_table'
                                st.session_state.confidence_scores[f"{page_num}_{len(results)}"] = 0.9
                                results.append(df)
                    
                    # If no tables found, extract text and use universal extraction
                    if not results:
                        text = page.extract_text()
                        if text:
                            universal_results = universal_extract(text, page_num)
                            results.extend(universal_results)
        except Exception as e:
            if st.session_state.debug_mode:
                st.warning(f"PDF extraction error: {e}")
    
    # Try camelot if available
    if CAMELOT_AVAILABLE and not results:
        try:
            tables_camelot = camelot.read_pdf(pdf_path, pages=str(page_num), flavor='lattice')
            for table in tables_camelot:
                df = table.df
                if not df.empty and len(df) > 1:
                    st.session_state.extraction_method_used[f"{page_num}_{len(results)}"] = 'camelot_lattice'
                    st.session_state.confidence_scores[f"{page_num}_{len(results)}"] = 0.85
                    results.append(df)
            
            if not results:
                tables_camelot = camelot.read_pdf(pdf_path, pages=str(page_num), flavor='stream')
                for table in tables_camelot:
                    df = table.df
                    if not df.empty and len(df) > 1:
                        st.session_state.extraction_method_used[f"{page_num}_{len(results)}"] = 'camelot_stream'
                        st.session_state.confidence_scores[f"{page_num}_{len(results)}"] = 0.8
                        results.append(df)
        except Exception as e:
            if st.session_state.debug_mode:
                st.warning(f"Camelot error: {e}")
    
    return results

# ============================================================================
# MAIN EXTRACTION FUNCTION (UPDATED)
# ============================================================================

def extract_tables_from_document(file_path: str, pages: List[int], mode: str) -> Dict[int, List[pd.DataFrame]]:
    """
    Enhanced extraction function with universal content handling
    """
    tables_by_page = {}
    file_ext = file_path.split('.')[-1].lower()
    
    for page_num in pages:
        page_tables = []
        
        # Handle PDF files
        if file_ext == 'pdf':
            # Try PDF extraction methods
            pdf_tables = extract_pdf_content(file_path, page_num)
            page_tables.extend(pdf_tables)
        
        # For images or scanned PDFs, use enhanced OCR
        if (not page_tables) and TESSERACT_AVAILABLE:
            try:
                # Convert PDF page to image if needed
                if file_ext == 'pdf' and PDF2IMAGE_AVAILABLE:
                    images = pdf2image.convert_from_path(file_path, first_page=page_num, 
                                                        last_page=page_num, dpi=300)
                    if images:
                        image = np.array(images[0])
                else:
                    # Load image directly
                    if PIL_AVAILABLE:
                        image = np.array(Image.open(file_path))
                    else:
                        image = cv2.imread(file_path)
                
                if image is not None:
                    # Use enhanced OCR with detection
                    ocr_tables = enhanced_ocr_with_detection(image)
                    page_tables.extend(ocr_tables)
                        
            except Exception as e:
                if st.session_state.debug_mode:
                    st.warning(f"OCR error on page {page_num}: {e}")
        
        # Clean up the tables
        cleaned_tables = []
        for table in page_tables:
            if isinstance(table, pd.DataFrame) and not table.empty:
                # Remove completely empty rows and columns
                table = table.replace('', pd.NA).dropna(how='all').dropna(axis=1, how='all')
                if not table.empty and len(table) >= 1:  # Allow single rows for prose conversion
                    cleaned_tables.append(table)
        
        if cleaned_tables:
            tables_by_page[page_num] = cleaned_tables
    
    return tables_by_page

# ============================================================================
# ENHANCED UI COMPONENTS
# ============================================================================

def render_sidebar():
    """Render the sidebar UI with enhanced options"""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/ms-excel.png", width=80)
        st.header("📁 Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif', 'webp', 'txt'],
            help="Upload any document - it will be converted to structured Excel format",
            key="file_uploader"
        )
        
        if uploaded_file:
            file_hash = get_file_hash(uploaded_file.getvalue())
            
            if not st.session_state.pdf_uploaded or st.session_state.current_file_hash != file_hash:
                st.session_state.pdf_uploaded = True
                st.session_state.current_file_hash = file_hash
                st.session_state.current_file = uploaded_file.name
                
                # Get document metadata
                try:
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    
                    if file_extension == 'pdf' and PDFPLUMBER_AVAILABLE:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                            tmp.write(uploaded_file.getvalue())
                            pdf_path = tmp.name
                        
                        with pdfplumber.open(pdf_path) as pdf:
                            total_pages = len(pdf.pages)
                            st.session_state.pdf_metadata = {
                                'total_pages': total_pages,
                                'file_name': uploaded_file.name,
                                'file_size': f"{uploaded_file.size / 1024:.1f} KB",
                                'file_type': 'PDF',
                                'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                        
                        os.unlink(pdf_path)
                    elif file_extension == 'txt':
                        # Text file
                        content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
                        lines = content.split('\n')
                        st.session_state.pdf_metadata = {
                            'total_pages': 1,
                            'file_name': uploaded_file.name,
                            'file_size': f"{uploaded_file.size / 1024:.1f} KB",
                            'file_type': 'Text',
                            'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'line_count': len(lines)
                        }
                    else:
                        # For images, just count as 1 page
                        st.session_state.pdf_metadata = {
                            'total_pages': 1,
                            'file_name': uploaded_file.name,
                            'file_size': f"{uploaded_file.size / 1024:.1f} KB",
                            'file_type': 'Image' if file_extension != 'pdf' else 'PDF (estimated)',
                            'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                    
                    st.success(f"✅ Document loaded successfully")
                    
                except Exception as e:
                    st.error(f"Error reading document: {e}")
        
        st.markdown("---")
        
        with st.expander("ℹ️ About", expanded=False):
            st.markdown("""
            **Universal Document to Excel Converter**
            
            **Features:**
            - Converts ANY content to structured tables
            - Prose → Entity extraction
            - Lists → Multi-column tables
            - Forms → Key-value pairs
            - Tables → Perfect preservation
            - Mixed content → Smart parsing
            
            **Supported Formats:**
            - PDF (digital & scanned)
            - Images (PNG, JPG, JPEG, TIFF, BMP, GIF, WebP)
            - Text files (TXT)
            
            **Version:** 4.0.0 (Universal Edition)
            """)
        
        return uploaded_file

def render_extraction_settings() -> Tuple[str, bool, int, int, List[int], str, bool]:
    """Render extraction settings with universal mode"""
    st.header("🎯 Extraction Settings")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        extraction_mode = st.selectbox(
            "Extraction Mode",
            options=[
                "Universal (Auto-detect ANY content)",
                "Bank Statement Mode (Optimized)",
                "Table Extraction Only",
                "Prose to Table Conversion",
                "List/Form Extraction",
                "All Methods (Comprehensive)"
            ],
            index=0,  # Default to Universal mode
            help="Universal mode automatically detects and converts any content type to structured tables."
        )
        
        # Map selection to mode string
        mode_map = {
            "Universal (Auto-detect ANY content)": "Universal",
            "Bank Statement Mode (Optimized)": "Bank Statement",
            "Table Extraction Only": "Tables Only",
            "Prose to Table Conversion": "Prose",
            "List/Form Extraction": "Lists",
            "All Methods (Comprehensive)": "All"
        }
        st.session_state.extraction_mode = mode_map[extraction_mode]
    
    with col2:
        debug_mode = st.checkbox("Debug Mode", value=False, help="Show detailed extraction information")
        st.session_state.debug_mode = debug_mode
    
    st.markdown("---")
    
    # Page selection
    st.header("📄 Page Selection")
    
    total_pages = st.session_state.pdf_metadata['total_pages']
    
    page_options = st.radio(
        "Scan Mode",
        ["Quick Scan (first 10 pages)", "Custom Range", "All Pages"],
        horizontal=True
    )
    
    selected_pages = []
    
    if page_options == "Quick Scan (first 10 pages)":
        selected_pages = list(range(1, min(11, total_pages + 1)))
        st.info(f"📋 Will scan pages 1-{len(selected_pages)}")
    
    elif page_options == "Custom Range":
        col1, col2 = st.columns(2)
        with col1:
            start_page = st.number_input("From Page", 1, total_pages, 1)
        with col2:
            end_page = st.number_input("To Page", start_page, total_pages, min(start_page + 10, total_pages))
        selected_pages = list(range(start_page, end_page + 1))
        st.info(f"📋 Will scan pages {start_page}-{end_page}")
    
    else:  # All Pages
        selected_pages = list(range(1, total_pages + 1))
        if total_pages > 50:
            st.warning(f"⚠️ Scanning all {total_pages} pages may take a while")
        else:
            st.info(f"📋 Will scan all {total_pages} pages")
    
    # Basic settings
    with st.expander("⚙️ Advanced Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            min_rows = st.number_input(
                "Minimum rows", 
                1, 1000, 1,
                help="Minimum number of rows to consider as valid (1 for prose conversion)"
            )
            
            entity_extraction = st.checkbox("Enable entity extraction", value=True,
                                           help="Extract named entities from prose")
            
            relationship_detection = st.checkbox("Detect relationships", value=True,
                                                help="Find relationships between entities")
        
        with col2:
            ocr_lang = st.selectbox(
                "OCR Language",
                options=['eng', 'fra', 'deu', 'spa', 'ita', 'por'],
                index=0,
                help="Select language for OCR"
            )
            
            enhance_handwriting = st.checkbox("Enhance image", value=True,
                                             help="Apply additional preprocessing for better OCR")
    
    return extraction_mode, debug_mode, min_rows, 1, selected_pages, ocr_lang, enhance_handwriting

def display_extraction_results(total_tables_found: int, filtered_tables: Dict, tables_ignored: int, min_rows: int, min_cols: int):
    """Display extraction results with content type information"""
    if total_tables_found > 0:
        # Count content types
        content_types = st.session_state.content_types
        type_counts = Counter(content_types.values())
        
        type_display = " • ".join([f"{t}: {c}" for t, c in type_counts.most_common()])
        
        st.markdown(f"""
        <div class="success-box">
        <h3>✅ Extraction Complete!</h3>
        <p>Found <strong>{total_tables_found}</strong> data structures</p>
        <p>Pages processed: <strong>{len(filtered_tables)}</strong></p>
        <p>Content types detected: <strong>{type_display}</strong></p>
        <p>Items ignored: <strong>{tables_ignored}</strong> (empty or below threshold)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show statistics
        stats = st.session_state.extraction_stats
        if stats['total_tables'] > 0:
            st.markdown(f"""
            <div class="info-box">
            <strong>📊 Extraction Statistics:</strong><br>
            Total rows extracted: {stats['total_rows']:,}<br>
            Average table size: {stats['avg_rows_per_table']:.0f} rows × {stats['avg_columns_per_table']:.0f} columns<br>
            Average confidence: {stats.get('avg_confidence', 0):.1%}<br>
            Extraction methods used: {', '.join([f"{k}: {v}" for k, v in stats.get('extraction_methods', {}).items()])}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ No data found. Try adjusting settings or using a different extraction mode.")
        
        # Provide suggestions
        st.info("""
        **Suggestions:**
        - Try "Universal" mode for automatic detection
        - Ensure document is clear and well-lit
        - For prose, try "Prose to Table Conversion"
        - For forms, try "List/Form Extraction"
        - Enable Debug Mode to see raw extraction
        """)

def render_table_preview(table_id: str, table: pd.DataFrame):
    """Enhanced table preview with content type info"""
    if st.session_state.get(f"show_preview_{table_id}", False):
        with st.container():
            # Get content type if available
            page_num = st.session_state.selected_tables[table_id]["page"]
            content_type = st.session_state.content_types.get(page_num, 'unknown')
            
            # Show content type badge
            type_class = f"type-{content_type}"
            st.markdown(f"""
            <div class="content-type {type_class}">
                📄 Detected: {content_type.upper()}
            </div>
            """, unsafe_allow_html=True)
            
            # Get filtered data based on selections
            use_all_rows = st.session_state.row_selections[table_id].get("all_rows", True)
            selected_columns = [col for col, is_selected in st.session_state.column_selections.get(table_id, {}).items() if is_selected]
            
            if use_all_rows:
                preview_df = table[selected_columns] if selected_columns else table
            else:
                start_row = st.session_state.row_selections[table_id].get("start_row", 0)
                end_row = st.session_state.row_selections[table_id].get("end_row", len(table) - 1)
                preview_df = table.iloc[start_row:end_row+1][selected_columns] if selected_columns else table.iloc[start_row:end_row+1]
            
            if not preview_df.empty:
                st.dataframe(
                    preview_df.head(20),
                    use_container_width=True,
                    height=300
                )
                st.caption(f"Showing first 20 rows of {len(preview_df):,} total rows")
                
                # Show data types
                st.markdown("**Column Data Types:**")
                dtypes = preview_df.dtypes.to_dict()
                dtype_cols = st.columns(min(4, len(dtypes)))
                for idx, (col, dtype) in enumerate(list(dtypes.items())[:4]):
                    with dtype_cols[idx % 4]:
                        st.markdown(f"""
                        <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px;'>
                            <strong>{str(col)[:20]}</strong><br>
                            <small>{dtype}</small>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No data to preview")
    
    st.markdown("---")

def render_welcome_screen():
    """Enhanced welcome screen for universal converter"""
    st.markdown("""
    <div class="info-box">
    <h1>🔄 Universal Document to Excel Converter</h1>
    <p style='font-size: 18px;'>Convert <strong>ANY content</strong> - tables, prose, lists, forms, or mixed - into structured Excel tables.</p>
    <p><strong>Intelligent parsing that understands your document's structure</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature showcase
    st.markdown("### ✨ What Can It Convert?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Tables & Spreadsheets</div>
            <div class="feature-description">Perfect preservation of rows, columns, and headers from any table format</div>
        </div>
        
        <div class="feature-card">
            <div class="feature-icon">📝</div>
            <div class="feature-title">Prose & Paragraphs</div>
            <div class="feature-description">Extracts entities, relationships, and key information into structured format</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📋</div>
            <div class="feature-title">Lists & Bullet Points</div>
            <div class="feature-description">Converts any list format into multi-column tables with hierarchy</div>
        </div>
        
        <div class="feature-card">
            <div class="feature-icon">🏦</div>
            <div class="feature-title">Bank Statements</div>
            <div class="feature-description">Specialized extraction for financial documents with date/amount patterns</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📄</div>
            <div class="feature-title">Forms & Applications</div>
            <div class="feature-description">Extracts field:value pairs into organized key-value tables</div>
        </div>
        
        <div class="feature-card">
            <div class="feature-icon">🔄</div>
            <div class="feature-title">Mixed Content</div>
            <div class="feature-description">Smart parsing of documents with multiple content types</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start guide
    st.markdown("### 🚀 How It Works")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="progress-step">
            <div class="step-number">1</div>
            <div class="step-text">Upload Any Document</div>
        </div>
        <small>PDF, images, text files</small>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="progress-step">
            <div class="step-number">2</div>
            <div class="step-text">Auto-detect Content</div>
        </div>
        <small>Table? Prose? List? Form?</small>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="progress-step">
            <div class="step-number">3</div>
            <div class="step-text">Intelligent Parsing</div>
        </div>
        <small>Convert to structured data</small>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="progress-step">
            <div class="step-number">4</div>
            <div class="step-text">Export to Excel</div>
        </div>
        <small>Ready for analysis</small>
        """, unsafe_allow_html=True)
    
    # Tips section
    with st.expander("💡 Tips for Best Results", expanded=False):
        st.markdown("""
        - **Universal mode** works for most documents automatically
        - **For clear tables**: Use "Table Extraction Only" for speed
        - **For narrative text**: "Prose to Table" extracts entities
        - **For forms**: "List/Form Extraction" captures field-value pairs
        - **For financial docs**: "Bank Statement Mode" is optimized
        - **For mixed content**: "All Methods" combines approaches
        """)
    
    st.markdown("---")
    st.markdown("*Upload a document using the sidebar to begin extraction*")

# ============================================================================
# MAIN APPLICATION (UPDATED)
# ============================================================================

def main():
    """Main application entry point"""
    # Load CSS
    load_css()
    
    # Initialize session states
    init_session_states()
    
    # Render sidebar and get uploaded file
    uploaded_file = render_sidebar()
    
    # Show availability warnings if needed
    if not st.session_state.cv_available:
        st.sidebar.warning("⚠️ OpenCV not installed. Image processing features limited.")
    
    if not st.session_state.pdf_available:
        st.sidebar.warning("⚠️ PDF libraries not fully installed. PDF extraction limited.")
    
    if not st.session_state.tesseract_available:
        st.sidebar.warning("⚠️ Tesseract OCR not installed. Text recognition from images limited.")
    
    if not st.session_state.nlp_available:
        st.sidebar.info("💡 Install spaCy or NLTK for better prose extraction")
    
    # Main content
    if st.session_state.pdf_uploaded and uploaded_file:
        render_document_info()
        
        # Get extraction settings
        extraction_mode, debug_mode, min_rows, min_cols, selected_pages, ocr_lang, enhance_handwriting = render_extraction_settings()
        
        # Store settings
        st.session_state.ocr_language = ocr_lang
        
        # Extract button
        if selected_pages and st.button("🔍 Extract Data", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner(f"Analyzing {len(selected_pages)} pages..."):
                # Save file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp:
                    tmp.write(uploaded_file.getvalue())
                    file_path = tmp.name
                
                try:
                    # Update progress
                    status_text.text("Processing document...")
                    progress_bar.progress(20)
                    
                    # Reset content types for new extraction
                    st.session_state.content_types = {}
                    st.session_state.entity_extractions = {}
                    st.session_state.relationship_graphs = {}
                    st.session_state.confidence_scores = {}
                    st.session_state.extraction_method_used = {}
                    
                    # Extract tables based on mode
                    if st.session_state.extraction_mode == "Universal":
                        # Use universal extraction for all content
                        tables_by_page = extract_tables_from_document(
                            file_path, 
                            selected_pages, 
                            "Universal"
                        )
                    else:
                        # Use existing extraction modes
                        tables_by_page = extract_tables_from_document(
                            file_path, 
                            selected_pages, 
                            st.session_state.extraction_mode
                        )
                    
                    progress_bar.progress(60)
                    status_text.text("Analyzing extracted data...")
                    
                    # Filter tables based on criteria
                    filtered_tables, total_tables_found, tables_ignored = filter_tables_by_size(
                        tables_by_page, min_rows, 1  # min_cols = 1 for single column from prose
                    )
                    
                    st.session_state.tables_data = filtered_tables
                    
                    # Initialize selections
                    initialize_table_selections(filtered_tables)
                    
                    # Calculate statistics
                    st.session_state.extraction_stats = calculate_extraction_stats(filtered_tables)
                    
                    progress_bar.progress(100)
                    status_text.text("Extraction complete!")
                    time.sleep(1)
                    status_text.empty()
                    progress_bar.empty()
                    
                    # Show results
                    display_extraction_results(total_tables_found, filtered_tables, tables_ignored, min_rows, 1)
                    
                    # Show debug info if enabled
                    if debug_mode:
                        with st.expander("🔧 Debug Information"):
                            st.json({
                                'extraction_mode': st.session_state.extraction_mode,
                                'pages_scanned': len(selected_pages),
                                'tables_found': total_tables_found,
                                'tables_ignored': tables_ignored,
                                'content_types': dict(Counter(st.session_state.content_types.values())),
                                'extraction_methods': dict(st.session_state.extraction_method_used),
                                'avg_confidence': st.session_state.extraction_stats.get('avg_confidence', 0),
                                'ocr_settings': {
                                    'language': ocr_lang,
                                    'enhance_handwriting': enhance_handwriting
                                },
                                'libraries_available': {
                                    'opencv': st.session_state.cv_available,
                                    'tesseract': st.session_state.tesseract_available,
                                    'nlp': st.session_state.nlp_available,
                                    'pdfplumber': PDFPLUMBER_AVAILABLE,
                                    'camelot': CAMELOT_AVAILABLE
                                }
                            })
                    
                except Exception as e:
                    st.error(f"Error extracting data: {str(e)}")
                    if debug_mode:
                        st.exception(e)
                finally:
                    try:
                        os.unlink(file_path)
                    except:
                        pass
        
        # Display extracted tables
        if st.session_state.tables_data:
            render_table_selection()
            render_export_section(uploaded_file)
    
    else:
        render_welcome_screen()
    
    # Render footer
    render_footer()

# ============================================================================
# REMAINING FUNCTIONS (keeping your existing ones)
# ============================================================================

# Include all your existing functions that I haven't modified above
# (render_document_info, render_table_selection, render_export_section, etc.)
# They remain the same as in your original code

def render_document_info():
    """Render document information metrics"""
    st.markdown("### 📊 Document Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.pdf_metadata.get('file_type', 'Document')}</div>
            <div class="metric-label">File Type</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.pdf_metadata['total_pages']}</div>
            <div class="metric-label">Total Pages</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.pdf_metadata['file_size']}</div>
            <div class="metric-label">File Size</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.pdf_metadata.get('upload_time', 'Now')}</div>
            <div class="metric-label">Upload Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

def render_table_selection():
    """Render table selection interface"""
    st.markdown("---")
    st.header("📋 Review & Select Data")
    
    # Summary metrics
    total_tables = sum(len(tables) for tables in st.session_state.tables_data.values())
    selected_tables = sum(1 for info in st.session_state.selected_tables.values() if info.get("selected", False))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_tables}</div>
            <div class="metric-label">Tables Found</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{selected_tables}</div>
            <div class="metric-label">Selected for Export</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_rows_selected = 0
        for table_id, info in st.session_state.selected_tables.items():
            if info.get("selected", False):
                df = info.get("df")
                if df is not None:
                    total_rows_selected += len(df)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_rows_selected:,}</div>
            <div class="metric-label">Total Rows</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_cols_selected = 0
        for table_id, info in st.session_state.selected_tables.items():
            if info.get("selected", False):
                df = info.get("df")
                if df is not None:
                    total_cols_selected += len(df.columns)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_cols_selected}</div>
            <div class="metric-label">Total Columns</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Table browser
    pages_with_tables = sorted(st.session_state.tables_data.keys())
    
    if pages_with_tables:
        render_table_tabs(pages_with_tables)

def render_table_tabs(pages_with_tables: List[int]):
    """Render tabs for each page with tables"""
    tab_labels = [f"Page {page} ({len(st.session_state.tables_data[page])} tables)" 
                 for page in pages_with_tables]
    tabs = st.tabs(tab_labels)
    
    for tab_idx, (page_num, tab) in enumerate(zip(pages_with_tables, tabs)):
        with tab:
            tables_on_page = st.session_state.tables_data[page_num]
            
            for table_idx, table in enumerate(tables_on_page):
                # Find table ID
                table_id = None
                for t_id, t_info in st.session_state.selected_tables.items():
                    if t_info["page"] == page_num and t_info["table_idx"] == table_idx:
                        table_id = t_id
                        break
                
                if table_id:
                    render_single_table(table_id, table, table_idx, page_num)

def render_single_table(table_id: str, table: pd.DataFrame, table_idx: int, page_num: int):
    """Render a single table with selection options"""
    # Table header with analysis
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### 📊 Table {table_idx + 1}")
    with col2:
        st.markdown(f"<span class='badge-info'>{len(table)} rows × {len(table.columns)} cols</span>", 
                  unsafe_allow_html=True)
    
    # Show content type if available
    content_type = st.session_state.content_types.get(page_num, 'unknown')
    type_class = f"type-{content_type}"
    st.markdown(f"""
    <div class="content-type {type_class}" style="margin-top: -10px; margin-bottom: 10px;">
        📄 Detected: {content_type.upper()}
    </div>
    """, unsafe_allow_html=True)
    
    # Table selection
    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        is_selected = st.checkbox(
            "Include in export",
            value=st.session_state.selected_tables[table_id].get("selected", True),
            key=f"select_{table_id}",
            help="Check to include this table in the final Excel file"
        )
        st.session_state.selected_tables[table_id]["selected"] = is_selected
    
    with col2:
        if is_selected:
            st.markdown("<span class='badge-success'>✓ Will be exported</span>", 
                      unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge-warning'>✗ Excluded</span>", 
                      unsafe_allow_html=True)
    
    with col3:
        if st.button("Preview Table", key=f"preview_btn_{table_id}"):
            st.session_state[f"show_preview_{table_id}"] = not st.session_state.get(f"show_preview_{table_id}", False)
    
    if is_selected:
        render_column_selection(table_id, table)
        render_row_selection(table_id, table)
        render_table_preview(table_id, table)

def render_column_selection(table_id: str, table: pd.DataFrame):
    """Render column selection interface"""
    st.markdown("#### 🗂️ Column Selection")
    
    # Analyze columns for patterns
    column_analysis = analyze_columns_for_patterns(table)
    
    # Create multi-select for columns
    all_columns = table.columns.tolist()
    
    # Get current selections
    current_selections = st.session_state.column_selections.get(table_id, {})
    default_selected = [col for col in all_columns if current_selections.get(col, True)]
    
    selected_columns = st.multiselect(
        f"Choose columns",
        options=all_columns,
        default=default_selected,
        key=f"cols_{table_id}",
        help="Select columns to include in export"
    )
    
    # Update column selections
    st.session_state.column_selections[table_id] = {
        col: (col in selected_columns) for col in all_columns
    }
    
    # Show column analysis
    if selected_columns and column_analysis:
        st.markdown("**Detected Column Patterns:**")
        cols = st.columns(min(4, len(selected_columns)))
        for idx, col in enumerate(selected_columns[:4]):
            with cols[idx % 4]:
                if col in column_analysis and column_analysis[col]:
                    col_str = str(col) if col is not None else ""
                    if col_str:
                        tags = " • ".join(column_analysis[col])
                        st.markdown(f"""
                        <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px;'>
                            <strong>{col_str[:20]}</strong><br>
                            <small>{tags}</small>
                        </div>
                        """, unsafe_allow_html=True)

def render_row_selection(table_id: str, table: pd.DataFrame):
    """Render row selection interface"""
    st.markdown("#### 📊 Row Range")
    total_rows = len(table)
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        use_all_rows = st.checkbox(
            "Export all rows",
            value=st.session_state.row_selections[table_id].get("all_rows", True),
            key=f"allrows_{table_id}"
        )
        st.session_state.row_selections[table_id]["all_rows"] = use_all_rows
    
    if not use_all_rows:
        with col2:
            start_row = st.number_input(
                "Start row",
                0,
                total_rows - 1,
                st.session_state.row_selections[table_id].get("start_row", 0),
                key=f"start_{table_id}",
                help="Starting row index (0-based)"
            )
            st.session_state.row_selections[table_id]["start_row"] = start_row
        
        with col3:
            end_row = st.number_input(
                "End row",
                start_row + 1,
                total_rows - 1,
                st.session_state.row_selections[table_id].get("end_row", total_rows - 1),
                key=f"end_{table_id}",
                help="Ending row index (inclusive)"
            )
            st.session_state.row_selections[table_id]["end_row"] = end_row
        
        if start_row < end_row:
            rows_to_export = end_row - start_row + 1
            st.info(f"📋 Will export rows {start_row:,} to {end_row:,} ({rows_to_export:,} rows)")

def analyze_columns_for_patterns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Advanced column analysis to detect patterns in data
    Returns detected patterns for each column
    """
    column_analysis = {}
    
    for col in df.columns:
        col_name = str(col).lower().strip()
        col_data = df[col].astype(str).str.lower()
        
        # Expanded pattern dictionary
        patterns = {
            'debit': ['debit', 'dr', 'withdrawal', 'charge', 'payment', 'debit amount', 'debit amt', 'withdrawals'],
            'credit': ['credit', 'cr', 'deposit', 'receipt', 'income', 'credit amount', 'credit amt', 'deposits'],
            'amount': ['amount', 'amt', 'value', 'total', 'sum', 'balance amount', 'transaction amount'],
            'date': ['date', 'time', 'day', 'month', 'year', 'transaction date', 'posting date', 'value date'],
            'description': ['description', 'desc', 'detail', 'note', 'remark', 'particulars', 'narration', 'transaction details'],
            'balance': ['balance', 'bal', 'remaining', 'closing balance', 'opening balance', 'running balance'],
            'account': ['account', 'acct', 'account no', 'acc no', 'account number', 'a/c', 'account name'],
            'name': ['name', 'customer', 'client', 'person', 'party name', 'beneficiary', 'payee', 'payer'],
            'id': ['id', 'no.', 'number', 'ref', 'reference', 'transaction id', 'trxn id', 'invoice no'],
            'status': ['status', 'state', 'condition', 'payment status', 'transaction status'],
            'type': ['type', 'category', 'class', 'classification', 'transaction type'],
            'code': ['code', 'code no', 'code number', 'code', 'sort code', 'branch code'],
            'phone': ['phone', 'mobile', 'tel', 'telephone', 'contact no'],
            'email': ['email', 'e-mail', 'mail'],
            'address': ['address', 'addr', 'location', 'city', 'state', 'zip', 'postal'],
            'quantity': ['quantity', 'qty', 'count', 'number of', 'units'],
            'price': ['price', 'rate', 'cost', 'unit price', 'rate per'],
            'tax': ['tax', 'vat', 'gst', 'sales tax', 'service tax'],
            'discount': ['discount', 'dis', 'offer', 'concession'],
            'reference': ['reference', 'ref', 'ref no', 'reference number'],
            'entity': ['entity', 'person', 'organization', 'company', 'individual'],
            'relationship': ['relationship', 'relation', 'connected', 'associated']
        }
        
        matches = []
        for pattern_name, keywords in patterns.items():
            # Check column name
            if any(keyword in col_name for keyword in keywords):
                matches.append(pattern_name)
            
            # Check sample data for patterns
            if len(matches) < 2:  # Limit to top 2 matches
                sample = col_data.head(20).dropna()
                if not sample.empty:
                    sample_text = ' '.join(sample)
                    # Check for pattern indicators in data
                    if pattern_name == 'date':
                        # Look for date patterns
                        if re.search(r'\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}', sample_text):
                            matches.append(pattern_name)
                    elif pattern_name == 'amount':
                        # Look for currency patterns
                        if re.search(r'[\$€£¥₹]?\s*\d+[,.]?\d*', sample_text):
                            matches.append(pattern_name)
                    elif pattern_name == 'phone':
                        # Look for phone patterns
                        if re.search(r'\+?\d{1,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}', sample_text):
                            matches.append(pattern_name)
                    elif pattern_name == 'email':
                        # Look for email patterns
                        if re.search(r'[\w\.-]+@[\w\.-]+\.\w+', sample_text):
                            matches.append(pattern_name)
                    elif pattern_name == 'entity':
                        # Look for proper nouns (capitalized words)
                        if re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', sample_text):
                            matches.append(pattern_name)
                    elif any(keyword in sample_text for keyword in keywords):
                        matches.append(pattern_name)
        
        # Remove duplicates and limit to top 3
        matches = list(dict.fromkeys(matches))[:3]
        column_analysis[col] = matches
    
    return column_analysis

def filter_tables_by_size(tables_by_page: Dict, min_rows: int, min_cols: int) -> Tuple[Dict, int, int]:
    """Filter tables based on minimum size criteria"""
    filtered_tables = {}
    total_tables_found = 0
    tables_ignored = 0
    
    for page_num, tables in tables_by_page.items():
        filtered_page_tables = []
        for table in tables:
            if len(table) >= min_rows and len(table.columns) >= min_cols:
                filtered_page_tables.append(table)
                total_tables_found += 1
            else:
                tables_ignored += 1
        
        if filtered_page_tables:
            filtered_tables[page_num] = filtered_page_tables
    
    return filtered_tables, total_tables_found, tables_ignored

def initialize_table_selections(filtered_tables: Dict):
    """Initialize selection dictionaries for tables"""
    st.session_state.selected_tables = {}
    st.session_state.column_selections = {}
    st.session_state.row_selections = {}
    st.session_state.all_columns = {}
    
    table_counter = 1
    
    for page_num, tables in filtered_tables.items():
        for table_idx, table in enumerate(tables):
            table_id = f"table_{table_counter}"
            
            # Store table info
            st.session_state.selected_tables[table_id] = {
                "page": page_num,
                "table_idx": table_idx,
                "selected": True,
                "df": table,
                "shape": f"{len(table)}x{len(table.columns)}"
            }
            
            # Initialize column selections (all selected by default)
            st.session_state.column_selections[table_id] = {
                col: True for col in table.columns
            }
            
            # Initialize row range selection
            st.session_state.row_selections[table_id] = {
                "start_row": 0,
                "end_row": len(table) - 1,
                "all_rows": True
            }
            
            # Store all columns for analysis
            st.session_state.all_columns[table_id] = table.columns.tolist()
            
            table_counter += 1

def render_export_section(uploaded_file):
    """Render export section"""
    st.markdown("---")
    st.header("🚀 Export to Excel")
    
    col1, col2 = st.columns(2)
    
    with col1:
        excel_name = st.text_input(
            "Excel file name",
            value=f"{st.session_state.pdf_metadata['file_name'].split('.')[0]}_extracted.xlsx"
        )
        
        export_mode = st.radio(
            "Export format",
            ["Each table → Separate sheet", "Combine all → One sheet", "Group by page"],
            horizontal=True
        )
    
    with col2:
        include_metadata = st.checkbox("Include metadata sheet", value=True,
                                      help="Add a sheet with extraction information")
        auto_format = st.checkbox("Auto-format Excel", value=True,
                                 help="Auto-adjust column widths and format numbers/dates")
        
        # Data cleaning options
        st.markdown("**Data Cleaning:**")
        remove_empty = st.checkbox("Remove empty rows", value=True,
                                  help="Remove rows that are completely empty")
        strip_whitespace = st.checkbox("Strip whitespace", value=True,
                                      help="Remove extra spaces from text")
    
    # Export button
    if st.button("📥 Generate Excel File", type="primary", use_container_width=True):
        tables_to_export, export_summary = prepare_tables_for_export(
            remove_empty, strip_whitespace
        )
        
        if not tables_to_export:
            st.warning("No data selected for export! Please select at least one column per table.")
        else:
            total_rows = sum(len(t["df"]) for t in tables_to_export)
            st.info(f"Preparing to export {total_rows:,} total rows from {len(tables_to_export)} tables...")
            
            with st.spinner(f"Creating Excel file with {total_rows:,} rows..."):
                total_tables = sum(len(tables) for tables in st.session_state.tables_data.values())
                excel_data = export_to_excel(
                    tables_to_export, export_mode, include_metadata, 
                    auto_format, excel_name, st.session_state.pdf_metadata, total_tables
                )
                
                # Success message
                avg_rows = total_rows // len(tables_to_export) if tables_to_export else 0
                
                st.markdown(f"""
                <div class="success-box">
                <h3>✅ Excel File Created Successfully!</h3>
                <p><strong>File:</strong> {excel_name}</p>
                <p><strong>Tables exported:</strong> {len(tables_to_export)}</p>
                <p><strong>Total rows exported:</strong> {total_rows:,}</p>
                <p><strong>Average rows per table:</strong> {avg_rows:,}</p>
                <p><strong>Extraction Mode:</strong> {st.session_state.extraction_mode}</p>
                <p><strong>File size:</strong> {len(excel_data) / 1024:.1f} KB</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show summary table
                st.subheader("📊 Export Summary")
                summary_df = pd.DataFrame(export_summary)
                st.dataframe(summary_df, use_container_width=True)
                
                # Add download button
                col1, col2, col3 = st.columns(3)
                with col2:
                    st.download_button(
                        label=f"⬇️ Download Excel ({total_rows:,} rows)",
                        data=excel_data,
                        file_name=excel_name if excel_name.endswith('.xlsx') else excel_name + '.xlsx',
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

def prepare_tables_for_export(remove_empty: bool, strip_whitespace: bool) -> Tuple[List[Dict], List[Dict]]:
    """Prepare tables for export with filtering and cleaning"""
    tables_to_export = []
    export_summary = []
    
    for table_id, table_info in st.session_state.selected_tables.items():
        if table_info.get("selected", False):
            df = table_info.get("df")
            if df is not None:
                filtered_df = filter_and_clean_table(
                    df, table_id, remove_empty, strip_whitespace
                )
                
                if not filtered_df.empty:
                    tables_to_export.append({
                        "df": filtered_df,
                        "page": table_info["page"],
                        "table_idx": table_info["table_idx"],
                        "original_shape": table_info["shape"],
                        "filtered_shape": f"{len(filtered_df)}x{len(filtered_df.columns)}"
                    })
                    
                    export_summary.append({
                        "Page": table_info["page"],
                        "Table": table_info["table_idx"] + 1,
                        "Original": table_info["shape"],
                        "Exported": f"{len(filtered_df)}x{len(filtered_df.columns)}",
                        "Columns": len(filtered_df.columns),
                        "Rows": len(filtered_df)
                    })
    
    return tables_to_export, export_summary

def filter_and_clean_table(df: pd.DataFrame, table_id: str, 
                          remove_empty: bool, strip_whitespace: bool) -> pd.DataFrame:
    """Apply filtering and cleaning to a single table"""
    # Apply column selection
    selected_cols = [
        col for col, is_selected in st.session_state.column_selections.get(table_id, {}).items()
        if is_selected
    ]
    
    if not selected_cols:
        return pd.DataFrame()
    
    # Apply row selection
    row_selection = st.session_state.row_selections.get(table_id, {})
    
    if row_selection.get("all_rows", True):
        filtered_df = df[selected_cols].copy()
    else:
        start_row = row_selection.get("start_row", 0)
        end_row = row_selection.get("end_row", len(df) - 1)
        filtered_df = df.iloc[start_row:end_row+1][selected_cols].copy()
    
    # Apply data cleaning
    if strip_whitespace:
        # Strip whitespace from string columns
        for col in filtered_df.select_dtypes(include=['object']).columns:
            filtered_df[col] = filtered_df[col].astype(str).str.strip()
            filtered_df[col] = filtered_df[col].replace('', pd.NA)
    
    if remove_empty:
        filtered_df = filtered_df.replace(r'^\s*$', np.nan, regex=True)
        filtered_df = filtered_df.dropna(how='all')
    
    # Convert to proper data types for Excel
    filtered_df = convert_to_proper_types(filtered_df)
    
    return filtered_df

def export_to_excel(tables_to_export: List[Dict], 
                   export_mode: str, 
                   include_metadata: bool,
                   auto_format: bool,
                   excel_name: str,
                   metadata: Dict,
                   total_tables: int) -> bytes:
    """Export tables to Excel file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_excel:
        excel_path = tmp_excel.name
    
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Metadata sheet
            if include_metadata:
                total_rows = sum(len(t["df"]) for t in tables_to_export)
                metadata_df = pd.DataFrame({
                    'Property': [
                        'File Name', 'File Type', 'Total Pages', 
                        'Tables Found', 'Tables Exported', 'Total Rows Exported',
                        'Export Date', 'Extraction Mode', 'OCR Language',
                        'Data Cleaning', 'Universal Mode', 'Content Types'
                    ],
                    'Value': [
                        metadata['file_name'],
                        metadata.get('file_type', 'Unknown'),
                        metadata['total_pages'],
                        total_tables,
                        len(tables_to_export),
                        f"{total_rows:,}",
                        pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                        st.session_state.extraction_mode,
                        st.session_state.ocr_language if 'ocr_language' in st.session_state else 'N/A',
                        "Yes",
                        "Yes" if st.session_state.extraction_mode == "Universal" else "No",
                        ', '.join([f"{k}: {v}" for k, v in Counter(st.session_state.content_types.values()).items()])
                    ]
                })
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
            
            # Export based on selected mode
            if export_mode == "Each table → Separate sheet":
                export_each_table_separate(writer, tables_to_export)
            elif export_mode == "Combine all → One sheet":
                export_all_tables_combined(writer, tables_to_export)
            else:  # Group by page
                export_tables_grouped_by_page(writer, tables_to_export)
            
            # Apply formatting
            apply_excel_formatting(writer, auto_format)
        
        # Read the Excel file
        with open(excel_path, 'rb') as f:
            return f.read()
    
    finally:
        try:
            os.unlink(excel_path)
        except:
            pass

def export_each_table_separate(writer: pd.ExcelWriter, tables_to_export: List[Dict]):
    """Export each table to a separate sheet"""
    for i, table_data in enumerate(tables_to_export):
        df = table_data["df"]
        sheet_name = f"P{table_data['page']}_T{table_data['table_idx']+1}"
        # Truncate sheet name to Excel limit (31 chars)
        if len(sheet_name) > 31:
            sheet_name = sheet_name[:31]
        df.to_excel(writer, sheet_name=sheet_name, index=False)

def export_all_tables_combined(writer: pd.ExcelWriter, tables_to_export: List[Dict]):
    """Combine all tables into a single sheet"""
    all_dfs = []
    for table_data in tables_to_export:
        df = table_data["df"].copy()
        df.insert(0, 'Source_Page', table_data["page"])
        df.insert(1, 'Source_Table', table_data["table_idx"] + 1)
        all_dfs.append(df)
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_excel(writer, sheet_name='All_Data', index=False)

def export_tables_grouped_by_page(writer: pd.ExcelWriter, tables_to_export: List[Dict]):
    """Group tables by page"""
    tables_by_page = {}
    for table_data in tables_to_export:
        page = table_data["page"]
        if page not in tables_by_page:
            tables_by_page[page] = []
        tables_by_page[page].append(table_data)
    
    for page_num, page_tables in tables_by_page.items():
        page_dfs = []
        for table_data in page_tables:
            df = table_data["df"].copy()
            df.insert(0, 'Table_No', table_data["table_idx"] + 1)
            page_dfs.append(df)
        
        if page_dfs:
            combined_page_df = pd.concat(page_dfs, ignore_index=True)
            sheet_name = f"Page_{page_num}"
            if len(sheet_name) > 31:
                sheet_name = sheet_name[:31]
            combined_page_df.to_excel(writer, sheet_name=sheet_name, index=False)

def apply_excel_formatting(writer: pd.ExcelWriter, auto_format: bool):
    """Apply formatting to Excel sheets"""
    if not auto_format:
        return
    
    try:
        from openpyxl.utils import get_column_letter
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            
            # Skip metadata sheet for heavy formatting
            if sheet_name == 'Metadata':
                # Simple formatting for metadata
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = get_column_letter(column[0].column)
                    for cell in column:
                        try:
                            if cell.value:
                                max_length = max(max_length, len(str(cell.value)))
                        except:
                            pass
                    worksheet.column_dimensions[column_letter].width = min(max_length + 2, 50)
                continue
            
            # Format header row
            header_font = Font(bold=True, color="FFFFFF")
            header_fill = PatternFill(start_color="4CAF50", end_color="4CAF50", fill_type="solid")
            header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            
            for cell in worksheet[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = header_alignment
            
            # Auto-fit columns and format data
            for column in worksheet.columns:
                max_length = 0
                column_letter = get_column_letter(column[0].column)
                
                for row_idx, cell in enumerate(column, 1):
                    try:
                        if cell.value:
                            # Format based on data type
                            if isinstance(cell.value, (datetime, pd.Timestamp)):
                                cell.number_format = 'yyyy-mm-dd hh:mm:ss' if hasattr(cell.value, 'time') and cell.value.time() else 'yyyy-mm-dd'
                                cell.alignment = Alignment(horizontal="center")
                            elif isinstance(cell.value, (int, float)):
                                if isinstance(cell.value, float):
                                    cell.number_format = '#,##0.00'
                                else:
                                    cell.number_format = '#,##0'
                                cell.alignment = Alignment(horizontal="right")
                            elif isinstance(cell.value, str):
                                cell.alignment = Alignment(horizontal="left")
                            
                            # Update max length for column width
                            cell_length = len(str(cell.value))
                            if row_idx == 1:  # Header
                                cell_length = max(cell_length, len(str(cell.value)))
                            max_length = max(max_length, cell_length)
                    except:
                        pass
                
                # Set column width (with min and max limits)
                adjusted_width = min(max(max_length + 2, 8), 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
            
            # Add borders
            thin_border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
            
            for row in worksheet.iter_rows():
                for cell in row:
                    cell.border = thin_border
    except:
        pass

def render_footer():
    """Render footer"""
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #6c757d; padding: 20px;'>
            <p style='font-size: 14px;'>
                <strong>Universal Document to Excel Converter</strong> • Version 4.0.0 (Universal Edition)<br>
                Built with Streamlit, OpenCV, Tesseract OCR, and Advanced NLP<br>
                Converts ANY content - tables, prose, lists, forms - to structured Excel<br>
                © 2024 - Turn any document into actionable data
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Clear session button
    if st.button("Clear Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
