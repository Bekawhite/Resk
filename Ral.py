"""
Advanced PDF Table & Data Extractor
Supports: Digital PDFs, Scanned Documents, Handwritten Text, Images
Extracts ANY tabular data (rows & columns) from any document type
Specialized for Bank Statements and Financial Documents
"""

import streamlit as st
import pandas as pd
import tempfile
import os
import re
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from datetime import datetime
import io
import base64
import hashlib
import time
from collections import defaultdict

# ============================================================================
# PAGE CONFIGURATION - MUST BE FIRST STREAMLIT COMMAND
# ============================================================================

st.set_page_config(
    page_title="Advanced Document to Excel Converter",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# IMPORTS WITH ERROR HANDLING AND FALLBACKS
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
# CUSTOM CSS STYLES
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
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def init_session_states():
    """Initialize all session state variables"""
    defaults = {
        'pdf_uploaded': False,
        'tables_data': {},
        'pdf_metadata': {},
        'selected_tables': {},
        'column_selections': {},
        'row_selections': {},
        'all_columns': {},
        'extraction_mode': "Bank Statement",
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
        'tesseract_available': TESSERACT_AVAILABLE
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_file_hash(file_content: bytes) -> str:
    """Generate hash for file content"""
    return hashlib.md5(file_content).hexdigest()

def convert_to_proper_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert DataFrame columns to proper data types for Excel compatibility
    Handles numbers, dates, currencies, and text intelligently
    """
    df_converted = df.copy()
    
    for col in df_converted.columns:
        # Skip if column is empty
        if df_converted[col].isna().all():
            continue
            
        # Try to convert to numeric
        try:
            # First, clean the data: remove currency symbols and commas
            if df_converted[col].dtype == 'object':
                # Check if column contains currency patterns
                sample = df_converted[col].dropna().iloc[0] if not df_converted[col].dropna().empty else ""
                if isinstance(sample, str):
                    # Check for various currency symbols
                    currency_symbols = ['$', '€', '£', '¥', '₹', '₽', '₿', '₩', '₦', '₱', '﷼', '៛', '₫', '฿', '₲', '₴', '₸', '₺', '₼', '₡', '₭', '₮', '₦', '₱', '₲', '₴', '₸', '₺', '₼', '₡']
                    if any(symbol in sample for symbol in currency_symbols):
                        # Remove currency symbols and commas, then try numeric conversion
                        cleaned = df_converted[col].astype(str).str.replace(r'[$€£¥₹₽₿₩₦₱﷼៛₫฿₲₴₸₺₼₡₭₮]', '', regex=True)
                        cleaned = cleaned.str.replace(',', '')
                        cleaned = cleaned.str.replace(' ', '')
                        numeric_col = pd.to_numeric(cleaned, errors='coerce')
                        if not numeric_col.isna().all():
                            df_converted[col] = numeric_col
                            continue
                    
                    # Check for percentage
                    if '%' in sample:
                        cleaned = df_converted[col].astype(str).str.replace('%', '')
                        numeric_col = pd.to_numeric(cleaned, errors='coerce') / 100
                        if not numeric_col.isna().all():
                            df_converted[col] = numeric_col
                            continue
            
            # Direct numeric conversion
            numeric_col = pd.to_numeric(df_converted[col], errors='coerce')
            if not numeric_col.isna().all():  # If at least some values converted successfully
                df_converted[col] = numeric_col
                continue
        except:
            pass
        
        # Try to convert to datetime
        try:
            if df_converted[col].dtype == 'object':
                # Try infer format
                date_col = pd.to_datetime(df_converted[col], errors='coerce', infer_datetime_format=True)
                if not date_col.isna().all():
                    df_converted[col] = date_col
        except:
            pass
        
        # If all else fails, ensure it's string type for Excel compatibility
        if df_converted[col].dtype == 'object':
            df_converted[col] = df_converted[col].astype(str).replace('nan', '')
    
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
        'most_tables_count': 0
    }
    
    all_rows = []
    all_cols = []
    
    for page_num, tables in tables_data.items():
        page_table_count = len(tables)
        if page_table_count > stats['most_tables_count']:
            stats['most_tables_count'] = page_table_count
            stats['page_with_most_tables'] = page_num
        
        for table in tables:
            rows, cols = table.shape
            all_rows.append(rows)
            all_cols.append(cols)
            stats['total_rows'] += rows
            stats['total_columns'] += cols
            stats['max_rows'] = max(stats['max_rows'], rows)
            stats['max_columns'] = max(stats['max_columns'], cols)
    
    if stats['total_tables'] > 0:
        stats['avg_rows_per_table'] = stats['total_rows'] / stats['total_tables']
        stats['avg_columns_per_table'] = stats['total_columns'] / stats['total_tables']
    
    return stats

# ============================================================================
# ENHANCED BANK STATEMENT EXTRACTION FUNCTIONS
# ============================================================================

def extract_bank_statement_pattern(text_lines: List[str]) -> List[pd.DataFrame]:
    """
    Specifically designed to extract bank statement transaction data
    Handles the format: Date | Value Date | Reference | Description | Debit | Credit | Balance
    """
    transactions = []
    current_transaction = []
    headers_detected = False
    header_patterns = ['date', 'value date', 'ref', 'description', 'debit', 'credit', 'balance', 
                       'transaction', 'particulars', 'withdrawal', 'deposit', 'amount']
    
    # First pass: try to identify the header row
    header_row_index = -1
    for i, line in enumerate(text_lines):
        line_lower = line.lower().strip()
        header_matches = sum(1 for pattern in header_patterns if pattern in line_lower)
        if header_matches >= 2:  # If we find at least 2 header keywords
            headers_detected = True
            header_row_index = i
            break
    
    # Second pass: extract transactions
    for i, line in enumerate(text_lines):
        if i <= header_row_index:  # Skip header and above
            continue
            
        line = line.strip()
        if not line:
            continue
            
        # Check if line contains transaction data (has date pattern and amounts)
        date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'  # DD/MM/YYYY or DD-MM-YYYY
        amount_pattern = r'[\d,]+\.\d{2}'  # Amounts with decimals
        ugx_pattern = r'UGX\s*[\d,]+\.\d{2}'  # UGX amount pattern
        
        has_date = re.search(date_pattern, line)
        has_amount = re.search(amount_pattern, line) or re.search(ugx_pattern, line)
        
        if has_date and has_amount:
            # This is likely a transaction line
            if current_transaction:
                # Process the accumulated transaction
                processed = process_transaction_line(' '.join(current_transaction))
                if processed:
                    transactions.append(processed)
                current_transaction = []
            current_transaction = [line]
        elif current_transaction and line:
            # Continuation of previous transaction (multi-line description)
            current_transaction.append(line)
        elif not current_transaction and has_amount:
            # Some statements might not have dates in every line
            current_transaction = [line]
    
    # Don't forget the last transaction
    if current_transaction:
        processed = process_transaction_line(' '.join(current_transaction))
        if processed:
            transactions.append(processed)
    
    # Convert to DataFrame
    if transactions:
        df = pd.DataFrame(transactions)
        # Try to split into proper columns
        df = split_transaction_columns(df)
        return [df] if not df.empty else []
    
    return []

def process_transaction_line(line: str) -> Dict:
    """
    Process a single transaction line and extract components
    """
    transaction = {
        'raw_text': line,
        'date': '',
        'value_date': '',
        'reference': '',
        'description': '',
        'debit': '',
        'credit': '',
        'balance': ''
    }
    
    # Extract date patterns
    date_pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
    dates = re.findall(date_pattern, line)
    
    if len(dates) >= 1:
        transaction['date'] = dates[0]
    if len(dates) >= 2:
        transaction['value_date'] = dates[1]
    
    # Extract UGX amounts
    ugx_pattern = r'UGX\s*([\d,]+\.\d{2})'
    ugx_matches = re.findall(ugx_pattern, line)
    
    # Extract regular amounts
    amount_pattern = r'([\d,]+\.\d{2})'
    amounts = re.findall(amount_pattern, line)
    
    # Combine and clean amounts
    all_amounts = ugx_matches + amounts
    all_amounts = [amt.replace(',', '') for amt in all_amounts]
    
    # Bank statements typically have 2-3 amounts: debit/credit + balance
    if len(all_amounts) >= 2:
        # Check for debit/credit indicators
        if 'dr' in line.lower() or 'debit' in line.lower() or 'withdrawal' in line.lower():
            transaction['debit'] = all_amounts[0]
            if len(all_amounts) > 1:
                transaction['balance'] = all_amounts[-1]
        elif 'cr' in line.lower() or 'credit' in line.lower() or 'deposit' in line.lower():
            transaction['credit'] = all_amounts[0]
            if len(all_amounts) > 1:
                transaction['balance'] = all_amounts[-1]
        else:
            # Try to guess based on typical format
            if len(all_amounts) == 2:
                # Assume first is amount, second is balance
                # Try to determine if it's debit or credit from context
                if float(all_amounts[0]) > 0:
                    # Could be either, but we'll put in debit by default
                    transaction['debit'] = all_amounts[0]
                transaction['balance'] = all_amounts[1]
            elif len(all_amounts) >= 3:
                transaction['debit'] = all_amounts[0]
                transaction['credit'] = all_amounts[1]
                transaction['balance'] = all_amounts[-1]
    
    # Extract reference number (if present)
    ref_pattern = r'([A-Z0-9]{8,})'  # Alphanumeric code of 8+ characters
    refs = re.findall(ref_pattern, line)
    if refs and not any(word in line.lower() for word in ['ugx', 'date']):
        transaction['reference'] = refs[0]
    
    # Extract description (everything between dates and amounts)
    description = line
    # Remove dates
    for date in dates:
        description = description.replace(date, '')
    # Remove amounts and UGX
    for amt in ugx_matches + amounts:
        description = description.replace(amt, '')
    description = re.sub(r'UGX', '', description, flags=re.IGNORECASE)
    # Clean up
    description = re.sub(r'\s+', ' ', description).strip()
    # Remove common transaction codes and indicators
    description = re.sub(r'\b(dr|cr|debit|credit|withdrawal|deposit)\b', '', description, flags=re.IGNORECASE).strip()
    # Remove reference numbers
    for ref in refs:
        description = description.replace(ref, '')
    description = re.sub(r'\s+', ' ', description).strip()
    
    transaction['description'] = description if description else 'No Description'
    
    return transaction

def split_transaction_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to split the raw text into proper columns
    """
    if df.empty:
        return df
    
    # Create a new DataFrame with proper columns
    structured_data = []
    
    for _, row in df.iterrows():
        transaction = row['raw_text']
        
        # Try to split by multiple spaces (typical in bank statements)
        parts = re.split(r'\s{2,}', transaction)
        
        if len(parts) >= 3:  # At least date, description, amount
            structured_row = {
                'Date': '',
                'Value Date': '',
                'Reference': '',
                'Description': '',
                'Debit': '',
                'Credit': '',
                'Balance': ''
            }
            
            # First part is usually date
            if parts and re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', parts[0]):
                structured_row['Date'] = parts[0]
            
            # Last part is usually balance
            if parts and re.search(r'[\d,]+\.\d{2}', parts[-1]):
                structured_row['Balance'] = parts[-1]
            
            # Second last is usually amount
            if len(parts) >= 2 and re.search(r'[\d,]+\.\d{2}', parts[-2]):
                amount = parts[-2]
                
                # Check if it's debit or credit
                if 'dr' in transaction.lower() or 'debit' in transaction.lower():
                    structured_row['Debit'] = amount
                elif 'cr' in transaction.lower() or 'credit' in transaction.lower():
                    structured_row['Credit'] = amount
                else:
                    # Try to determine by position or context
                    if len(parts) >= 3:
                        structured_row['Debit'] = amount
            
            # Everything in between is description and reference
            if len(parts) > 3:
                desc_parts = parts[1:-2]
                # Check for reference in description
                for part in desc_parts:
                    if re.search(r'[A-Z0-9]{8,}', part):
                        structured_row['Reference'] = part
                        desc_parts.remove(part)
                structured_row['Description'] = ' '.join(desc_parts).strip()
            elif len(parts) > 2:
                structured_row['Description'] = parts[1]
            
            structured_data.append(structured_row)
    
    if structured_data:
        result_df = pd.DataFrame(structured_data)
        # Remove empty columns
        result_df = result_df.loc[:, (result_df != '').any(axis=0)]
        return result_df
    
    # If splitting fails, return original with basic parsing
    result_df = df.copy()
    if 'date' in result_df.columns:
        result_df['Date'] = result_df['date']
    if 'description' in result_df.columns:
        result_df['Description'] = result_df['description']
    if 'debit' in result_df.columns:
        result_df['Debit'] = result_df['debit']
    if 'credit' in result_df.columns:
        result_df['Credit'] = result_df['credit']
    if 'balance' in result_df.columns:
        result_df['Balance'] = result_df['balance']
    
    # Select only the standard columns that exist
    standard_cols = ['Date', 'Value Date', 'Reference', 'Description', 'Debit', 'Credit', 'Balance']
    existing_cols = [col for col in standard_cols if col in result_df.columns]
    
    return result_df[existing_cols] if existing_cols else result_df

# ============================================================================
# ENHANCED OCR EXTRACTION FOR BANK STATEMENTS
# ============================================================================

def enhanced_bank_statement_ocr(image):
    """
    Enhanced OCR specifically for bank statements
    """
    if not TESSERACT_AVAILABLE:
        return []
    
    try:
        # Preprocess image for better OCR
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Enhance image for better text recognition
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Remove noise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Additional preprocessing for bank statements
        # Detect and enhance table lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, vertical_kernel)
        
        table_structure = cv2.add(horizontal_lines, vertical_lines)
        enhanced = cv2.add(enhanced, table_structure)
        
        # Custom OCR configuration for bank statements
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,/-:UGX '
        
        # Get OCR data
        data = pytesseract.image_to_data(enhanced, config=custom_config, output_type=pytesseract.Output.DICT)
        
        # Group by line and sort by vertical position
        lines = {}
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 30:  # Confidence threshold
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
        
        # Sort words in each line by x-coordinate
        sorted_lines = []
        for line_num in sorted(lines.keys()):
            words = lines[line_num]
            words.sort(key=lambda x: x['x'])
            line_text = ' '.join([word['text'] for word in words])
            if line_text.strip():
                sorted_lines.append(line_text)
        
        # Use bank statement specific extraction
        tables = extract_bank_statement_pattern(sorted_lines)
        
        return tables
        
    except Exception as e:
        if st.session_state.debug_mode:
            st.warning(f"Enhanced OCR error: {e}")
        return []

# ============================================================================
# SIMPLIFIED PATTERN DETECTION (WITHOUT HEAVY ML DEPENDENCIES)
# ============================================================================

def simple_pattern_detection(image):
    """
    Simplified pattern detection that doesn't rely on heavy ML libraries
    Returns detected regions as list of (x, y, w, h) tuples
    """
    regions = []
    
    if not CV_AVAILABLE:
        return regions
    
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and shape
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 100 and h > 50:  # Minimum region size
                aspect_ratio = w / h
                if 0.5 < aspect_ratio < 3.0:  # Reasonable aspect ratio for tables
                    regions.append({'x': x, 'y': y, 'width': w, 'height': h, 'confidence': 0.5})
        
        return regions
        
    except Exception as e:
        return []

# ============================================================================
# BASIC OCR EXTRACTION FUNCTIONS
# ============================================================================

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
            if int(data['conf'][i]) > 30:  # Reasonable confidence
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
        
        # Convert to simple table format
        table_data = []
        for line_num in sorted(lines.keys()):
            words = lines[line_num]
            words.sort(key=lambda x: x['x'])
            row = [word['text'] for word in words]
            if row:
                table_data.append(row)
        
        if len(table_data) >= 2:
            df = pd.DataFrame(table_data)
            return [df]
        
        return []
        
    except Exception as e:
        return []

# ============================================================================
# SIMPLE PDF EXTRACTION
# ============================================================================

def extract_pdf_tables(pdf_path: str, page_num: int) -> List[pd.DataFrame]:
    """
    Extract tables from PDF using available libraries
    """
    tables = []
    
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
                                tables.append(df)
                    
                    # If no tables found, try text extraction and parse as bank statement
                    if not tables:
                        text = page.extract_text()
                        if text:
                            lines = text.split('\n')
                            bank_tables = extract_bank_statement_pattern(lines)
                            tables.extend(bank_tables)
        except Exception as e:
            pass
    
    # Try camelot if available
    if CAMELOT_AVAILABLE:
        try:
            tables_camelot = camelot.read_pdf(pdf_path, pages=str(page_num), flavor='lattice')
            for table in tables_camelot:
                df = table.df
                if not df.empty and len(df) > 1:
                    tables.append(df)
            
            if not tables:
                tables_camelot = camelot.read_pdf(pdf_path, pages=str(page_num), flavor='stream')
                for table in tables_camelot:
                    df = table.df
                    if not df.empty and len(df) > 1:
                        tables.append(df)
        except Exception as e:
            pass
    
    return tables

# ============================================================================
# MAIN EXTRACTION FUNCTION
# ============================================================================

def extract_tables_from_document(file_path: str, pages: List[int], mode: str) -> Dict[int, List[pd.DataFrame]]:
    """
    Enhanced extraction function with bank statement support
    """
    tables_by_page = {}
    file_ext = file_path.split('.')[-1].lower()
    
    for page_num in pages:
        page_tables = []
        
        # Handle PDF files
        if file_ext == 'pdf':
            # Try PDF extraction methods
            pdf_tables = extract_pdf_tables(file_path, page_num)
            
            # If standard extraction fails or returns few tables, try bank statement specific
            if not pdf_tables or (pdf_tables and len(pdf_tables[0]) < 3):
                # Try to extract text and parse as bank statement
                if PDFPLUMBER_AVAILABLE:
                    try:
                        with pdfplumber.open(file_path) as pdf:
                            if page_num <= len(pdf.pages):
                                page = pdf.pages[page_num - 1]
                                text = page.extract_text()
                                if text:
                                    lines = text.split('\n')
                                    bank_tables = extract_bank_statement_pattern(lines)
                                    page_tables.extend(bank_tables)
                    except Exception as e:
                        if st.session_state.debug_mode:
                            st.warning(f"Text extraction error: {e}")
            
            page_tables.extend(pdf_tables)
        
        # For images or scanned PDFs, use enhanced OCR
        if (not page_tables or (page_tables and len(page_tables[0]) < 3)) and TESSERACT_AVAILABLE:
            try:
                # Convert PDF page to image if needed
                if file_ext == 'pdf' and PDF2IMAGE_AVAILABLE:
                    images = pdf2image.convert_from_path(file_path, first_page=page_num, 
                                                        last_page=page_num, dpi=300)  # Higher DPI for better OCR
                    if images:
                        image = np.array(images[0])
                else:
                    # Load image directly
                    if PIL_AVAILABLE:
                        image = np.array(Image.open(file_path))
                    else:
                        image = cv2.imread(file_path)
                
                if image is not None:
                    # Try enhanced bank statement OCR first
                    if mode == "Bank Statement" or mode == "All Methods":
                        bank_tables = enhanced_bank_statement_ocr(image)
                        page_tables.extend(bank_tables)
                    
                    # If still no tables, try basic OCR
                    if not page_tables:
                        # Simple pattern detection
                        regions = simple_pattern_detection(image)
                        
                        if regions:
                            for region in regions:
                                # Extract region
                                x, y, w, h = region['x'], region['y'], region['width'], region['height']
                                roi = image[y:y+h, x:x+w]
                                
                                # OCR the region
                                ocr_tables = basic_ocr_extraction(roi)
                                page_tables.extend(ocr_tables)
                        
                        # If no regions found, try OCR on whole image
                        if not page_tables:
                            ocr_tables = basic_ocr_extraction(image)
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
                if not table.empty and len(table) >= 2:  # At least 2 rows
                    cleaned_tables.append(table)
        
        if cleaned_tables:
            tables_by_page[page_num] = cleaned_tables
    
    return tables_by_page

# ============================================================================
# DATA ANALYSIS FUNCTIONS
# ============================================================================

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
            'reference': ['reference', 'ref', 'ref no', 'reference number']
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
                    elif any(keyword in sample_text for keyword in keywords):
                        matches.append(pattern_name)
        
        # Remove duplicates and limit to top 3
        matches = list(dict.fromkeys(matches))[:3]
        column_analysis[col] = matches
    
    return column_analysis

# ============================================================================
# EXCEL EXPORT FUNCTIONS
# ============================================================================

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
                        'Data Cleaning', 'Bank Statement Mode'
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
                        "Yes" if st.session_state.extraction_mode == "Bank Statement" else "No"
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

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_sidebar():
    """Render the sidebar UI"""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/ms-excel.png", width=80)
        st.header("📁 Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif', 'webp'],
            help="Upload any document containing rows and columns - bank statements, tables, forms, lists, etc.",
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
            **Advanced Document to Excel Converter**
            
            **Features:**
            - Bank Statement optimized extraction
            - Detects ANY rows & columns pattern
            - Digital PDF table extraction
            - Handwritten text recognition
            - Scanned document processing
            - Forms and lists extraction
            - Multi-format support
            - Excel-ready output
            
            **Supported Formats:**
            - PDF (digital & scanned)
            - Images (PNG, JPG, JPEG, TIFF, BMP, GIF, WebP)
            
            **Version:** 3.2.0 (Bank Statement Edition)
            """)
        
        return uploaded_file

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

def render_extraction_settings() -> Tuple[str, bool, int, int, List[int], str, bool]:
    """Render extraction settings with bank statement option"""
    st.header("🎯 Extraction Settings")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        extraction_mode = st.selectbox(
            "Extraction Mode",
            options=[
                "Bank Statement Mode (Optimized)",
                "Auto-detect (General)",
                "PDF Tables Only",
                "OCR Only (Images/Scans)",
                "All Methods"
            ],
            index=0,  # Default to Bank Statement Mode
            help="Bank Statement Mode is optimized for financial statements with date, description, debit, credit, balance columns."
        )
        
        # Map selection to mode string
        mode_map = {
            "Bank Statement Mode (Optimized)": "Bank Statement",
            "Auto-detect (General)": "Auto-detect",
            "PDF Tables Only": "PDF Tables",
            "OCR Only (Images/Scans)": "OCR Only",
            "All Methods": "All Methods"
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
                2, 1000, 3,  # Lowered to 3 for bank statements
                help="Minimum number of rows to consider as a valid table"
            )
            
            min_cols = st.number_input(
                "Minimum columns", 
                2, 50, 3,  # Lowered to 3 for bank statements
                help="Minimum number of columns to consider as a valid table"
            )
        
        with col2:
            ocr_lang = st.selectbox(
                "OCR Language",
                options=['eng', 'fra', 'deu', 'spa', 'ita', 'por'],
                index=0,
                help="Select language for OCR"
            )
            
            enhance_handwriting = st.checkbox("Enhance image", value=True,
                                             help="Apply additional preprocessing for better OCR")
    
    return extraction_mode, debug_mode, min_rows, min_cols, selected_pages, ocr_lang, enhance_handwriting

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
                    tags = " • ".join(column_analysis[col])
                    st.markdown(f"""
                    <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px;'>
                        <strong>{col[:20]}</strong><br>
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

def render_table_preview(table_id: str, table: pd.DataFrame):
    """Render table preview"""
    if st.session_state.get(f"show_preview_{table_id}", False):
        with st.container():
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
            else:
                st.warning("No data to preview")
    
    st.markdown("---")

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

def render_welcome_screen():
    """Render welcome screen when no document is uploaded"""
    st.markdown("""
    <div class="info-box">
    <h1>📊 Advanced Document to Excel Converter</h1>
    <p style='font-size: 18px;'>Extract <strong>ANY rows and columns</strong> from any document type - printed, handwritten, scanned, or digital.</p>
    <p><strong>Specialized for Bank Statements and Financial Documents</strong></p>
    <p><strong>Supported Formats:</strong> PDF, PNG, JPG, JPEG, TIFF, BMP, GIF, WebP</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature showcase
    st.markdown("### ✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🏦</div>
            <div class="feature-title">Bank Statement Optimized</div>
            <div class="feature-description">Specialized extraction for financial statements with date, description, debit, credit, and balance columns</div>
        </div>
        
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Intelligent Pattern Detection</div>
            <div class="feature-description">Automatically detects any rows & columns pattern - tables, forms, lists, invoices, receipts</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">✍️</div>
            <div class="feature-title">Handwriting Recognition</div>
            <div class="feature-description">Advanced OCR with preprocessing for handwritten forms and documents</div>
        </div>
        
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Excel-Ready Output</div>
            <div class="feature-description">Numbers and dates properly formatted for Excel functions and analysis</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">💰</div>
            <div class="feature-title">Multi-Currency Support</div>
            <div class="feature-description">Handles UGX, USD, EUR, GBP and other currency formats automatically</div>
        </div>
        
        <div class="feature-card">
            <div class="feature-icon">🔄</div>
            <div class="feature-title">Flexible Export Options</div>
            <div class="feature-description">Export as separate sheets, combined, or grouped by page</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start guide
    st.markdown("### 🚀 Quick Start Guide")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="progress-step">
            <div class="step-number">1</div>
            <div class="step-text">Upload Document</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="progress-step">
            <div class="step-number">2</div>
            <div class="step-text">Choose Bank Statement Mode</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="progress-step">
            <div class="step-number">3</div>
            <div class="step-text">Select Pages</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="progress-step">
            <div class="step-number">4</div>
            <div class="step-text">Export to Excel</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Tips section
    with st.expander("💡 Tips for Best Results", expanded=False):
        st.markdown("""
        - **For bank statements**: Use "Bank Statement Mode" for best results
        - **For clear tables**: Use "Auto-detect" mode for faster extraction
        - **For handwritten forms**: Use "OCR Only" mode
        - **For scanned documents**: Ensure good lighting and straight angle
        - **For multi-currency statements**: The system automatically detects UGX, USD, EUR formats
        """)
    
    st.markdown("---")
    st.markdown("*Upload a document using the sidebar to begin extraction*")

def render_footer():
    """Render footer"""
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #6c757d; padding: 20px;'>
            <p style='font-size: 14px;'>
                <strong>Advanced Document to Excel Converter</strong> • Version 3.2.0 (Bank Statement Edition)<br>
                Built with Streamlit, OpenCV, Tesseract OCR, and pdfplumber<br>
                Specialized for Financial Documents and Bank Statements<br>
                © 2024 - Extract any rows & columns from any document
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
# MAIN APPLICATION
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
        st.sidebar.info("💡 Install Tesseract OCR for better results with scanned documents")
    
    # Main content
    if st.session_state.pdf_uploaded and uploaded_file:
        render_document_info()
        
        # Get extraction settings
        extraction_mode, debug_mode, min_rows, min_cols, selected_pages, ocr_lang, enhance_handwriting = render_extraction_settings()
        
        # Store OCR language in session state
        st.session_state.ocr_language = ocr_lang
        
        # Extract button
        if selected_pages and st.button("🔍 Extract Data", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner(f"Analyzing {len(selected_pages)} pages for row/column patterns..."):
                # Save file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp:
                    tmp.write(uploaded_file.getvalue())
                    file_path = tmp.name
                
                try:
                    # Update progress
                    status_text.text("Processing document...")
                    progress_bar.progress(20)
                    
                    # Extract tables
                    tables_by_page = extract_tables_from_document(
                        file_path, 
                        selected_pages, 
                        st.session_state.extraction_mode
                    )
                    
                    progress_bar.progress(60)
                    status_text.text("Analyzing extracted data...")
                    
                    # Filter tables based on criteria
                    filtered_tables, total_tables_found, tables_ignored = filter_tables_by_size(
                        tables_by_page, min_rows, min_cols
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
                    display_extraction_results(total_tables_found, filtered_tables, tables_ignored, min_rows, min_cols)
                    
                    # Show debug info if enabled
                    if debug_mode:
                        with st.expander("🔧 Debug Information"):
                            st.json({
                                'extraction_mode': st.session_state.extraction_mode,
                                'pages_scanned': len(selected_pages),
                                'tables_found': total_tables_found,
                                'tables_ignored': tables_ignored,
                                'pages_with_tables': list(filtered_tables.keys()),
                                'min_requirements': {
                                    'min_rows': min_rows,
                                    'min_cols': min_cols
                                },
                                'ocr_settings': {
                                    'language': ocr_lang,
                                    'enhance_handwriting': enhance_handwriting
                                },
                                'libraries_available': {
                                    'opencv': st.session_state.cv_available,
                                    'tesseract': st.session_state.tesseract_available,
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

def display_extraction_results(total_tables_found: int, filtered_tables: Dict, tables_ignored: int, min_rows: int, min_cols: int):
    """Display extraction results"""
    if total_tables_found > 0:
        st.markdown(f"""
        <div class="success-box">
        <h3>✅ Extraction Complete!</h3>
        <p>Found <strong>{total_tables_found}</strong> tables/patterns with at least <strong>{min_rows}</strong> rows and <strong>{min_cols}</strong> columns</p>
        <p>Pages with data: <strong>{len(filtered_tables)}</strong></p>
        <p>Tables ignored: <strong>{tables_ignored}</strong> (didn't meet size criteria)</p>
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
            Largest table: {stats['max_rows']} rows × {stats['max_columns']} columns
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ No data found with at least {min_rows} rows and {min_cols} columns. Try adjusting the minimum requirements or extraction mode.")
        
        # Provide suggestions
        st.info("""
        **Suggestions:**
        - Try "Bank Statement Mode" for financial documents
        - Reduce minimum rows requirement (try 2 or 3)
        - Enable Debug Mode to see what text is being extracted
        - Make sure the document is clear and well-lit
        - For scanned documents, ensure the image is straight and not skewed
        """)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
