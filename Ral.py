"""
Advanced PDF Table & Data Extractor
Supports: Digital PDFs, Scanned Documents, Handwritten Text, Images
Extracts tabular data from any document type
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

# ============================================================================
# IMPORTS WITH ERROR HANDLING
# ============================================================================

# Computer Vision and OCR imports
try:
    import cv2
    import pytesseract
    from PIL import Image
    import pdf2image
    CV_AVAILABLE = True
except ImportError as e:
    CV_AVAILABLE = False
    st.warning(f"Some features may be limited: {e}. Install opencv-python, pytesseract, pillow, pdf2image for full functionality.")

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
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Advanced PDF Table Extractor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        'extraction_mode': "Auto-detect",
        'processing_history': [],
        'current_file_hash': None,
        'extraction_stats': {},
        'ocr_language': 'eng',
        'debug_mode': False
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
                # Common date formats
                date_formats = [
                    '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%Y.%m.%d',
                    '%d.%m.%Y', '%m.%d.%Y', '%Y%m%d', '%d%m%Y', '%m%d%Y',
                    '%b %d, %Y', '%d %b %Y', '%B %d, %Y', '%d %B %Y',
                    '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M', '%m/%d/%Y %I:%M %p'
                ]
                
                for fmt in date_formats:
                    try:
                        date_col = pd.to_datetime(df_converted[col], format=fmt, errors='coerce')
                        if not date_col.isna().all():
                            df_converted[col] = date_col
                            break
                    except:
                        continue
                else:
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
# IMAGE PROCESSING FUNCTIONS
# ============================================================================

def preprocess_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Advanced image preprocessing for better OCR results
    Handles various image quality issues
    """
    if not CV_AVAILABLE:
        return image
    
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive histogram equalization for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 30, 7, 21)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Remove small noise
        kernel = np.ones((1,1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Deskew if needed
        coords = np.column_stack(np.where(cleaned > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if abs(angle) > 0.5:
                (h, w) = cleaned.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                cleaned = cv2.warpAffine(cleaned, M, (w, h), 
                                        flags=cv2.INTER_CUBIC, 
                                        borderMode=cv2.BORDER_REPLICATE)
        
        return cleaned
        
    except Exception as e:
        st.warning(f"Image preprocessing warning: {e}")
        return image

def detect_table_structure_using_lines(image: np.ndarray) -> tuple:
    """
    Detect table structure by finding horizontal and vertical lines
    Returns cell coordinates and processed image
    """
    if not CV_AVAILABLE:
        return [], image
    
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Combine lines
        table_structure = cv2.add(horizontal_lines, vertical_lines)
        
        # Dilate to connect nearby lines
        kernel = np.ones((3,3), np.uint8)
        table_structure = cv2.dilate(table_structure, kernel, iterations=1)
        
        # Find contours to detect cells
        contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cells = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 20 and h > 20:  # Filter out very small contours
                cells.append((x, y, w, h))
        
        return cells, thresh
        
    except Exception as e:
        st.warning(f"Table structure detection error: {e}")
        return [], image

# ============================================================================
# OCR EXTRACTION FUNCTIONS
# ============================================================================

def extract_handwritten_table_with_ocr(image_path: str, 
                                       lang: str = 'eng', 
                                       confidence_threshold: int = 50,
                                       enhance: bool = True) -> List[pd.DataFrame]:
    """
    Extract tables from handwritten or scanned documents using advanced OCR
    """
    if not CV_AVAILABLE:
        st.error("OpenCV and Tesseract are required for handwritten text extraction")
        return []
    
    tables_found = []
    
    try:
        # Convert PDF to images if needed
        if image_path.lower().endswith('.pdf'):
            images = pdf2image.convert_from_path(image_path, dpi=300)
        else:
            # Single image file
            images = [Image.open(image_path)]
        
        for page_num, image in enumerate(images):
            # Convert PIL to OpenCV
            open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Preprocess image
            if enhance:
                processed = preprocess_image_for_ocr(open_cv_image)
            else:
                # Basic preprocessing
                if len(open_cv_image.shape) == 3:
                    processed = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
                else:
                    processed = open_cv_image.copy()
            
            # Method 1: Line-based table detection
            cells, thresh = detect_table_structure_using_lines(processed)
            
            if len(cells) > 5:  # Found potential table with lines
                tables_found.extend(extract_tables_from_cells(processed, cells, lang))
            
            # Method 2: OCR with column detection for documents without lines
            else:
                tables = extract_tables_without_lines(processed, lang, confidence_threshold)
                tables_found.extend(tables)
        
        return tables_found
        
    except Exception as e:
        st.error(f"OCR extraction error: {str(e)}")
        return []

def extract_tables_from_cells(processed: np.ndarray, cells: List[tuple], lang: str) -> List[pd.DataFrame]:
    """Extract tables from detected cell regions"""
    tables_found = []
    
    # Sort cells by row and column
    cells.sort(key=lambda x: (x[1], x[0]))
    
    # Group cells into rows
    rows = []
    current_row = []
    current_y = cells[0][1]
    y_threshold = 30  # Max vertical distance to consider same row
    
    for cell in cells:
        if abs(cell[1] - current_y) < y_threshold:
            current_row.append(cell)
        else:
            if current_row:
                # Sort row by x coordinate
                current_row.sort(key=lambda x: x[0])
                rows.append(current_row)
            current_row = [cell]
            current_y = cell[1]
    
    if current_row:
        current_row.sort(key=lambda x: x[0])
        rows.append(current_row)
    
    # Extract text from each cell
    table_data = []
    for row in rows:
        row_data = []
        for cell in row:
            x, y, w, h = cell
            # Add padding
            padding = 5
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(processed.shape[1] - x, w + 2*padding)
            h = min(processed.shape[0] - y, h + 2*padding)
            
            # Extract cell region
            cell_image = processed[y:y+h, x:x+w]
            
            # Perform OCR on cell with custom config
            custom_config = f'--psm 7 --oem 3 -l {lang}'
            text = pytesseract.image_to_string(cell_image, config=custom_config).strip()
            
            # Only add if text passes confidence threshold
            if text:
                row_data.append(text)
            else:
                row_data.append('')
        
        # Only add rows with at least one non-empty cell
        if any(row_data):
            table_data.append(row_data)
    
    if len(table_data) > 1:
        df = create_dataframe_from_table_data(table_data)
        if not df.empty:
            tables_found.append(df)
    
    return tables_found

def extract_tables_without_lines(processed: np.ndarray, lang: str, confidence_threshold: int) -> List[pd.DataFrame]:
    """Extract tables from images without visible lines using text clustering"""
    tables_found = []
    
    # Perform OCR on whole page with detailed output
    custom_config = f'--psm 6 --oem 3 -l {lang}'
    ocr_data = pytesseract.image_to_data(processed, 
                                         output_type=pytesseract.Output.DICT,
                                         config=custom_config)
    
    # Group text by lines
    lines = {}
    for i in range(len(ocr_data['text'])):
        conf = float(ocr_data['conf'][i])
        if conf > confidence_threshold:  # Only use high confidence text
            text = ocr_data['text'][i].strip()
            if text:
                line_num = ocr_data['line_num'][i]
                if line_num not in lines:
                    lines[line_num] = []
                lines[line_num].append({
                    'text': text,
                    'left': ocr_data['left'][i],
                    'top': ocr_data['top'][i],
                    'width': ocr_data['width'][i],
                    'height': ocr_data['height'][i],
                    'conf': conf
                })
    
    # Sort lines by vertical position
    sorted_lines = sorted(lines.items(), key=lambda x: x[1][0]['top'] if x[1] else 0)
    
    # Detect columns based on horizontal positions using clustering
    all_words = []
    for line_num, words in sorted_lines:
        for word in words:
            all_words.append(word)
    
    if len(all_words) > 5:
        # Cluster words into columns
        left_positions = [w['left'] for w in all_words]
        left_positions.sort()
        
        # Simple clustering based on gaps
        column_boundaries = []
        threshold = 50  # Min distance between columns
        
        if left_positions:
            current_boundary = left_positions[0]
            for pos in left_positions[1:]:
                if pos - current_boundary > threshold:
                    column_boundaries.append((current_boundary + pos) // 2)
                    current_boundary = pos
            
            # Add final boundary
            column_boundaries.append(left_positions[-1] + 100)
        
        # Build table structure
        table_data = build_table_from_words(sorted_lines, column_boundaries)
        
        if len(table_data) > 1:
            df = create_dataframe_from_table_data(table_data)
            if not df.empty:
                tables_found.append(df)
    
    return tables_found

def build_table_from_words(sorted_lines: List, column_boundaries: List) -> List[List[str]]:
    """Build table structure from word data"""
    table_data = []
    current_line_words = []
    current_line_num = sorted_lines[0][0] if sorted_lines else 0
    
    for line_num, words in sorted_lines:
        if line_num != current_line_num and current_line_words:
            # Sort words in line by left position
            current_line_words.sort(key=lambda x: x['left'])
            
            # Assign words to columns
            row = [''] * len(column_boundaries)
            for word in current_line_words:
                # Find which column this word belongs to
                for i, boundary in enumerate(column_boundaries):
                    if word['left'] <= boundary:
                        if row[i]:
                            row[i] += ' ' + word['text']
                        else:
                            row[i] = word['text']
                        break
            
            table_data.append(row)
            current_line_words = []
            current_line_num = line_num
        
        current_line_words.extend(words)
    
    # Add last line
    if current_line_words:
        current_line_words.sort(key=lambda x: x['left'])
        row = [''] * len(column_boundaries)
        for word in current_line_words:
            for i, boundary in enumerate(column_boundaries):
                if word['left'] <= boundary:
                    if row[i]:
                        row[i] += ' ' + word['text']
                    else:
                        row[i] = word['text']
                    break
        table_data.append(row)
    
    return table_data

def create_dataframe_from_table_data(table_data: List[List[str]]) -> pd.DataFrame:
    """Create and clean DataFrame from table data"""
    df = pd.DataFrame(table_data)
    df = df.replace('', pd.NA).dropna(how='all').reset_index(drop=True)
    df = df.dropna(axis=1, how='all')
    
    # Try to detect header
    if len(df) > 1:
        first_row = df.iloc[0].astype(str)
        # Check if first row looks like header (has text, not just numbers)
        has_text = any(len(str(val)) > 0 and not str(val).replace('.', '').replace('-', '').isdigit() 
                      for val in first_row)
        if has_text:
            df.columns = first_row
            df = df[1:].reset_index(drop=True)
    
    return df

# ============================================================================
# PDF EXTRACTION FUNCTIONS
# ============================================================================

def extract_with_pdfplumber(pdf_path: str, page_num: int) -> List[pd.DataFrame]:
    """Extract tables using pdfplumber"""
    if not PDFPLUMBER_AVAILABLE:
        return []
    
    page_tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num <= len(pdf.pages):
                page = pdf.pages[page_num - 1]
                
                # Try table extraction
                tables = page.extract_tables()
                for table_data in tables:
                    df = process_table_data(table_data)
                    if not df.empty:
                        page_tables.append(df)
                
                # Try text-based table detection
                if not page_tables:
                    text = page.extract_text()
                    if text:
                        df = extract_table_from_text(text)
                        if not df.empty:
                            page_tables.append(df)
    except Exception as e:
        st.warning(f"pdfplumber extraction warning on page {page_num}: {e}")
    
    return page_tables

def extract_with_camelot(pdf_path: str, page_num: int) -> List[pd.DataFrame]:
    """Extract tables using camelot"""
    if not CAMELOT_AVAILABLE:
        return []
    
    page_tables = []
    try:
        # Try lattice mode first (for tables with borders)
        tables = camelot.read_pdf(pdf_path, pages=str(page_num), flavor='lattice')
        for table in tables:
            df = table.df
            if not df.empty and len(df) > 1:
                page_tables.append(df)
        
        # If no tables found, try stream mode
        if not page_tables:
            tables = camelot.read_pdf(pdf_path, pages=str(page_num), flavor='stream')
            for table in tables:
                df = table.df
                if not df.empty and len(df) > 1:
                    page_tables.append(df)
    except Exception as e:
        st.warning(f"camelot extraction warning on page {page_num}: {e}")
    
    return page_tables

def process_table_data(table_data: List[List]) -> pd.DataFrame:
    """Process raw table data into a clean DataFrame"""
    if not table_data or len(table_data) <= 1:
        return pd.DataFrame()
    
    df = pd.DataFrame(table_data)
    df = df.replace('', pd.NA).dropna(how='all').reset_index(drop=True)
    df = df.dropna(axis=1, how='all')
    
    if not df.empty and len(df) > 0:
        # Try to set header
        if len(df) > 1:
            first_row = df.iloc[0].astype(str)
            if first_row.str.contains(r'[a-zA-Z]').any():
                df.columns = first_row
                df = df[1:].reset_index(drop=True)
        
        df.columns = [f'Column_{i+1}' if pd.isna(col) or str(col).strip() == '' else str(col).strip() 
                     for i, col in enumerate(df.columns)]
    
    return df

def extract_table_from_text(text: str) -> pd.DataFrame:
    """Extract table from structured text"""
    lines = text.split('\n')
    current_table = []
    
    for line in lines:
        parts = [part.strip() for part in re.split(r'\s{2,}', line) if part.strip()]
        if len(parts) >= 2:
            current_table.append(parts)
        elif current_table and len(current_table) >= 2:
            # Convert to DataFrame
            return create_dataframe_from_text_table(current_table)
    
    if current_table and len(current_table) >= 2:
        return create_dataframe_from_text_table(current_table)
    
    return pd.DataFrame()

def create_dataframe_from_text_table(table_data: List[List]) -> pd.DataFrame:
    """Create DataFrame from text-based table"""
    max_cols = max(len(row) for row in table_data)
    padded_table = [row + [''] * (max_cols - len(row)) for row in table_data]
    df = pd.DataFrame(padded_table)
    df = df.replace('', pd.NA).dropna(how='all').reset_index(drop=True)
    df = df.dropna(axis=1, how='all')
    
    if not df.empty and len(df) >= 2:
        return df
    
    return pd.DataFrame()

# ============================================================================
# MAIN EXTRACTION FUNCTION
# ============================================================================

def extract_any_format(file_path: str, 
                      pages: List[int], 
                      mode: str = "Auto-detect",
                      ocr_lang: str = 'eng',
                      ocr_confidence: int = 50,
                      enhance_handwriting: bool = True) -> Dict[int, List[pd.DataFrame]]:
    """
    Universal extractor that handles any document format
    Uses multiple methods and combines results
    """
    tables_by_page = {}
    file_ext = file_path.split('.')[-1].lower()
    
    for page_num in pages:
        page_tables = []
        
        # Method 1: Try pdfplumber for digital PDFs
        if mode in ["Digital", "Auto-detect"] and file_ext == 'pdf':
            pdf_tables = extract_with_pdfplumber(file_path, page_num)
            page_tables.extend(pdf_tables)
        
        # Method 2: Try camelot for complex tables
        if mode in ["Complex Tables", "Auto-detect"] and file_ext == 'pdf':
            camelot_tables = extract_with_camelot(file_path, page_num)
            page_tables.extend(camelot_tables)
        
        # Method 3: Try OCR for handwritten/scanned documents
        if mode in ["Handwritten", "Scanned", "Auto-detect"]:
            if (file_ext in ['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'] and 
                (not page_tables or mode != "Digital")):
                ocr_tables = extract_handwritten_table_with_ocr(
                    file_path, 
                    lang=ocr_lang,
                    confidence_threshold=ocr_confidence,
                    enhance=enhance_handwriting
                )
                page_tables.extend(ocr_tables)
        
        # Method 4: Try text-based extraction for any format
        if mode in ["Text", "Auto-detect"] and file_ext == 'pdf' and not page_tables:
            try:
                with pdfplumber.open(file_path) as pdf:
                    if page_num <= len(pdf.pages):
                        text = pdf.pages[page_num - 1].extract_text()
                        if text:
                            df = extract_table_from_text(text)
                            if not df.empty:
                                page_tables.append(df)
            except:
                pass
        
        # Deduplicate tables (remove very similar ones)
        unique_tables = deduplicate_tables(page_tables)
        
        if unique_tables:
            tables_by_page[page_num] = unique_tables
    
    return tables_by_page

def deduplicate_tables(tables: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Remove duplicate tables"""
    unique_tables = []
    for table in tables:
        is_duplicate = False
        for existing in unique_tables:
            if table.shape == existing.shape and table.equals(existing):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_tables.append(table)
    return unique_tables

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
                            cell.number_format = 'yyyy-mm-dd hh:mm:ss' if cell.value.time() else 'yyyy-mm-dd'
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
                        'Data Cleaning'
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
                        "Yes"
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
            type=['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            help="Upload any document containing tabular data - printed, handwritten, or scanned",
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
                        st.session_state.pdf_metadata = {
                            'total_pages': 1,
                            'file_name': uploaded_file.name,
                            'file_size': f"{uploaded_file.size / 1024:.1f} KB",
                            'file_type': 'PDF (estimated)' if file_extension == 'pdf' else 'Image',
                            'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                    
                    st.success(f"✅ Document loaded successfully")
                    
                except Exception as e:
                    st.error(f"Error reading document: {e}")
        
        st.markdown("---")
        
        with st.expander("ℹ️ About", expanded=False):
            st.markdown("""
            **Advanced PDF Table Extractor**
            
            **Features:**
            - Digital PDF table extraction
            - Handwritten text recognition
            - Scanned document processing
            - Multi-format support
            - Excel-ready output
            
            **Supported Formats:**
            - PDF (digital & scanned)
            - Images (PNG, JPG, JPEG, TIFF, BMP)
            
            **Version:** 2.0.0
            """)
        
        return uploaded_file

def render_document_info():
    """Render document information metrics"""
    st.markdown("### 📊 Document Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.pdf_metadata.get('file_type', 'PDF')}</div>
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

def render_extraction_settings() -> Tuple[str, bool, int, int, int, str, int, bool]:
    """Render extraction settings and return configuration"""
    st.header("🎯 Extraction Settings")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        extraction_mode = st.selectbox(
            "Extraction Mode",
            options=[
                "Auto-detect (Recommended)",
                "Digital PDF (Tables)",
                "Handwritten/Scanned",
                "Complex Tables",
                "Raw Text with Structure"
            ],
            index=0,
            help="Choose the extraction method that best matches your document type"
        )
        
        # Map selection to mode string
        mode_map = {
            "Auto-detect (Recommended)": "Auto-detect",
            "Digital PDF (Tables)": "Digital",
            "Handwritten/Scanned": "Handwritten",
            "Complex Tables": "Complex Tables",
            "Raw Text with Structure": "Text"
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
    
    # Advanced options
    with st.expander("⚙️ Advanced Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            exact_rows = st.number_input(
                "Exact rows per table (0 = any)", 
                0, 1000000, 0,
                help="Set to 0 for any row count, or specify exact number to extract only tables with that many rows"
            )
            
            min_cols = st.number_input(
                "Minimum columns", 
                1, 50, 2,
                help="Minimum number of columns a table must have"
            )
        
        with col2:
            if st.session_state.extraction_mode in ["Handwritten", "Auto-detect"]:
                ocr_confidence = st.slider(
                    "OCR Confidence (%)", 
                    0, 100, 50,
                    help="Higher values mean more accurate but fewer results"
                )
                
                enhance_handwriting = st.checkbox("Enhance handwriting", value=True,
                                                 help="Apply additional preprocessing for handwritten text")
                
                ocr_lang = st.selectbox(
                    "OCR Language",
                    options=['eng', 'fra', 'deu', 'spa', 'ita', 'por', 'rus', 'chi_sim', 'jpn', 'kor'],
                    index=0,
                    help="Select language for OCR"
                )
            else:
                ocr_confidence = 50
                enhance_handwriting = True
                ocr_lang = 'eng'
    
    return extraction_mode, debug_mode, exact_rows, min_cols, selected_pages, ocr_lang, ocr_confidence, enhance_handwriting

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
    
    # Table browser with column/row selection
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
        
        # Column naming options
        st.markdown("**Column Options:**")
        rename_columns = st.checkbox("Use smart column names", value=True, 
                                     help="Rename columns based on detected patterns")
        preserve_original = st.checkbox("Keep original column names", value=False,
                                       help="Keep the original column names as found in the document")
    
    with col2:
        include_metadata = st.checkbox("Include metadata sheet", value=True,
                                      help="Add a sheet with extraction information")
        auto_format = st.checkbox("Auto-format Excel", value=True,
                                 help="Auto-adjust column widths and format numbers/dates")
        
        # Data cleaning options
        st.markdown("**Data Cleaning:**")
        remove_duplicates = st.checkbox("Remove duplicate rows", value=False,
                                       help="Remove duplicate rows from each table")
        remove_empty = st.checkbox("Remove empty rows", value=True,
                                  help="Remove rows that are completely empty")
        strip_whitespace = st.checkbox("Strip whitespace", value=True,
                                      help="Remove extra spaces from text")
    
    # Export button
    if st.button("📥 Generate Excel File", type="primary", use_container_width=True):
        tables_to_export, export_summary = prepare_tables_for_export(
            remove_duplicates, remove_empty, strip_whitespace, 
            rename_columns, preserve_original
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

def prepare_tables_for_export(remove_duplicates: bool, remove_empty: bool, 
                             strip_whitespace: bool, rename_columns: bool,
                             preserve_original: bool) -> Tuple[List[Dict], List[Dict]]:
    """Prepare tables for export with filtering and cleaning"""
    tables_to_export = []
    export_summary = []
    
    for table_id, table_info in st.session_state.selected_tables.items():
        if table_info.get("selected", False):
            df = table_info.get("df")
            if df is not None:
                filtered_df = filter_and_clean_table(
                    df, table_id, remove_duplicates, remove_empty, 
                    strip_whitespace, rename_columns, preserve_original
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
                          remove_duplicates: bool, remove_empty: bool,
                          strip_whitespace: bool, rename_columns: bool,
                          preserve_original: bool) -> pd.DataFrame:
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
    
    if remove_duplicates:
        filtered_df = filtered_df.drop_duplicates()
    
    # Rename columns if requested
    if rename_columns and not filtered_df.empty:
        filtered_df = rename_columns_smartly(filtered_df)
    elif not preserve_original:
        filtered_df = clean_column_names(filtered_df)
    
    # Convert to proper data types for Excel
    filtered_df = convert_to_proper_types(filtered_df)
    
    return filtered_df

def rename_columns_smartly(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns based on detected patterns"""
    column_analysis = analyze_columns_for_patterns(df)
    new_columns = []
    
    for col in df.columns:
        if column_analysis.get(col) and column_analysis[col]:
            # Use first detected pattern
            new_name = column_analysis[col][0].title()
            # Add index if duplicate
            if new_name in new_columns:
                counter = 2
                while f"{new_name}_{counter}" in new_columns:
                    counter += 1
                new_name = f"{new_name}_{counter}"
            new_columns.append(new_name)
        else:
            # Keep original but clean it
            new_name = str(col).strip()
            if new_name in new_columns:
                counter = 2
                while f"{new_name}_{counter}" in new_columns:
                    counter += 1
                new_name = f"{new_name}_{counter}"
            new_columns.append(new_name)
    
    df.columns = new_columns
    return df

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names by removing special characters"""
    new_columns = []
    for col in df.columns:
        # Remove special characters and clean
        clean_col = re.sub(r'[^\w\s]', '', str(col))
        clean_col = clean_col.strip().replace(' ', '_')
        if not clean_col:
            clean_col = f"Column_{len(new_columns)+1}"
        
        # Handle duplicates
        if clean_col in new_columns:
            counter = 2
            while f"{clean_col}_{counter}" in new_columns:
                counter += 1
            clean_col = f"{clean_col}_{counter}"
        new_columns.append(clean_col)
    
    df.columns = new_columns
    return df

def render_welcome_screen():
    """Render welcome screen when no document is uploaded"""
    st.markdown("""
    <div class="info-box">
    <h1>📊 Advanced Document Table Extractor</h1>
    <p style='font-size: 18px;'>Extract tabular data from <strong>ANY document type</strong> - printed, handwritten, scanned, or digital PDFs.</p>
    <p><strong>Supported Formats:</strong> PDF, PNG, JPG, JPEG, TIFF, BMP</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature showcase
    st.markdown("### ✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Multi-Format Support</div>
            <div class="feature-description">Extract from PDFs, images, scanned documents, and handwritten notes</div>
        </div>
        
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">Smart Column Detection</div>
            <div class="feature-description">Automatically identifies debit, credit, date, amount, and other column types</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">✍️</div>
            <div class="feature-title">Handwriting Recognition</div>
            <div class="feature-description">Advanced OCR with preprocessing for handwritten documents</div>
        </div>
        
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Excel-Ready Output</div>
            <div class="feature-description">Numbers and dates properly formatted for Excel functions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Precise Extraction</div>
            <div class="feature-description">Extract exactly 16 rows or any specific number you need</div>
        </div>
        
        <div class="feature-card">
            <div class="feature-icon">🔄</div>
            <div class="feature-title">Multiple Export Modes</div>
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
            <div class="step-text">Choose Mode</div>
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
        - **For digital PDFs**: Use "Digital PDF" mode for fastest extraction
        - **For handwritten documents**: Enable "Enhance handwriting" and adjust OCR confidence
        - **For specific row counts**: Set "Exact rows per table" to your desired number
        - **For better accuracy**: Use high-quality scans (300 DPI or higher)
        - **For complex tables**: Try "Complex Tables" mode with camelot
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
                <strong>Advanced Document Table Extractor</strong> • Version 2.0.0<br>
                Built with Streamlit, OpenCV, Tesseract OCR, and pdfplumber<br>
                © 2024 - Extract tables from any document
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
    
    # Main content
    if st.session_state.pdf_uploaded and uploaded_file:
        render_document_info()
        
        # Get extraction settings
        extraction_mode, debug_mode, exact_rows, min_cols, selected_pages, ocr_lang, ocr_confidence, enhance_handwriting = render_extraction_settings()
        
        # Extract button
        if selected_pages and st.button("🔍 Extract Data", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner(f"Extracting data from {len(selected_pages)} pages..."):
                # Save file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp:
                    tmp.write(uploaded_file.getvalue())
                    file_path = tmp.name
                
                try:
                    # Update progress
                    status_text.text("Processing document...")
                    progress_bar.progress(20)
                    
                    # Extract tables using universal method
                    tables_by_page = extract_any_format(
                        file_path, 
                        selected_pages, 
                        st.session_state.extraction_mode,
                        ocr_lang=ocr_lang,
                        ocr_confidence=ocr_confidence,
                        enhance_handwriting=enhance_handwriting
                    )
                    
                    progress_bar.progress(60)
                    status_text.text("Analyzing extracted data...")
                    
                    # Filter tables based on criteria
                    filtered_tables, total_tables_found, tables_ignored = filter_tables(
                        tables_by_page, exact_rows, min_cols
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
                    display_extraction_results(total_tables_found, filtered_tables, tables_ignored, exact_rows)
                    
                    # Show debug info if enabled
                    if debug_mode:
                        with st.expander("🔧 Debug Information"):
                            st.json({
                                'extraction_mode': st.session_state.extraction_mode,
                                'pages_scanned': len(selected_pages),
                                'tables_found': total_tables_found,
                                'tables_ignored': tables_ignored,
                                'pages_with_tables': list(filtered_tables.keys()),
                                'ocr_settings': {
                                    'language': ocr_lang,
                                    'confidence': ocr_confidence,
                                    'enhance_handwriting': enhance_handwriting
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

def filter_tables(tables_by_page: Dict, exact_rows: int, min_cols: int) -> Tuple[Dict, int, int]:
    """Filter tables based on criteria"""
    filtered_tables = {}
    total_tables_found = 0
    tables_ignored = 0
    
    for page_num, tables in tables_by_page.items():
        filtered_page_tables = []
        for table in tables:
            # Apply row filtering
            row_match = True
            if exact_rows > 0:
                row_match = (len(table) == exact_rows)
            
            if row_match and len(table.columns) >= min_cols:
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

def display_extraction_results(total_tables_found: int, filtered_tables: Dict, tables_ignored: int, exact_rows: int):
    """Display extraction results"""
    if total_tables_found > 0:
        if exact_rows > 0:
            st.markdown(f"""
            <div class="success-box">
            <h3>✅ Extraction Complete!</h3>
            <p>Found <strong>{total_tables_found}</strong> tables with exactly <strong>{exact_rows}</strong> rows</p>
            <p>Pages with tables: <strong>{len(filtered_tables)}</strong></p>
            <p>Tables ignored: <strong>{tables_ignored}</strong> (didn't meet criteria)</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="success-box">
            <h3>✅ Extraction Complete!</h3>
            <p>Found <strong>{total_tables_found}</strong> tables across <strong>{len(filtered_tables)}</strong> pages</p>
            <p>Total rows: <strong>{st.session_state.extraction_stats['total_rows']:,}</strong></p>
            <p>Average table size: <strong>{st.session_state.extraction_stats['avg_rows_per_table']:.0f}</strong> rows × <strong>{st.session_state.extraction_stats['avg_columns_per_table']:.0f}</strong> columns</p>
            </div>
            """, unsafe_allow_html=True)
        
        if tables_ignored > 0:
            st.info(f"ℹ️ Ignored {tables_ignored} tables that didn't meet criteria")
    else:
        if exact_rows > 0:
            st.warning(f"⚠️ No tables found with exactly {exact_rows} rows. Try a different row count or extraction mode.")
        else:
            st.warning(f"⚠️ No tables found. Try a different extraction mode or adjust settings.")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
