"""
Advanced PDF Table & Data Extractor
Supports: Digital PDFs, Scanned Documents, Handwritten Text, Images
Extracts ANY tabular data (rows & columns) from any document type
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
from sklearn.cluster import DBSCAN
from scipy import stats

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

# ML imports for pattern detection
try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.warning("scikit-learn not available. Install for enhanced pattern detection.")

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Advanced Document to Excel Converter",
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
        'extraction_mode': "Auto-detect (Any Rows/Columns)",
        'processing_history': [],
        'current_file_hash': None,
        'extraction_stats': {},
        'ocr_language': 'eng',
        'debug_mode': False,
        'detected_regions': [],
        'pattern_confidence': {}
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
# ADVANCED ROW/COLUMN DETECTION FUNCTIONS
# ============================================================================

def detect_any_row_column_pattern(image: np.ndarray) -> Dict[str, Any]:
    """
    Detect ANY pattern of rows and columns in an image using multiple methods
    Returns detected regions and their confidence scores
    """
    if not CV_AVAILABLE:
        return {'regions': [], 'confidence': 0}
    
    detected_regions = []
    methods_used = []
    
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Method 1: Line detection for grids/tables
        regions1, conf1 = detect_grid_pattern(gray)
        if regions1:
            detected_regions.extend(regions1)
            methods_used.append(('grid', conf1))
        
        # Method 2: Text block alignment detection
        regions2, conf2 = detect_aligned_text_blocks(gray)
        if regions2:
            detected_regions.extend(regions2)
            methods_used.append(('text_blocks', conf2))
        
        # Method 3: Vertical/Horizontal line clustering
        regions3, conf3 = detect_line_clusters(gray)
        if regions3:
            detected_regions.extend(regions3)
            methods_used.append(('line_clusters', conf3))
        
        # Method 4: Content density analysis
        regions4, conf4 = detect_content_density_pattern(gray)
        if regions4:
            detected_regions.extend(regions4)
            methods_used.append(('density', conf4))
        
        # Method 5: Contour-based region detection
        regions5, conf5 = detect_contour_regions(gray)
        if regions5:
            detected_regions.extend(regions5)
            methods_used.append(('contours', conf5))
        
        # Merge overlapping regions
        merged_regions = merge_overlapping_regions(detected_regions)
        
        # Calculate overall confidence
        overall_confidence = calculate_overall_confidence(methods_used, merged_regions)
        
        return {
            'regions': merged_regions,
            'confidence': overall_confidence,
            'methods_used': methods_used,
            'region_count': len(merged_regions)
        }
        
    except Exception as e:
        st.warning(f"Pattern detection error: {e}")
        return {'regions': [], 'confidence': 0, 'methods_used': [], 'region_count': 0}

def detect_grid_pattern(gray: np.ndarray) -> Tuple[List[Dict], float]:
    """
    Detect grid-like patterns (traditional tables)
    """
    regions = []
    
    try:
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
        grid = cv2.add(horizontal_lines, vertical_lines)
        
        # Find grid intersections
        kernel = np.ones((3,3), np.uint8)
        grid = cv2.dilate(grid, kernel, iterations=1)
        
        # Find contours of grid cells
        contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 4:  # At least 4 cells to be considered a grid
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 20 and h > 20:  # Filter out very small cells
                    regions.append({
                        'x': x, 'y': y, 'width': w, 'height': h,
                        'type': 'grid_cell',
                        'confidence': min(1.0, len(contours) / 50)  # More cells = higher confidence
                    })
        
        # Calculate confidence based on grid structure
        confidence = min(1.0, len(regions) / 30) if regions else 0
        
        return regions, confidence
        
    except Exception as e:
        return [], 0

def detect_aligned_text_blocks(gray: np.ndarray) -> Tuple[List[Dict], float]:
    """
    Detect aligned text blocks (forms, lists, structured text)
    """
    regions = []
    
    try:
        # Perform OCR to get text blocks
        custom_config = r'--psm 6 --oem 3'
        ocr_data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)
        
        text_blocks = []
        for i in range(len(ocr_data['text'])):
            if int(ocr_data['conf'][i]) > 30:  # Only consider decent confidence text
                text = ocr_data['text'][i].strip()
                if text:
                    text_blocks.append({
                        'text': text,
                        'x': ocr_data['left'][i],
                        'y': ocr_data['top'][i],
                        'width': ocr_data['width'][i],
                        'height': ocr_data['height'][i],
                        'line': ocr_data['line_num'][i],
                        'block': ocr_data['block_num'][i]
                    })
        
        if len(text_blocks) < 5:
            return [], 0
        
        # Group by vertical position (rows)
        y_positions = [block['y'] for block in text_blocks]
        if ML_AVAILABLE:
            # Use DBSCAN to cluster rows
            y_scaled = StandardScaler().fit_transform(np.array(y_positions).reshape(-1, 1))
            row_clusters = DBSCAN(eps=0.5, min_samples=2).fit(y_scaled).labels_
        else:
            # Simple threshold-based clustering
            y_sorted = sorted(y_positions)
            row_clusters = []
            current_cluster = 0
            for i, y in enumerate(y_sorted):
                if i == 0:
                    row_clusters.append(0)
                else:
                    if y - y_sorted[i-1] > 20:  # New row if gap > 20 pixels
                        current_cluster += 1
                    row_clusters.append(current_cluster)
        
        # Group by horizontal position (columns) within each row
        rows_data = defaultdict(list)
        for block, cluster in zip(text_blocks, row_clusters):
            rows_data[cluster].append(block)
        
        # Detect consistent column patterns
        column_positions = []
        for row_num, blocks in rows_data.items():
            if len(blocks) > 1:  # Row has multiple items
                x_positions = sorted([b['x'] for b in blocks])
                column_positions.extend(x_positions)
        
        if column_positions:
            # Cluster column positions
            if ML_AVAILABLE:
                col_scaled = StandardScaler().fit_transform(np.array(column_positions).reshape(-1, 1))
                col_clusters = DBSCAN(eps=0.5, min_samples=2).fit(col_scaled).labels_
                unique_cols = len(set(col_clusters)) - (1 if -1 in col_clusters else 0)
            else:
                # Simple clustering by gaps
                col_sorted = sorted(set(column_positions))
                unique_cols = 1
                for i in range(1, len(col_sorted)):
                    if col_sorted[i] - col_sorted[i-1] > 50:  # Gap > 50 pixels = new column
                        unique_cols += 1
            
            if unique_cols >= 2:  # At least 2 columns detected
                # Create region for entire structured area
                all_x = [b['x'] for b in text_blocks]
                all_y = [b['y'] for b in text_blocks]
                all_widths = [b['width'] for b in text_blocks]
                all_heights = [b['height'] for b in text_blocks]
                
                if all_x and all_y:
                    x_min = min(all_x)
                    y_min = min(all_y)
                    x_max = max([x + w for x, w in zip(all_x, all_widths)])
                    y_max = max([y + h for y, h in zip(all_y, all_heights)])
                    
                    regions.append({
                        'x': x_min,
                        'y': y_min,
                        'width': x_max - x_min,
                        'height': y_max - y_min,
                        'type': 'structured_text',
                        'columns_detected': unique_cols,
                        'rows_detected': len(rows_data),
                        'confidence': min(1.0, (unique_cols * len(rows_data)) / 30)
                    })
        
        confidence = max([r.get('confidence', 0) for r in regions]) if regions else 0
        
        return regions, confidence
        
    except Exception as e:
        return [], 0

def detect_line_clusters(gray: np.ndarray) -> Tuple[List[Dict], float]:
    """
    Detect patterns formed by clustered lines (like forms with lines)
    """
    regions = []
    
    try:
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                minLineLength=50, maxLineGap=10)
        
        if lines is None or len(lines) < 5:
            return [], 0
        
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 20:  # Near horizontal
                horizontal_lines.append((y1, abs(x2 - x1)))
            elif angle > 70:  # Near vertical
                vertical_lines.append((x1, abs(y2 - y1)))
        
        # Cluster horizontal lines to find rows
        if horizontal_lines:
            y_positions = [h[0] for h in horizontal_lines]
            if ML_AVAILABLE:
                y_scaled = StandardScaler().fit_transform(np.array(y_positions).reshape(-1, 1))
                row_clusters = DBSCAN(eps=0.3, min_samples=1).fit(y_scaled).labels_
                row_count = len(set(row_clusters))
            else:
                # Simple clustering
                y_sorted = sorted(set(y_positions))
                row_count = 1
                for i in range(1, len(y_sorted)):
                    if y_sorted[i] - y_sorted[i-1] > 30:
                        row_count += 1
        else:
            row_count = 0
        
        # Cluster vertical lines to find columns
        if vertical_lines:
            x_positions = [v[0] for v in vertical_lines]
            if ML_AVAILABLE:
                x_scaled = StandardScaler().fit_transform(np.array(x_positions).reshape(-1, 1))
                col_clusters = DBSCAN(eps=0.3, min_samples=1).fit(x_scaled).labels_
                col_count = len(set(col_clusters))
            else:
                x_sorted = sorted(set(x_positions))
                col_count = 1
                for i in range(1, len(x_sorted)):
                    if x_sorted[i] - x_sorted[i-1] > 30:
                        col_count += 1
        else:
            col_count = 0
        
        if row_count >= 2 and col_count >= 2:
            # Create region covering the line intersections
            all_x = [v[0] for v in vertical_lines] if vertical_lines else []
            all_y = [h[0] for h in horizontal_lines] if horizontal_lines else []
            
            if all_x and all_y:
                regions.append({
                    'x': min(all_x) - 20,
                    'y': min(all_y) - 20,
                    'width': max(all_x) - min(all_x) + 40,
                    'height': max(all_y) - min(all_y) + 40,
                    'type': 'line_grid',
                    'rows': row_count,
                    'columns': col_count,
                    'confidence': min(1.0, (row_count * col_count) / 50)
                })
        
        confidence = max([r.get('confidence', 0) for r in regions]) if regions else 0
        
        return regions, confidence
        
    except Exception as e:
        return [], 0

def detect_content_density_pattern(gray: np.ndarray) -> Tuple[List[Dict], float]:
    """
    Detect patterns based on content density (where text/objects are clustered)
    """
    regions = []
    
    try:
        # Threshold to get foreground
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Create density map
        density_map = np.zeros_like(gray, dtype=np.float32)
        kernel_size = 20
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        
        # Calculate local density
        density = cv2.filter2D(binary.astype(np.float32), -1, kernel)
        
        # Threshold density to find high-density regions
        density_threshold = 0.3  # 30% density threshold
        high_density = (density > density_threshold).astype(np.uint8) * 255
        
        # Find contours of high-density regions
        contours, _ = cv2.findContours(high_density, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 100 and h > 50:  # Minimum region size
                # Analyze internal structure
                roi = binary[y:y+h, x:x+w]
                
                # Count vertical and horizontal transitions (potential columns/rows)
                vertical_projection = np.sum(roi, axis=1) / 255
                horizontal_projection = np.sum(roi, axis=0) / 255
                
                # Detect peaks in projections (indicating rows/columns)
                from scipy.signal import find_peaks
                
                vertical_peaks, _ = find_peaks(vertical_projection, distance=10, height=5)
                horizontal_peaks, _ = find_peaks(horizontal_projection, distance=10, height=5)
                
                row_count = len(vertical_peaks)
                col_count = len(horizontal_peaks)
                
                if row_count >= 2 and col_count >= 2:
                    regions.append({
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'type': 'density_pattern',
                        'rows': row_count,
                        'columns': col_count,
                        'confidence': min(1.0, (row_count * col_count) / 40)
                    })
        
        confidence = max([r.get('confidence', 0) for r in regions]) if regions else 0
        
        return regions, confidence
        
    except Exception as e:
        return [], 0

def detect_contour_regions(gray: np.ndarray) -> Tuple[List[Dict], float]:
    """
    Detect regions based on contours (separate blocks that might form a table)
    """
    regions = []
    
    try:
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area
                x, y, w, h = cv2.boundingRect(contour)
                if w > 30 and h > 30:
                    valid_contours.append((x, y, w, h))
        
        if len(valid_contours) < 4:
            return [], 0
        
        # Sort by position
        valid_contours.sort(key=lambda x: (x[1], x[0]))
        
        # Group into rows
        rows = []
        current_row = []
        current_y = valid_contours[0][1]
        
        for contour in valid_contours:
            x, y, w, h = contour
            if abs(y - current_y) < 30:  # Same row
                current_row.append(contour)
            else:
                if current_row:
                    rows.append(current_row)
                current_row = [contour]
                current_y = y
        
        if current_row:
            rows.append(current_row)
        
        # Check if we have a grid-like structure
        if len(rows) >= 2:
            # Sort each row by x position
            for row in rows:
                row.sort(key=lambda x: x[0])
            
            # Check column consistency across rows
            col_counts = [len(row) for row in rows]
            if max(col_counts) >= 2 and stats.mode(col_counts)[0][0] >= 2:
                # Create region for entire structure
                all_x = [c[0] for row in rows for c in row]
                all_y = [c[1] for row in rows for c in row]
                all_x2 = [c[0] + c[2] for row in rows for c in row]
                all_y2 = [c[1] + c[3] for row in rows for c in row]
                
                regions.append({
                    'x': min(all_x) - 10,
                    'y': min(all_y) - 10,
                    'width': max(all_x2) - min(all_x) + 20,
                    'height': max(all_y2) - min(all_y) + 20,
                    'type': 'contour_grid',
                    'rows': len(rows),
                    'columns': stats.mode(col_counts)[0][0],
                    'confidence': min(1.0, len(valid_contours) / 50)
                })
        
        confidence = max([r.get('confidence', 0) for r in regions]) if regions else 0
        
        return regions, confidence
        
    except Exception as e:
        return [], 0

def merge_overlapping_regions(regions: List[Dict]) -> List[Dict]:
    """
    Merge overlapping regions to avoid duplicates
    """
    if not regions:
        return []
    
    # Sort by area (largest first)
    regions.sort(key=lambda x: x['width'] * x['height'], reverse=True)
    
    merged = []
    used = set()
    
    for i, region in enumerate(regions):
        if i in used:
            continue
        
        current = region.copy()
        used.add(i)
        
        # Check for overlapping regions
        for j, other in enumerate(regions[i+1:], i+1):
            if j in used:
                continue
            
            # Calculate overlap
            x_overlap = max(0, min(current['x'] + current['width'], 
                                  other['x'] + other['width']) - 
                          max(current['x'], other['x']))
            y_overlap = max(0, min(current['y'] + current['height'],
                                  other['y'] + other['height']) -
                          max(current['y'], other['y']))
            
            overlap_area = x_overlap * y_overlap
            current_area = current['width'] * current['height']
            
            if overlap_area > 0.3 * current_area:  # Significant overlap
                # Merge regions
                current['x'] = min(current['x'], other['x'])
                current['y'] = min(current['y'], other['y'])
                current['width'] = max(current['x'] + current['width'],
                                     other['x'] + other['width']) - current['x']
                current['height'] = max(current['y'] + current['height'],
                                      other['y'] + other['height']) - current['y']
                current['confidence'] = max(current['confidence'], other.get('confidence', 0))
                used.add(j)
        
        merged.append(current)
    
    return merged

def calculate_overall_confidence(methods_used: List[Tuple[str, float]], 
                                 regions: List[Dict]) -> float:
    """
    Calculate overall confidence in detected patterns
    """
    if not methods_used:
        return 0
    
    # Weight each method's confidence
    method_weights = {
        'grid': 1.2,        # Grid patterns are most reliable
        'text_blocks': 1.0,  # Text alignment is reliable
        'line_clusters': 0.9,
        'density': 0.7,
        'contours': 0.6
    }
    
    weighted_sum = sum(conf * method_weights.get(method, 0.5) 
                      for method, conf in methods_used)
    max_possible = sum(method_weights.get(method, 0.5) for method, _ in methods_used)
    
    base_confidence = weighted_sum / max_possible if max_possible > 0 else 0
    
    # Adjust based on number of regions
    region_factor = min(1.0, len(regions) / 20) if regions else 0
    
    # Adjust based on region consistency
    if regions:
        confidences = [r.get('confidence', 0) for r in regions]
        consistency_factor = 1 - np.std(confidences) if confidences else 0
    else:
        consistency_factor = 0
    
    overall = (base_confidence * 0.5 + region_factor * 0.3 + consistency_factor * 0.2)
    
    return min(1.0, overall)

# ============================================================================
# ENHANCED OCR EXTRACTION FUNCTIONS
# ============================================================================

def extract_from_any_pattern(image: np.ndarray, 
                           detection_result: Dict,
                           lang: str = 'eng',
                           min_confidence: float = 0.3) -> List[pd.DataFrame]:
    """
    Extract data from detected patterns using appropriate method
    """
    tables_found = []
    
    if not detection_result['regions']:
        return tables_found
    
    for region in detection_result['regions']:
        if region.get('confidence', 0) < min_confidence:
            continue
        
        # Extract region of interest
        x, y, w, h = region['x'], region['y'], region['width'], region['height']
        
        # Add padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2*padding)
        h = min(image.shape[0] - y, h + 2*padding)
        
        roi = image[y:y+h, x:x+w]
        
        # Extract based on region type
        if region.get('type') in ['grid_cell', 'line_grid']:
            # Structured grid extraction
            df = extract_structured_grid(roi, lang)
        elif region.get('type') == 'structured_text':
            # Text alignment based extraction
            df = extract_aligned_text(roi, lang, 
                                     region.get('columns_detected', 2))
        elif region.get('type') in ['density_pattern', 'contour_grid']:
            # Generic pattern extraction
            df = extract_generic_pattern(roi, lang)
        else:
            # Default extraction
            df = extract_generic_pattern(roi, lang)
        
        if df is not None and not df.empty and len(df) >= 2 and len(df.columns) >= 2:
            tables_found.append(df)
    
    return tables_found

def extract_structured_grid(roi: np.ndarray, lang: str) -> Optional[pd.DataFrame]:
    """
    Extract from grid-structured regions
    """
    try:
        # Detect grid lines
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # Apply threshold
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Find line positions
        h_projection = np.sum(horizontal_lines, axis=1) / 255
        v_projection = np.sum(vertical_lines, axis=0) / 255
        
        # Find row boundaries
        row_boundaries = []
        in_row = False
        row_start = 0
        
        for i, val in enumerate(h_projection):
            if val > 10 and not in_row:
                in_row = True
                row_start = i
            elif val <= 10 and in_row:
                if i - row_start > 10:
                    row_boundaries.append((row_start, i))
                in_row = False
        
        # Find column boundaries
        col_boundaries = []
        in_col = False
        col_start = 0
        
        for i, val in enumerate(v_projection):
            if val > 10 and not in_col:
                in_col = True
                col_start = i
            elif val <= 10 and in_col:
                if i - col_start > 10:
                    col_boundaries.append((col_start, i))
                in_col = False
        
        # Extract cells
        table_data = []
        for row_start, row_end in row_boundaries:
            row_data = []
            for col_start, col_end in col_boundaries:
                # Extract cell
                cell = gray[row_start:row_end, col_start:col_end]
                if cell.size > 0:
                    # OCR on cell
                    text = pytesseract.image_to_string(cell, config=f'--psm 7 -l {lang}').strip()
                    row_data.append(text)
                else:
                    row_data.append('')
            if any(row_data):
                table_data.append(row_data)
        
        if len(table_data) >= 2:
            df = pd.DataFrame(table_data)
            df = df.replace('', pd.NA).dropna(how='all')
            
            # Try to set header
            if len(df) > 1:
                first_row = df.iloc[0].astype(str)
                if first_row.str.contains(r'[a-zA-Z]').any():
                    df.columns = first_row
                    df = df[1:].reset_index(drop=True)
            
            return df
        
        return None
        
    except Exception as e:
        return None

def extract_aligned_text(roi: np.ndarray, lang: str, expected_cols: int) -> Optional[pd.DataFrame]:
    """
    Extract from aligned text regions
    """
    try:
        # Get OCR data with position
        custom_config = f'--psm 6 --oem 3 -l {lang}'
        ocr_data = pytesseract.image_to_data(roi, config=custom_config, 
                                            output_type=pytesseract.Output.DICT)
        
        # Group by line
        lines = {}
        for i in range(len(ocr_data['text'])):
            if int(ocr_data['conf'][i]) > 30:
                text = ocr_data['text'][i].strip()
                if text:
                    line_num = ocr_data['line_num'][i]
                    if line_num not in lines:
                        lines[line_num] = []
                    lines[line_num].append({
                        'text': text,
                        'x': ocr_data['left'][i],
                        'width': ocr_data['width'][i],
                        'conf': ocr_data['conf'][i]
                    })
        
        # Sort lines by y position
        sorted_lines = sorted(lines.items(), key=lambda x: x[1][0]['x'] if x[1] else 0)
        
        # Detect column boundaries
        all_x = []
        for _, words in sorted_lines:
            for word in words:
                all_x.append(word['x'])
        
        if not all_x:
            return None
        
        # Cluster x positions to find columns
        if ML_AVAILABLE and len(all_x) > 10:
            x_scaled = StandardScaler().fit_transform(np.array(all_x).reshape(-1, 1))
            col_clusters = DBSCAN(eps=0.5, min_samples=2).fit(x_scaled).labels_
            unique_cols = len(set(col_clusters)) - (1 if -1 in col_clusters else 0)
            
            # Calculate column boundaries
            col_boundaries = []
            for cluster_id in range(unique_cols):
                cluster_x = [x for x, c in zip(all_x, col_clusters) if c == cluster_id]
                if cluster_x:
                    col_boundaries.append((min(cluster_x), max(cluster_x)))
            
            col_boundaries.sort(key=lambda x: x[0])
        else:
            # Simple threshold-based column detection
            all_x_sorted = sorted(set(all_x))
            col_boundaries = []
            current_start = all_x_sorted[0]
            
            for i in range(1, len(all_x_sorted)):
                if all_x_sorted[i] - all_x_sorted[i-1] > 50:
                    col_boundaries.append((current_start, all_x_sorted[i-1]))
                    current_start = all_x_sorted[i]
            
            col_boundaries.append((current_start, all_x_sorted[-1]))
        
        # Build table
        table_data = []
        for line_num, words in sorted_lines:
            # Sort words in line by x position
            words.sort(key=lambda x: x['x'])
            
            # Create row with all columns
            row = [''] * len(col_boundaries)
            
            for word in words:
                # Find which column this word belongs to
                for i, (col_start, col_end) in enumerate(col_boundaries):
                    if word['x'] >= col_start and word['x'] <= col_end:
                        if row[i]:
                            row[i] += ' ' + word['text']
                        else:
                            row[i] = word['text']
                        break
            
            if any(row):
                table_data.append(row)
        
        if len(table_data) >= 2:
            df = pd.DataFrame(table_data)
            df = df.replace('', pd.NA).dropna(how='all')
            
            # Try to detect header
            if len(df) > 1:
                first_row = df.iloc[0].astype(str)
                if first_row.str.contains(r'[a-zA-Z]').any():
                    df.columns = first_row
                    df = df[1:].reset_index(drop=True)
            
            return df
        
        return None
        
    except Exception as e:
        return None

def extract_generic_pattern(roi: np.ndarray, lang: str) -> Optional[pd.DataFrame]:
    """
    Generic extraction for any pattern that might be tabular
    """
    try:
        # Perform OCR with position data
        custom_config = f'--psm 6 --oem 3 -l {lang}'
        ocr_data = pytesseract.image_to_data(roi, config=custom_config,
                                            output_type=pytesseract.Output.DICT)
        
        # Group text by lines
        lines = {}
        for i in range(len(ocr_data['text'])):
            if int(ocr_data['conf'][i]) > 30:
                text = ocr_data['text'][i].strip()
                if text:
                    line_num = ocr_data['line_num'][i]
                    if line_num not in lines:
                        lines[line_num] = []
                    lines[line_num].append({
                        'text': text,
                        'x': ocr_data['left'][i],
                        'y': ocr_data['top'][i],
                        'width': ocr_data['width'][i]
                    })
        
        if len(lines) < 2:
            return None
        
        # Detect potential column separators
        all_x = []
        for line_words in lines.values():
            for word in line_words:
                all_x.append(word['x'])
        
        # Find gaps in x positions (potential column boundaries)
        if len(all_x) > 5:
            all_x_sorted = sorted(all_x)
            gaps = []
            for i in range(1, len(all_x_sorted)):
                gap = all_x_sorted[i] - all_x_sorted[i-1]
                if gap > 20:
                    gaps.append((all_x_sorted[i-1], all_x_sorted[i], gap))
            
            # Sort gaps by size and take the largest ones as column boundaries
            gaps.sort(key=lambda x: x[2], reverse=True)
            
            # Determine number of columns based on gaps
            if gaps:
                # Use top N gaps where N is the number of columns - 1
                potential_cols = len(gaps) + 1
                if potential_cols >= 2 and potential_cols <= 10:
                    # Build table with detected columns
                    table_data = []
                    for line_num in sorted(lines.keys()):
                        words = lines[line_num]
                        words.sort(key=lambda x: x['x'])
                        
                        row = [''] * potential_cols
                        for word in words:
                            # Find column based on x position
                            col_idx = 0
                            for i, (gap_start, gap_end, _) in enumerate(gaps):
                                if word['x'] > gap_end:
                                    col_idx = i + 1
                                else:
                                    break
                            
                            if row[col_idx]:
                                row[col_idx] += ' ' + word['text']
                            else:
                                row[col_idx] = word['text']
                        
                        if any(row):
                            table_data.append(row)
                    
                    if len(table_data) >= 2:
                        df = pd.DataFrame(table_data)
                        df = df.replace('', pd.NA).dropna(how='all')
                        
                        # Try to detect header
                        if len(df) > 1:
                            first_row = df.iloc[0].astype(str)
                            if first_row.str.contains(r'[a-zA-Z]').any():
                                df.columns = first_row
                                df = df[1:].reset_index(drop=True)
                        
                        return df
        
        # If no clear columns detected, try simpler approach - split by multiple spaces
        full_text = pytesseract.image_to_string(roi, config=f'--psm 6 -l {lang}')
        lines = full_text.split('\n')
        
        table_data = []
        for line in lines:
            # Split by multiple spaces
            parts = [part.strip() for part in re.split(r'\s{2,}', line) if part.strip()]
            if len(parts) >= 2:
                table_data.append(parts)
        
        if len(table_data) >= 2:
            df = pd.DataFrame(table_data)
            return df
        
        return None
        
    except Exception as e:
        return None

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

# ============================================================================
# ENHANCED MAIN EXTRACTION FUNCTION
# ============================================================================

def extract_any_pattern_from_document(file_path: str,
                                     pages: List[int],
                                     mode: str = "Auto-detect (Any Rows/Columns)",
                                     ocr_lang: str = 'eng',
                                     min_confidence: float = 0.3,
                                     enhance: bool = True) -> Dict[int, List[pd.DataFrame]]:
    """
    Universal extractor that detects ANY row/column pattern in documents
    """
    tables_by_page = {}
    file_ext = file_path.split('.')[-1].lower()
    
    for page_num in pages:
        page_tables = []
        
        # Get page image
        try:
            if file_ext == 'pdf':
                # Convert PDF page to image
                images = pdf2image.convert_from_path(file_path, first_page=page_num, 
                                                    last_page=page_num, dpi=300)
                if images:
                    image = np.array(images[0])
                    if len(image.shape) == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                # Load image directly
                image = cv2.imread(file_path)
                if image is None:
                    # Try with PIL
                    pil_image = Image.open(file_path)
                    image = np.array(pil_image)
                    if len(image.shape) == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except Exception as e:
            st.warning(f"Could not load page {page_num}: {e}")
            continue
        
        if image is None:
            continue
        
        # Preprocess image
        if enhance:
            processed = preprocess_image_for_ocr(image)
        else:
            processed = image
        
        # Method 1: Pattern-based detection (for any row/column structure)
        if mode in ["Auto-detect (Any Rows/Columns)", "Pattern Detection", "All Methods"]:
            detection_result = detect_any_row_column_pattern(processed)
            
            if detection_result['regions']:
                pattern_tables = extract_from_any_pattern(processed, detection_result,
                                                        lang=ocr_lang,
                                                        min_confidence=min_confidence)
                page_tables.extend(pattern_tables)
                
                # Store detection info for debugging
                if 'detected_regions' not in st.session_state:
                    st.session_state.detected_regions = {}
                st.session_state.detected_regions[page_num] = detection_result
        
        # Method 2: Traditional table extraction (for structured tables)
        if mode in ["Traditional Tables", "All Methods"] and file_ext == 'pdf':
            if PDFPLUMBER_AVAILABLE:
                pdf_tables = extract_with_pdfplumber(file_path, page_num)
                page_tables.extend(pdf_tables)
            
            if CAMELOT_AVAILABLE:
                camelot_tables = extract_with_camelot(file_path, page_num)
                page_tables.extend(camelot_tables)
        
        # Method 3: Simple text-based extraction (as fallback)
        if mode in ["Text-based", "All Methods"]:
            try:
                if file_ext == 'pdf' and PDFPLUMBER_AVAILABLE:
                    with pdfplumber.open(file_path) as pdf:
                        if page_num <= len(pdf.pages):
                            text = pdf.pages[page_num - 1].extract_text()
                            if text:
                                df = extract_table_from_text(text)
                                if df is not None and not df.empty:
                                    page_tables.append(df)
                else:
                    # OCR the whole image
                    text = pytesseract.image_to_string(processed, config=f'--psm 6 -l {ocr_lang}')
                    df = extract_table_from_text(text)
                    if df is not None and not df.empty:
                        page_tables.append(df)
            except:
                pass
        
        # Deduplicate tables
        unique_tables = deduplicate_tables(page_tables)
        
        if unique_tables:
            tables_by_page[page_num] = unique_tables
    
    return tables_by_page

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
    table_data = []
    
    for line in lines:
        # Try multiple space separation
        parts = [part.strip() for part in re.split(r'\s{2,}', line) if part.strip()]
        if len(parts) >= 2:
            table_data.append(parts)
        else:
            # Try common delimiters
            parts = [part.strip() for part in re.split(r'[|,;\t]', line) if part.strip()]
            if len(parts) >= 2:
                table_data.append(parts)
    
    if len(table_data) >= 2:
        df = pd.DataFrame(table_data)
        df = df.replace('', pd.NA).dropna(how='all')
        
        # Try to detect header
        if len(df) > 1:
            first_row = df.iloc[0].astype(str)
            if first_row.str.contains(r'[a-zA-Z]').any():
                df.columns = first_row
                df = df[1:].reset_index(drop=True)
        
        return df
    
    return pd.DataFrame()

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
                        'Data Cleaning', 'Pattern Detection Confidence'
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
                        f"{st.session_state.get('pattern_confidence', {}).get('overall', 0):.1%}"
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
            help="Upload any document containing rows and columns - tables, forms, lists, etc.",
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
            
            **Version:** 3.0.0 (Pattern Detection)
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

def render_extraction_settings() -> Tuple[str, bool, int, int, List[int], str, int, bool, float]:
    """Render extraction settings and return configuration"""
    st.header("🎯 Extraction Settings")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        extraction_mode = st.selectbox(
            "Extraction Mode",
            options=[
                "Auto-detect (Any Rows/Columns) - Recommended",
                "Pattern Detection (Images/Scans)",
                "Traditional Tables (PDF only)",
                "Text-based (Fallback)",
                "All Methods (Slow but thorough)"
            ],
            index=0,
            help="Choose how to detect rows and columns in your document"
        )
        
        # Map selection to mode string
        mode_map = {
            "Auto-detect (Any Rows/Columns) - Recommended": "Auto-detect (Any Rows/Columns)",
            "Pattern Detection (Images/Scans)": "Pattern Detection",
            "Traditional Tables (PDF only)": "Traditional Tables",
            "Text-based (Fallback)": "Text-based",
            "All Methods (Slow but thorough)": "All Methods"
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
            min_rows = st.number_input(
                "Minimum rows", 
                2, 1000, 2,
                help="Minimum number of rows to consider as a valid table"
            )
            
            min_cols = st.number_input(
                "Minimum columns", 
                2, 50, 2,
                help="Minimum number of columns to consider as a valid table"
            )
            
            min_confidence = st.slider(
                "Minimum confidence", 
                0.0, 1.0, 0.3, 0.05,
                help="Minimum confidence level for pattern detection (lower = more regions, higher = more accurate)"
            )
        
        with col2:
            ocr_confidence = st.slider(
                "OCR Confidence (%)", 
                0, 100, 30,
                help="Higher values mean more accurate but fewer results"
            )
            
            enhance_handwriting = st.checkbox("Enhance image", value=True,
                                             help="Apply additional preprocessing for better OCR")
            
            ocr_lang = st.selectbox(
                "OCR Language",
                options=['eng', 'fra', 'deu', 'spa', 'ita', 'por', 'rus', 'chi_sim', 'jpn', 'kor'],
                index=0,
                help="Select language for OCR"
            )
    
    return extraction_mode, debug_mode, min_rows, min_cols, selected_pages, ocr_lang, ocr_confidence, enhance_handwriting, min_confidence

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
    
    # Show pattern detection confidence if available
    if st.session_state.get('detected_regions'):
        overall_confidence = np.mean([r.get('confidence', 0) for r in st.session_state.detected_regions.values()])
        st.session_state.pattern_confidence = {'overall': overall_confidence}
        
        st.markdown(f"""
        <div class="info-box">
        <strong>🔍 Pattern Detection Confidence:</strong> {overall_confidence:.1%}<br>
        <small>Higher confidence means more reliable row/column detection</small>
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
            
            # Show detected regions for this page if in debug mode
            if st.session_state.debug_mode and st.session_state.get('detected_regions', {}).get(page_num):
                detection = st.session_state.detected_regions[page_num]
                st.markdown(f"**Detection Info:** {detection['region_count']} regions, {detection['methods_used']}")
            
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
    <h1>📊 Advanced Document to Excel Converter</h1>
    <p style='font-size: 18px;'>Extract <strong>ANY rows and columns</strong> from any document type - printed, handwritten, scanned, or digital.</p>
    <p><strong>Supported Formats:</strong> PDF, PNG, JPG, JPEG, TIFF, BMP, GIF, WebP</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature showcase
    st.markdown("### ✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Intelligent Pattern Detection</div>
            <div class="feature-description">Automatically detects any rows & columns pattern - tables, forms, lists, invoices, receipts</div>
        </div>
        
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">Smart Column Detection</div>
            <div class="feature-description">Identifies debit, credit, date, amount, and other column types automatically</div>
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
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Multi-Method Detection</div>
            <div class="feature-description">Uses grid detection, text alignment, line clustering, and density analysis</div>
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
            <div class="step-text">Choose Detection Mode</div>
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
        - **For clear tables**: Use "Auto-detect" mode for fastest extraction
        - **For handwritten forms**: Enable image enhancement and adjust OCR confidence
        - **For complex layouts**: Use "All Methods" for thorough detection
        - **For better accuracy**: Use high-quality scans (300 DPI or higher)
        - **For photos of documents**: Ensure good lighting and straight angle
        - **For forms without lines**: Pattern Detection mode works best
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
                <strong>Advanced Document to Excel Converter</strong> • Version 3.0.0<br>
                Built with Streamlit, OpenCV, Tesseract OCR, scikit-learn, and pdfplumber<br>
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
    
    # Main content
    if st.session_state.pdf_uploaded and uploaded_file:
        render_document_info()
        
        # Get extraction settings
        extraction_mode, debug_mode, min_rows, min_cols, selected_pages, ocr_lang, ocr_confidence, enhance_handwriting, min_confidence = render_extraction_settings()
        
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
                    
                    # Extract using enhanced pattern detection
                    tables_by_page = extract_any_pattern_from_document(
                        file_path, 
                        selected_pages, 
                        st.session_state.extraction_mode,
                        ocr_lang=ocr_lang,
                        min_confidence=min_confidence,
                        enhance=enhance_handwriting
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
                                    'confidence': ocr_confidence,
                                    'enhance_handwriting': enhance_handwriting
                                },
                                'detection_confidence': st.session_state.get('pattern_confidence', {})
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

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
