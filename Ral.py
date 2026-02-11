# app_pdf_table_extractor_nojava.py - PDF Table Extractor without Java dependency
import streamlit as st
import pandas as pd
import tempfile
import os
import re
from typing import List, Dict, Tuple
import numpy as np

st.set_page_config(
    page_title="PDF Table Extractor",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #17a2b8;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    .column-selector {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #dee2e6;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 PDF Table Extractor")
st.markdown("Extract tabular data from PDFs without Java dependency")

# Initialize session states
if 'pdf_uploaded' not in st.session_state:
    st.session_state.pdf_uploaded = False
if 'tables_data' not in st.session_state:
    st.session_state.tables_data = {}
if 'pdf_metadata' not in st.session_state:
    st.session_state.pdf_metadata = {}
if 'selected_tables' not in st.session_state:
    st.session_state.selected_tables = {}
if 'column_selections' not in st.session_state:
    st.session_state.column_selections = {}
if 'row_selections' not in st.session_state:
    st.session_state.row_selections = {}
if 'all_columns' not in st.session_state:
    st.session_state.all_columns = {}

def is_table_like(data: List[List[str]]) -> bool:
    """Check if data looks like a table"""
    if not data or len(data) < 2:
        return False
    
    # Check for consistent number of columns
    col_counts = [len(row) for row in data]
    if max(col_counts) - min(col_counts) > 2:
        return False
    
    # Check for table patterns (numbers, dates, structured data)
    num_cells = 0
    for row in data[:5]:  # Check first 5 rows
        for cell in row:
            if cell and isinstance(cell, str):
                # Check for numbers, dates, currency
                if re.search(r'\d+', cell):
                    num_cells += 1
                # Check for common table patterns
                if any(pattern in cell.lower() for pattern in ['$', '€', '£', '%', '/', '-', ':']):
                    num_cells += 1
    
    # If more than 30% of cells have table-like patterns
    total_cells = sum(len(row) for row in data[:5])
    if total_cells > 0 and (num_cells / total_cells) > 0.3:
        return True
    
    return len(data) >= 3 and len(data[0]) >= 2

def extract_tables_with_pdfplumber(pdf_path: str, pages: List[int]) -> Dict[int, List[pd.DataFrame]]:
    """Extract tables using pdfplumber only"""
    import pdfplumber
    from pdfplumber.table import TableSettings
    
    tables_by_page = {}
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num in pages:
            page = pdf.pages[page_num - 1]
            
            # Method 1: Use pdfplumber's table extraction
            tables = page.extract_tables()
            page_tables = []
            
            for table_data in tables:
                if table_data and len(table_data) > 1:
                    try:
                        # Convert to DataFrame
                        df = pd.DataFrame(table_data)
                        
                        # Clean up the DataFrame
                        df = df.replace('', pd.NA).dropna(how='all').reset_index(drop=True)
                        
                        # Remove empty columns
                        df = df.dropna(axis=1, how='all')
                        
                        # Set first non-empty row as header if it looks like header
                        if len(df) > 1:
                            first_row = df.iloc[0].astype(str)
                            # Check if first row might be header (contains text, not just numbers)
                            if first_row.str.contains(r'[a-zA-Z]').any():
                                df.columns = first_row
                                df = df[1:].reset_index(drop=True)
                        
                        # Rename columns if needed
                        df.columns = [f'Column_{i+1}' if pd.isna(col) or str(col).strip() == '' else str(col).strip() 
                                    for i, col in enumerate(df.columns)]
                        
                        if not df.empty and len(df) > 0:
                            page_tables.append(df)
                    except Exception as e:
                        st.warning(f"Error processing table on page {page_num}: {e}")
            
            # Method 2: Extract text and look for tabular patterns
            if not page_tables:
                text = page.extract_text()
                if text:
                    lines = text.split('\n')
                    
                    # Look for tabular patterns (aligned columns, consistent spacing)
                    tabular_lines = []
                    current_table = []
                    
                    for line in lines:
                        # Check if line has tabular pattern (multiple items separated by 2+ spaces)
                        parts = [part.strip() for part in re.split(r'\s{2,}', line) if part.strip()]
                        
                        if len(parts) >= 2:
                            current_table.append(parts)
                        elif current_table:
                            if len(current_table) >= 2:
                                # Convert to DataFrame
                                try:
                                    # Find max columns
                                    max_cols = max(len(row) for row in current_table)
                                    padded_table = [row + [''] * (max_cols - len(row)) for row in current_table]
                                    
                                    df = pd.DataFrame(padded_table)
                                    df = df.replace('', pd.NA).dropna(how='all').reset_index(drop=True)
                                    df = df.dropna(axis=1, how='all')
                                    
                                    if not df.empty and len(df) >= 2:
                                        page_tables.append(df)
                                except:
                                    pass
                            current_table = []
                    
                    # Check last table
                    if current_table and len(current_table) >= 2:
                        try:
                            max_cols = max(len(row) for row in current_table)
                            padded_table = [row + [''] * (max_cols - len(row)) for row in current_table]
                            df = pd.DataFrame(padded_table)
                            df = df.replace('', pd.NA).dropna(how='all').reset_index(drop=True)
                            df = df.dropna(axis=1, how='all')
                            if not df.empty and len(df) >= 2:
                                page_tables.append(df)
                        except:
                            pass
            
            if page_tables:
                tables_by_page[page_num] = page_tables
    
    return tables_by_page

def analyze_columns_for_patterns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Analyze columns for common patterns like debit, credit, etc."""
    column_analysis = {}
    
    for col in df.columns:
        col_name = str(col).lower().strip()
        col_data = df[col].astype(str).str.lower()
        
        patterns = {
            'debit': ['debit', 'dr', 'withdrawal', 'charge', 'payment'],
            'credit': ['credit', 'cr', 'deposit', 'receipt', 'income'],
            'amount': ['amount', 'amt', 'value', 'total', 'sum'],
            'date': ['date', 'time', 'day', 'month', 'year'],
            'description': ['description', 'desc', 'detail', 'note', 'remark'],
            'balance': ['balance', 'bal', 'remaining'],
            'account': ['account', 'acct', 'account no', 'acc no'],
            'name': ['name', 'customer', 'client', 'person'],
            'id': ['id', 'no.', 'number', 'ref', 'reference'],
            'status': ['status', 'state', 'condition'],
        }
        
        matches = []
        for pattern_name, keywords in patterns.items():
            # Check column name
            if any(keyword in col_name for keyword in keywords):
                matches.append(pattern_name)
            
            # Check sample data for patterns
            if len(matches) == 0:
                sample = col_data.head(20).dropna()
                if not sample.empty:
                    sample_text = ' '.join(sample)
                    if any(keyword in sample_text for keyword in keywords):
                        matches.append(pattern_name)
        
        column_analysis[col] = matches[:3]  # Top 3 matches
    
    return column_analysis

# Sidebar
with st.sidebar:
    st.header("📁 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF file containing tables"
    )
    
    if uploaded_file:
        if not st.session_state.pdf_uploaded or st.session_state.get('current_file') != uploaded_file.name:
            st.session_state.pdf_uploaded = True
            st.session_state.current_file = uploaded_file.name
            
            # Get PDF metadata
            try:
                import pdfplumber
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    tmp.write(uploaded_file.getvalue())
                    pdf_path = tmp.name
                
                with pdfplumber.open(pdf_path) as pdf:
                    total_pages = len(pdf.pages)
                    st.session_state.pdf_metadata = {
                        'total_pages': total_pages,
                        'file_name': uploaded_file.name,
                        'file_size': f"{uploaded_file.size / 1024:.1f} KB"
                    }
                
                st.success(f"✅ PDF loaded successfully")
                st.info(f"📄 Total pages: {total_pages}")
                
                os.unlink(pdf_path)
                
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
    
    st.markdown("---")
    st.markdown("""
    **How it works:**
    1. Upload your PDF
    2. Select pages to scan
    3. Extract & analyze tables
    4. Choose columns & rows
    5. Export to Excel
    """)

# Main content
if st.session_state.pdf_uploaded:
    # Display PDF info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("File", st.session_state.pdf_metadata['file_name'][:20] + "...")
    with col2:
        st.metric("Pages", st.session_state.pdf_metadata['total_pages'])
    with col3:
        st.metric("Size", st.session_state.pdf_metadata['file_size'])
    
    st.markdown("---")
    
    # Page selection
    st.header("📄 Select Pages to Scan")
    
    total_pages = st.session_state.pdf_metadata['total_pages']
    
    page_options = st.radio(
        "Scan mode:",
        ["Quick scan (first 20 pages)", "Custom page range", "Scan all pages"],
        horizontal=True
    )
    
    selected_pages = []
    
    if page_options == "Quick scan (first 20 pages)":
        selected_pages = list(range(1, min(21, total_pages + 1)))
        st.info(f"Will scan pages 1-{len(selected_pages)} for tables")
    
    elif page_options == "Custom page range":
        col1, col2 = st.columns(2)
        with col1:
            start_page = st.number_input("From page", 1, total_pages, 1)
        with col2:
            end_page = st.number_input("To page", start_page, total_pages, min(start_page + 20, total_pages))
        selected_pages = list(range(start_page, end_page + 1))
        st.info(f"Will scan pages {start_page}-{end_page}")
    
    else:  # Scan all pages
        selected_pages = list(range(1, total_pages + 1))
        if total_pages > 50:
            st.warning(f"⚠️ Scanning all {total_pages} pages may take a while")
        else:
            st.info(f"Will scan all {total_pages} pages")
    
    # Advanced options
    with st.expander("⚙️ Table Detection Settings"):
        col1, col2 = st.columns(2)
        with col1:
            min_rows = st.number_input(
                "Minimum rows per table", 
                1, 1000000, 1,  # Changed: from 1 to 1,000,000, default 1
                help="Minimum number of rows a table must have to be extracted"
            )
            min_cols = st.number_input(
                "Minimum columns per table", 
                2, 50, 3,
                help="Minimum number of columns a table must have to be extracted"
            )
        with col2:
            extract_text_tables = st.checkbox("Extract text-based tables", value=True)
            merge_small_tables = st.checkbox("Merge adjacent small tables", value=True)
    
    # Extract button
    if selected_pages and st.button("🔍 Scan for Tables", type="primary", use_container_width=True):
        with st.spinner(f"Scanning {len(selected_pages)} pages for tables..."):
            # Save PDF temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(uploaded_file.getvalue())
                pdf_path = tmp.name
            
            try:
                # Extract tables
                tables_by_page = extract_tables_with_pdfplumber(pdf_path, selected_pages)
                
                # Filter tables based on criteria (flexible minimum rows)
                filtered_tables = {}
                total_tables_found = 0
                small_tables_ignored = 0
                
                for page_num, tables in tables_by_page.items():
                    filtered_page_tables = []
                    for table in tables:
                        if len(table) >= min_rows and len(table.columns) >= min_cols:
                            filtered_page_tables.append(table)
                            total_tables_found += 1
                        else:
                            small_tables_ignored += 1
                    
                    if filtered_page_tables:
                        filtered_tables[page_num] = filtered_page_tables
                
                st.session_state.tables_data = filtered_tables
                
                # Initialize selections
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
                
                if total_tables_found > 0:
                    if min_rows == 1:
                        st.success(f"✅ Found {total_tables_found} tables across {len(filtered_tables)} pages")
                    elif min_rows <= 10:
                        st.success(f"✅ Found {total_tables_found} tables (≥{min_rows} rows) across {len(filtered_tables)} pages")
                    else:
                        st.success(f"✅ Found {total_tables_found} large tables (≥{min_rows} rows) across {len(filtered_tables)} pages")
                    
                    if small_tables_ignored > 0:
                        st.info(f"ℹ️ Ignored {small_tables_ignored} tables with less than {min_rows} rows")
                else:
                    st.warning(f"⚠️ No tables found with {min_rows}+ rows. Try reducing the minimum rows or select more pages.")
                
            except Exception as e:
                st.error(f"Error extracting tables: {e}")
            finally:
                try:
                    os.unlink(pdf_path)
                except:
                    pass

# Display extracted tables with column/row selection
if st.session_state.tables_data:
    st.markdown("---")
    st.header("📋 Select Columns & Rows for Export")
    
    # Summary
    total_tables = sum(len(tables) for tables in st.session_state.tables_data.values())
    selected_tables = sum(1 for info in st.session_state.selected_tables.values() if info.get("selected", False))
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Tables Found", total_tables)
    with col2:
        st.metric("Selected for Export", selected_tables)
    
    # Table browser with column/row selection
    pages_with_tables = sorted(st.session_state.tables_data.keys())
    
    if pages_with_tables:
        # Create tabs for each page with tables
        tabs = st.tabs([f"Page {page}" for page in pages_with_tables])
        
        for tab_idx, (page_num, tab) in enumerate(zip(pages_with_tables, tabs)):
            with tab:
                tables_on_page = st.session_state.tables_data[page_num]
                st.subheader(f"📄 Page {page_num} - {len(tables_on_page)} table(s)")
                
                for table_idx, table in enumerate(tables_on_page):
                    # Find table ID
                    table_id = None
                    for t_id, t_info in st.session_state.selected_tables.items():
                        if t_info["page"] == page_num and t_info["table_idx"] == table_idx:
                            table_id = t_id
                            break
                    
                    if table_id:
                        # Table header
                        row_label = "rows" if min_rows == 1 else f"rows (≥{min_rows})"
                        st.markdown(f"### Table {table_idx + 1} ({len(table)} rows × {len(table.columns)} columns)")
                        
                        # Table selection
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            is_selected = st.checkbox(
                                "Include this table in export",
                                value=st.session_state.selected_tables[table_id].get("selected", True),
                                key=f"select_{table_id}",
                                help="Uncheck to exclude this table from export"
                            )
                            st.session_state.selected_tables[table_id]["selected"] = is_selected
                        
                        with col2:
                            if is_selected:
                                st.success("✅ This table will be exported")
                            else:
                                st.warning("❌ This table will NOT be exported")
                        
                        if is_selected:
                            # Column selection section
                            st.markdown("#### 🗂️ Select Columns")
                            
                            # Analyze columns for patterns
                            column_analysis = analyze_columns_for_patterns(table)
                            
                            # Create multi-select for columns
                            all_columns = table.columns.tolist()
                            
                            # Default columns to select (based on analysis)
                            default_selected = []
                            for col in all_columns:
                                if col in st.session_state.column_selections.get(table_id, {}):
                                    if st.session_state.column_selections[table_id].get(col, True):
                                        default_selected.append(col)
                            
                            selected_columns = st.multiselect(
                                f"Choose columns for Table {table_idx + 1}",
                                options=all_columns,
                                default=default_selected,
                                key=f"cols_{table_id}",
                                help="Select columns to include in export"
                            )
                            
                            # Update column selections
                            st.session_state.column_selections[table_id] = {
                                col: (col in selected_columns) for col in all_columns
                            }
                            
                            # Show column analysis tags
                            if selected_columns:
                                st.markdown("**Detected patterns in selected columns:**")
                                cols_container = st.container()
                                with cols_container:
                                    cols = st.columns(min(5, len(selected_columns)))
                                    for idx, col in enumerate(selected_columns[:5]):
                                        with cols[idx % 5]:
                                            if col in column_analysis and column_analysis[col]:
                                                tags = ", ".join(column_analysis[col][:2])
                                                st.caption(f"**{col[:15]}...**")
                                                st.markdown(f"<small>{tags}</small>", unsafe_allow_html=True)
                            
                            # Row selection section
                            st.markdown("#### 📊 Select Rows Range")
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
                                    st.info(f"Will export rows {start_row:,} to {end_row:,} ({rows_to_export:,} rows)")
                            
                            # Preview section
                            with st.expander("👁️ Preview Selected Data", expanded=False):
                                # Get filtered data based on selections
                                if use_all_rows:
                                    preview_df = table[selected_columns] if selected_columns else table
                                else:
                                    preview_df = table.iloc[start_row:end_row+1][selected_columns] if selected_columns else table.iloc[start_row:end_row+1]
                                
                                if not preview_df.empty:
                                    st.dataframe(
                                        preview_df.head(50),
                                        use_container_width=True,
                                        height=400
                                    )
                                    st.caption(f"Showing first 50 rows of {len(preview_df):,} total rows")
                                else:
                                    st.warning("No data to preview. Please select at least one column.")
                    
                    st.markdown("---")
    
    # Export section
    st.markdown("---")
    st.header("🚀 Export to Excel")
    
    col1, col2 = st.columns(2)
    
    with col1:
        excel_name = st.text_input(
            "Excel file name",
            value=f"{st.session_state.pdf_metadata['file_name'].replace('.pdf', '')}_tables.xlsx"
        )
        
        export_mode = st.radio(
            "Export format:",
            ["Each table → Separate sheet", "All tables → One sheet", "Tables by page → Sheets by page"]
        )
        
        # Column naming options
        st.markdown("**Column Handling:**")
        rename_columns = st.checkbox("Use smart column names", value=True, 
                                     help="Rename columns based on detected patterns (e.g., Debit, Credit)")
    
    with col2:
        include_metadata = st.checkbox("Include metadata sheet", value=True)
        auto_format = st.checkbox("Auto-format columns", value=True)
        
        # Data cleaning options
        st.markdown("**Data Cleaning:**")
        remove_duplicates = st.checkbox("Remove duplicate rows", value=False)
        remove_empty = st.checkbox("Remove empty rows", value=True)
    
    # Export button
    if st.button("📥 Generate Excel File", type="primary", use_container_width=True):
        # Collect selected tables with column/row selections
        tables_to_export = []
        export_summary = []
        
        for table_id, table_info in st.session_state.selected_tables.items():
            if table_info.get("selected", False):
                df = table_info.get("df")
                if df is not None:
                    # Apply column selection
                    selected_cols = [
                        col for col, is_selected in st.session_state.column_selections.get(table_id, {}).items()
                        if is_selected
                    ]
                    
                    if selected_cols:  # Only export if at least one column is selected
                        # Apply row selection
                        row_selection = st.session_state.row_selections.get(table_id, {})
                        
                        if row_selection.get("all_rows", True):
                            filtered_df = df[selected_cols].copy()
                        else:
                            start_row = row_selection.get("start_row", 0)
                            end_row = row_selection.get("end_row", len(df) - 1)
                            filtered_df = df.iloc[start_row:end_row+1][selected_cols].copy()
                        
                        # Apply data cleaning
                        if remove_empty:
                            filtered_df = filtered_df.replace(r'^\s*$', np.nan, regex=True)
                            filtered_df = filtered_df.dropna(how='all')
                        
                        if remove_duplicates:
                            filtered_df = filtered_df.drop_duplicates()
                        
                        # Rename columns if requested
                        if rename_columns and not filtered_df.empty:
                            column_analysis = analyze_columns_for_patterns(filtered_df)
                            new_columns = []
                            for col in filtered_df.columns:
                                if column_analysis.get(col):
                                    # Use first detected pattern
                                    new_name = column_analysis[col][0].title()
                                    # Add index if duplicate
                                    if new_name in new_columns:
                                        new_name = f"{new_name}_{new_columns.count(new_name) + 1}"
                                    new_columns.append(new_name)
                                else:
                                    new_columns.append(col)
                            filtered_df.columns = new_columns
                        
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
        
        if not tables_to_export:
            st.warning("No data selected for export! Please select at least one column per table.")
        else:
            total_rows = sum(len(t["df"]) for t in tables_to_export)
            st.info(f"Preparing to export {total_rows:,} total rows from {len(tables_to_export)} tables...")
            
            with st.spinner(f"Creating Excel file with {total_rows:,} rows..."):
                # Create temporary Excel file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_excel:
                    excel_path = tmp_excel.name
                
                try:
                    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                        # Metadata sheet
                        if include_metadata:
                            metadata_df = pd.DataFrame({
                                'Property': [
                                    'File Name', 'Total Pages', 'Tables Exported', 
                                    'Total Rows Exported', 'Export Date', 'Export Settings'
                                ],
                                'Value': [
                                    st.session_state.pdf_metadata['file_name'],
                                    st.session_state.pdf_metadata['total_pages'],
                                    len(tables_to_export),
                                    total_rows,
                                    pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    f"Smart Names: {rename_columns}, Cleaned: {remove_duplicates}|{remove_empty}"
                                ]
                            })
                            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
                        
                        # Export based on selected mode
                        if export_mode == "Each table → Separate sheet":
                            for i, table_data in enumerate(tables_to_export):
                                df = table_data["df"]
                                sheet_name = f"P{table_data['page']}_T{table_data['table_idx']+1}"
                                sheet_name = sheet_name[:31]  # Excel limit
                                df.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        elif export_mode == "All tables → One sheet":
                            all_dfs = []
                            for table_data in tables_to_export:
                                df = table_data["df"].copy()
                                df.insert(0, 'Source_Page', table_data["page"])
                                df.insert(1, 'Source_Table', table_data["table_idx"] + 1)
                                all_dfs.append(df)
                                # Add separator
                                separator = pd.DataFrame([{col: '---' for col in df.columns}])
                                all_dfs.append(separator)
                            
                            if all_dfs:
                                combined_df = pd.concat(all_dfs[:-1], ignore_index=True)  # Exclude last separator
                                combined_df.to_excel(writer, sheet_name='All_Tables', index=False)
                        
                        else:  # Tables by page → Sheets by page
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
                                    df.insert(0, 'Table', table_data["table_idx"] + 1)
                                    page_dfs.append(df)
                                    # Add separator
                                    separator = pd.DataFrame([{col: '---' for col in df.columns}])
                                    page_dfs.append(separator)
                                
                                if page_dfs:
                                    combined_page_df = pd.concat(page_dfs[:-1], ignore_index=True)
                                    sheet_name = f"Page_{page_num}"
                                    sheet_name = sheet_name[:31]
                                    combined_page_df.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        # Auto-format if enabled
                        if auto_format:
                            for sheet_name in writer.sheets:
                                worksheet = writer.sheets[sheet_name]
                                for column in worksheet.columns:
                                    max_length = 0
                                    column_letter = column[0].column_letter
                                    for cell in column:
                                        try:
                                            cell_value = str(cell.value) if cell.value is not None else ""
                                            if len(cell_value) > max_length:
                                                max_length = len(cell_value)
                                        except:
                                            pass
                                    adjusted_width = min(max_length + 2, 100)  # Increased max width
                                    worksheet.column_dimensions[column_letter].width = adjusted_width
                    
                    # Read the Excel file
                    with open(excel_path, 'rb') as f:
                        excel_data = f.read()
                    
                    # Success message
                    st.markdown(f"""
                    <div class="success-box">
                    <h3>✅ Excel File Created Successfully!</h3>
                    <p><strong>File:</strong> {excel_name}</p>
                    <p><strong>Tables exported:</strong> {len(tables_to_export)}</p>
                    <p><strong>Total rows exported:</strong> {total_rows:,}</p>
                    <p><strong>Average rows per table:</strong> {total_rows//len(tables_to_export):, if len(tables_to_export) > 0 else 0}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show summary
                    st.subheader("📊 Export Summary")
                    summary_df = pd.DataFrame(export_summary)
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Download button
                    st.download_button(
                        label=f"⬇️ Download Excel File ({total_rows:,} rows)",
                        data=excel_data,
                        file_name=excel_name if excel_name.endswith('.xlsx') else excel_name + '.xlsx',
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"Error creating Excel file: {e}")
                finally:
                    try:
                        os.unlink(excel_path)
                    except:
                        pass

else:
    # Welcome screen
    st.markdown("""
    <div class="info-box">
    <h3>📊 PDF Table Extractor</h3>
    <p>Extract tabular data from PDF files <strong>without Java dependency</strong>.</p>
    <p>This tool allows you to extract tables of any size with customizable column selection.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ **Key Features**")
        st.markdown("""
        - **Flexible table size** (1 row to 1,000,000+ rows)
        - **Smart column detection** (Debit, Credit, etc.)
        - **Column-by-column selection**
        - **Row range selection**
        - **No Java installation required**
        - **Multiple export formats**
        """)
    
    with col2:
        st.markdown("### 📋 **Table Size Options**")
        st.markdown("""
        - Extract small tables (1+ rows)
        - Extract medium tables (10+ rows)
        - Extract large tables (1000+ rows)
        - Extract very large tables (10000+ rows)
        - Customize minimum rows as needed
        """)
    
    st.markdown("---")
    st.markdown("*Upload a PDF file using the sidebar to begin*")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>PDF Table Extractor • Flexible Table Size • Built with Streamlit</div>",
    unsafe_allow_html=True
)
