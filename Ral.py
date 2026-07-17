# app_pdf_table_extractor_nojava.py - PDF Table Extractor without Java dependency
import streamlit as st
import pandas as pd
import tempfile
import os
import re
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime

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

def make_columns_unique(columns):
    """Make column names unique by appending numbers if needed"""
    seen = {}
    unique_columns = []
    for col in columns:
        col_str = str(col).strip()
        if col_str in seen:
            seen[col_str] += 1
            unique_columns.append(f"{col_str}_{seen[col_str]}")
        else:
            seen[col_str] = 0
            unique_columns.append(col_str)
    return unique_columns

def clean_numeric_value(val):
    """Clean numeric values by removing commas and converting to float"""
    if val is None or pd.isna(val) or str(val).strip() == '':
        return None
    try:
        # Remove commas and convert to float
        cleaned = str(val).replace(',', '').strip()
        return float(cleaned)
    except:
        return val

def convert_to_proper_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert DataFrame columns to proper data types for Excel compatibility"""
    df_converted = df.copy()
    
    for col in df_converted.columns:
        # Skip if column is empty
        if df_converted[col].isna().all():
            continue
        
        # Try to convert to numeric
        try:
            # Clean the data: remove currency symbols and commas
            if df_converted[col].dtype == 'object':
                # Check if column contains numeric patterns
                sample = df_converted[col].dropna().iloc[0] if not df_converted[col].dropna().empty else ""
                if isinstance(sample, str):
                    # Remove currency symbols and commas
                    cleaned = df_converted[col].astype(str).str.replace(r'[$,€£¥₹]', '', regex=True)
                    cleaned = cleaned.str.replace(',', '')
                    numeric_col = pd.to_numeric(cleaned, errors='coerce')
                    if not numeric_col.isna().all():
                        df_converted[col] = numeric_col
                        continue
            
            # Direct numeric conversion
            numeric_col = pd.to_numeric(df_converted[col], errors='coerce')
            if not numeric_col.isna().all():
                df_converted[col] = numeric_col
                continue
        except:
            pass
        
        # Try to convert to datetime
        try:
            if df_converted[col].dtype == 'object':
                # Common date formats
                date_col = pd.to_datetime(df_converted[col], errors='coerce', infer_datetime_format=True)
                if not date_col.isna().all():
                    df_converted[col] = date_col
                    continue
        except:
            pass
        
        # If all else fails, ensure it's string type
        if df_converted[col].dtype == 'object':
            df_converted[col] = df_converted[col].astype(str)
    
    return df_converted

def extract_tables_with_pdfplumber(pdf_path: str, pages: List[int]) -> Dict[int, List[pd.DataFrame]]:
    """Extract tables using pdfplumber"""
    import pdfplumber
    
    tables_by_page = {}
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num in pages:
            page = pdf.pages[page_num - 1]
            page_tables = []
            
            # Try multiple extraction strategies
            strategies = [
                # Strategy 1: Default extraction
                lambda: page.extract_tables(),
                # Strategy 2: Text-based extraction
                lambda: page.extract_tables({
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                }),
                # Strategy 3: Lines-based extraction
                lambda: page.extract_tables({
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                })
            ]
            
            for strategy in strategies:
                try:
                    tables = strategy()
                    for table_data in tables:
                        if table_data and len(table_data) > 1:
                            df = process_table_data(table_data)
                            if df is not None and not df.empty and len(df) >= 2 and len(df.columns) >= 3:
                                page_tables.append(df)
                    if page_tables:
                        break
                except:
                    continue
            
            # If no tables found, try text extraction
            if not page_tables:
                text = page.extract_text()
                if text:
                    lines = [line for line in text.split('\n') if line.strip()]
                    
                    # Look for table-like structures in text
                    table_data = []
                    for line in lines:
                        # Try to split by multiple spaces
                        parts = [p.strip() for p in re.split(r'\s{2,}', line) if p.strip()]
                        if len(parts) >= 3:  # At least 3 columns
                            table_data.append(parts)
                    
                    if table_data:
                        df = process_table_data(table_data)
                        if df is not None and not df.empty and len(df) >= 2 and len(df.columns) >= 3:
                            page_tables.append(df)
            
            if page_tables:
                tables_by_page[page_num] = page_tables
    
    return tables_by_page

def process_table_data(table_data):
    """Process table data and convert to DataFrame with proper structure"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame(table_data)
        
        # Clean up - remove empty rows and columns
        df = df.replace('', pd.NA).replace('None', pd.NA)
        df = df.dropna(how='all').reset_index(drop=True)
        df = df.dropna(axis=1, how='all')
        
        if df.empty or len(df) < 2 or len(df.columns) < 2:
            return None
        
        # Try to detect headers - look for text in first row
        if len(df) > 0:
            first_row = df.iloc[0].astype(str)
            # Check if first row contains header keywords
            header_keywords = ['date', 'transaction', 'value', 'cheque', 'remarks', 'withdrawal', 'deposit', 'balance', 'amount', 'description']
            header_score = 0
            for val in first_row:
                val_lower = str(val).lower()
                if any(keyword in val_lower for keyword in header_keywords):
                    header_score += 1
            
            # If first row looks like headers, use it
            if header_score >= len(first_row) * 0.3:  # At least 30% match
                df.columns = first_row
                df = df[1:].reset_index(drop=True)
        
        # Clean and make column names unique
        new_columns = []
        for i, col in enumerate(df.columns):
            col_str = str(col).strip()
            if pd.isna(col) or col_str == '' or col is None:
                new_columns.append(f'Column_{i+1}')
            else:
                new_columns.append(col_str)
        
        df.columns = make_columns_unique(new_columns)
        
        # Clean numeric columns
        for col in df.columns:
            # Check if column contains numbers
            sample = df[col].dropna().head(5)
            if len(sample) > 0:
                # Try to clean numeric values
                cleaned_values = []
                for val in sample:
                    cleaned = clean_numeric_value(val)
                    if cleaned is not None:
                        cleaned_values.append(cleaned)
                
                # If most values converted to numbers, apply to entire column
                if len(cleaned_values) > 0 and len(cleaned_values) / len(sample) > 0.5:
                    try:
                        df[col] = df[col].apply(clean_numeric_value)
                    except:
                        pass
        
        # Try to detect and convert date columns
        for col in df.columns:
            try:
                sample = df[col].dropna().head(5)
                if len(sample) > 0:
                    # Try to convert to datetime
                    date_col = pd.to_datetime(sample, errors='coerce', infer_datetime_format=True)
                    if not date_col.isna().all():
                        if date_col.notna().sum() / len(sample) > 0.5:
                            df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
            except:
                pass
        
        return df
    except Exception as e:
        return None

def analyze_columns_for_patterns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Analyze columns for common patterns"""
    column_analysis = {}
    
    for col in df.columns:
        if col is None or pd.isna(col):
            continue
            
        col_name = str(col).lower().strip()
        if not col_name:
            continue
            
        try:
            col_data = df[col].astype(str).str.lower()
        except:
            continue
        
        patterns = {
            'date': ['date', 'day', 'month', 'year', 'time'],
            'description': ['description', 'desc', 'remark', 'note', 'detail'],
            'amount': ['amount', 'amt', 'value', 'total', 'sum'],
            'withdrawal': ['withdrawal', 'withdraw', 'dr', 'debit', 'payment'],
            'deposit': ['deposit', 'credit', 'cr', 'receipt', 'income'],
            'balance': ['balance', 'bal', 'remaining'],
            'account': ['account', 'acct', 'account no'],
            'cheque': ['cheque', 'check', 'chq'],
            'transaction': ['transaction', 'trans', 'txn'],
        }
        
        matches = []
        for pattern_name, keywords in patterns.items():
            if any(keyword in col_name for keyword in keywords):
                matches.append(pattern_name)
            
            if len(matches) == 0:
                try:
                    sample = col_data.head(20).dropna()
                    if not sample.empty:
                        sample_text = ' '.join(sample)
                        if any(keyword in sample_text for keyword in keywords):
                            matches.append(pattern_name)
                except:
                    pass
        
        column_analysis[col] = matches[:3]
    
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
    2. Click "Scan for Tables"
    3. Review and select columns
    4. Export to Excel
    """)

# Main content
if st.session_state.pdf_uploaded:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("File", st.session_state.pdf_metadata['file_name'][:20] + "...")
    with col2:
        st.metric("Pages", st.session_state.pdf_metadata['total_pages'])
    with col3:
        st.metric("Size", st.session_state.pdf_metadata['file_size'])
    
    st.markdown("---")
    
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
    
    else:
        selected_pages = list(range(1, total_pages + 1))
        if total_pages > 50:
            st.warning(f"⚠️ Scanning all {total_pages} pages may take a while")
        else:
            st.info(f"Will scan all {total_pages} pages")
    
    # Extract button
    if selected_pages and st.button("🔍 Scan for Tables", type="primary", use_container_width=True):
        with st.spinner(f"Scanning {len(selected_pages)} pages for tables..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(uploaded_file.getvalue())
                pdf_path = tmp.name
            
            try:
                tables_by_page = extract_tables_with_pdfplumber(pdf_path, selected_pages)
                
                filtered_tables = {}
                total_tables_found = 0
                
                for page_num, tables in tables_by_page.items():
                    filtered_page_tables = []
                    for table in tables:
                        if len(table) >= 2 and len(table.columns) >= 3:
                            filtered_page_tables.append(table)
                            total_tables_found += 1
                    
                    if filtered_page_tables:
                        filtered_tables[page_num] = filtered_page_tables
                
                st.session_state.tables_data = filtered_tables
                
                st.session_state.selected_tables = {}
                st.session_state.column_selections = {}
                st.session_state.row_selections = {}
                st.session_state.all_columns = {}
                
                table_counter = 1
                
                for page_num, tables in filtered_tables.items():
                    for table_idx, table in enumerate(tables):
                        table_id = f"table_{table_counter}"
                        
                        st.session_state.selected_tables[table_id] = {
                            "page": page_num,
                            "table_idx": table_idx,
                            "selected": True,
                            "df": table,
                            "shape": f"{len(table)}x{len(table.columns)}"
                        }
                        
                        valid_columns = [col for col in table.columns if col is not None and not pd.isna(col)]
                        st.session_state.column_selections[table_id] = {
                            col: True for col in valid_columns
                        }
                        
                        st.session_state.row_selections[table_id] = {
                            "start_row": 0,
                            "end_row": len(table) - 1,
                            "all_rows": True
                        }
                        
                        st.session_state.all_columns[table_id] = valid_columns
                        
                        table_counter += 1
                
                if total_tables_found > 0:
                    st.success(f"✅ Found {total_tables_found} tables across {len(filtered_tables)} pages")
                else:
                    st.warning("⚠️ No tables found. Try scanning different pages.")
                
            except Exception as e:
                st.error(f"Error extracting tables: {e}")
            finally:
                try:
                    os.unlink(pdf_path)
                except:
                    pass

# Display extracted tables
if st.session_state.tables_data:
    st.markdown("---")
    st.header("📋 Review Extracted Tables")
    
    total_tables = sum(len(tables) for tables in st.session_state.tables_data.values())
    selected_tables = sum(1 for info in st.session_state.selected_tables.values() if info.get("selected", False))
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Tables Found", total_tables)
    with col2:
        st.metric("Selected for Export", selected_tables)
    
    pages_with_tables = sorted(st.session_state.tables_data.keys())
    
    if pages_with_tables:
        tabs = st.tabs([f"Page {page}" for page in pages_with_tables])
        
        for tab_idx, (page_num, tab) in enumerate(zip(pages_with_tables, tabs)):
            with tab:
                tables_on_page = st.session_state.tables_data[page_num]
                st.subheader(f"📄 Page {page_num} - {len(tables_on_page)} table(s)")
                
                for table_idx, table in enumerate(tables_on_page):
                    table_id = None
                    for t_id, t_info in st.session_state.selected_tables.items():
                        if t_info["page"] == page_num and t_info["table_idx"] == table_idx:
                            table_id = t_id
                            break
                    
                    if table_id:
                        st.markdown(f"### Table {table_idx + 1} ({len(table)} rows × {len(table.columns)} columns)")
                        
                        # Show table preview
                        st.dataframe(table, use_container_width=True, height=300)
                        
                        # Column selection
                        st.markdown("#### Select Columns to Export")
                        all_columns = [col for col in table.columns if col is not None and not pd.isna(col)]
                        
                        selected_columns = st.multiselect(
                            f"Columns for Table {table_idx + 1}",
                            options=all_columns,
                            default=all_columns,
                            key=f"cols_{table_id}"
                        )
                        
                        st.session_state.column_selections[table_id] = {
                            col: (col in selected_columns) for col in all_columns
                        }
                        
                        # Row range selection
                        st.markdown("#### Select Rows Range")
                        total_rows = len(table)
                        
                        use_all_rows = st.checkbox(
                            "Export all rows",
                            value=True,
                            key=f"allrows_{table_id}"
                        )
                        st.session_state.row_selections[table_id]["all_rows"] = use_all_rows
                        
                        if not use_all_rows:
                            col1, col2 = st.columns(2)
                            with col1:
                                start_row = st.number_input(
                                    "Start row",
                                    0, total_rows - 1, 0,
                                    key=f"start_{table_id}"
                                )
                            with col2:
                                end_row = st.number_input(
                                    "End row",
                                    start_row + 1, total_rows - 1, total_rows - 1,
                                    key=f"end_{table_id}"
                                )
                            
                            st.session_state.row_selections[table_id]["start_row"] = start_row
                            st.session_state.row_selections[table_id]["end_row"] = end_row
                        
                        st.markdown("---")
    
    # Export section
    st.markdown("---")
    st.header("🚀 Export to Excel")
    
    col1, col2 = st.columns(2)
    
    with col1:
        excel_name = st.text_input(
            "Excel file name",
            value=f"{st.session_state.pdf_metadata['file_name'].replace('.pdf', '')}_transactions.xlsx"
        )
        
        export_mode = st.radio(
            "Export format:",
            ["Each table → Separate sheet", "All tables → One sheet"]
        )
    
    with col2:
        include_metadata = st.checkbox("Include metadata sheet", value=True)
        auto_format = st.checkbox("Auto-format columns", value=True)
        convert_numbers = st.checkbox("Convert numbers to proper format", value=True)
    
    # Export button
    if st.button("📥 Generate Excel File", type="primary", use_container_width=True):
        tables_to_export = []
        export_summary = []
        
        for table_id, table_info in st.session_state.selected_tables.items():
            if table_info.get("selected", False):
                df = table_info.get("df")
                if df is not None:
                    selected_cols = [
                        col for col, is_selected in st.session_state.column_selections.get(table_id, {}).items()
                        if is_selected and col is not None and not pd.isna(col)
                    ]
                    
                    if selected_cols:
                        row_selection = st.session_state.row_selections.get(table_id, {})
                        
                        try:
                            if row_selection.get("all_rows", True):
                                filtered_df = df[selected_cols].copy()
                            else:
                                start_row = row_selection.get("start_row", 0)
                                end_row = row_selection.get("end_row", len(df) - 1)
                                filtered_df = df.iloc[start_row:end_row+1][selected_cols].copy()
                            
                            # Convert numbers
                            if convert_numbers:
                                filtered_df = convert_to_proper_types(filtered_df)
                            
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
                        except Exception as e:
                            st.warning(f"Error processing table {table_id}: {e}")
        
        if not tables_to_export:
            st.warning("No data selected for export!")
        else:
            total_rows = sum(len(t["df"]) for t in tables_to_export)
            st.info(f"Preparing to export {total_rows:,} rows from {len(tables_to_export)} tables...")
            
            with st.spinner(f"Creating Excel file..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_excel:
                    excel_path = tmp_excel.name
                
                try:
                    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                        if include_metadata:
                            metadata_df = pd.DataFrame({
                                'Property': [
                                    'File Name', 'Total Pages', 'Tables Exported', 
                                    'Total Rows Exported', 'Export Date'
                                ],
                                'Value': [
                                    st.session_state.pdf_metadata['file_name'],
                                    st.session_state.pdf_metadata['total_pages'],
                                    len(tables_to_export),
                                    total_rows,
                                    pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                                ]
                            })
                            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
                        
                        if export_mode == "Each table → Separate sheet":
                            for i, table_data in enumerate(tables_to_export):
                                df = table_data["df"]
                                sheet_name = f"P{table_data['page']}_T{table_data['table_idx']+1}"
                                sheet_name = sheet_name[:31]
                                df.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        else:  # All tables → One sheet
                            all_dfs = []
                            for table_data in tables_to_export:
                                df = table_data["df"].copy()
                                df.insert(0, 'Source_Page', table_data["page"])
                                df.insert(1, 'Source_Table', table_data["table_idx"] + 1)
                                all_dfs.append(df)
                                separator = pd.DataFrame([{col: '---' for col in df.columns}])
                                all_dfs.append(separator)
                            
                            if all_dfs:
                                combined_df = pd.concat(all_dfs[:-1], ignore_index=True)
                                combined_df.to_excel(writer, sheet_name='All_Data', index=False)
                        
                        # Auto-format columns
                        if auto_format:
                            from openpyxl.utils import get_column_letter
                            
                            for sheet_name in writer.sheets:
                                worksheet = writer.sheets[sheet_name]
                                
                                for column in worksheet.columns:
                                    max_length = 0
                                    column_letter = get_column_letter(column[0].column)
                                    
                                    for cell in column:
                                        try:
                                            if cell.value:
                                                cell_value = str(cell.value)
                                                if len(cell_value) > max_length:
                                                    max_length = len(cell_value)
                                        except:
                                            pass
                                    
                                    adjusted_width = min(max_length + 2, 50)
                                    worksheet.column_dimensions[column_letter].width = adjusted_width
                    
                    with open(excel_path, 'rb') as f:
                        excel_data = f.read()
                    
                    st.markdown(f"""
                    <div class="success-box">
                    <h3>✅ Excel File Created Successfully!</h3>
                    <p><strong>File:</strong> {excel_name}</p>
                    <p><strong>Tables exported:</strong> {len(tables_to_export)}</p>
                    <p><strong>Total rows exported:</strong> {total_rows:,}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.subheader("📊 Export Summary")
                    summary_df = pd.DataFrame(export_summary)
                    st.dataframe(summary_df, use_container_width=True)
                    
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
    st.markdown("""
    <div class="info-box">
    <h3>📊 PDF Table Extractor</h3>
    <p>Upload a PDF file using the sidebar to extract tables.</p>
    <p>This tool will detect tables with 7 columns: Transaction Date, Value Date, Cheque Number, Transaction Remarks, Withdrawal, Deposit, and Balance.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>PDF Table Extractor • Built with Streamlit</div>",
    unsafe_allow_html=True
)
