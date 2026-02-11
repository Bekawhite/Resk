# app_enhanced.py - Enhanced PDF to Excel with filtering options
import streamlit as st
import pandas as pd
import tempfile
import os
import numpy as np

st.set_page_config(page_title="Enhanced PDF to Excel Converter", layout="wide")

st.title("📄 Enhanced PDF to Excel Converter")

# Initialize session state for storing extracted data
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None
if 'available_columns' not in st.session_state:
    st.session_state.available_columns = {}
if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = None

# File upload section
uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])

def extract_pdf_data(pdf_path):
    """Extract data from PDF using multiple methods"""
    extracted_data = {}
    available_columns = {}
    
    # Method 1: Try with pdfplumber (for text extraction)
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            all_text = []
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    lines = text.split('\n')
                    for line in lines:
                        # Create structured data from text lines
                        if line.strip():
                            parts = line.split()
                            if len(parts) > 1:
                                # Create a simple structured format
                                row_data = {f'Col_{j+1}': part for j, part in enumerate(parts)}
                                all_text.append(row_data)
            
            if all_text:
                df_text = pd.DataFrame(all_text)
                extracted_data['Text_Data'] = df_text
                available_columns['Text_Data'] = list(df_text.columns)
    except Exception as e:
        st.warning(f"pdfplumber extraction had issues: {e}")

    # Method 2: Try with tabula (for table extraction)
    try:
        from tabula import read_pdf
        tables = read_pdf(pdf_path, pages='all', multiple_tables=True, lattice=True)
        
        for i, df in enumerate(tables):
            if not df.empty and len(df) > 0:
                # Clean up column names
                df = df.copy()
                df.columns = [str(col).strip() for col in df.columns]
                df = df.dropna(how='all')  # Remove completely empty rows
                
                if len(df) > 0:
                    sheet_name = f'Table_{i+1}'
                    extracted_data[sheet_name] = df
                    available_columns[sheet_name] = list(df.columns)
    except Exception as e:
        st.warning(f"Tabula extraction had issues: {e}")
    
    # Method 3: If no tables found, create structured data from text
    if not extracted_data:
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                all_lines = []
                for i, page in enumerate(pdf.pages[:3]):  # First 3 pages only
                    text = page.extract_text()
                    if text:
                        lines = text.split('\n')
                        for line_num, line in enumerate(lines):
                            if line.strip():
                                # Split line into potential columns
                                parts = [part.strip() for part in line.split() if part.strip()]
                                if parts:
                                    row_dict = {'Line_Number': f"{i+1}_{line_num+1}"}
                                    for idx, part in enumerate(parts[:10]):  # Max 10 columns
                                        row_dict[f'Column_{idx+1}'] = part
                                    all_lines.append(row_dict)
                
                if all_lines:
                    df_lines = pd.DataFrame(all_lines)
                    extracted_data['Structured_Text'] = df_lines
                    available_columns['Structured_Text'] = list(df_lines.columns)
        except:
            pass
    
    return extracted_data, available_columns

if uploaded_file:
    if st.session_state.pdf_name != uploaded_file.name:
        st.session_state.pdf_name = uploaded_file.name
        st.info(f"Processing: {uploaded_file.name}")
        
        # Save PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(uploaded_file.getvalue())
            pdf_path = tmp.name
        
        try:
            # Extract data from PDF
            with st.spinner("Extracting data from PDF..."):
                extracted_data, available_columns = extract_pdf_data(pdf_path)
                st.session_state.extracted_data = extracted_data
                st.session_state.available_columns = available_columns
            
            if not extracted_data:
                st.error("Could not extract any structured data from PDF")
            else:
                st.success(f"Successfully extracted {len(extracted_data)} sheet(s) from PDF")
                
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
        
        finally:
            # Cleanup temporary PDF
            try:
                os.unlink(pdf_path)
            except:
                pass

# Display controls if data is available
if st.session_state.extracted_data:
    st.divider()
    st.header("📊 Excel Export Configuration")
    
    # Create tabs for different sheets
    sheet_names = list(st.session_state.extracted_data.keys())
    tabs = st.tabs(sheet_names)
    
    all_selected_columns = {}
    all_row_ranges = {}
    
    for idx, (sheet_name, tab) in enumerate(zip(sheet_names, tabs)):
        with tab:
            df = st.session_state.extracted_data[sheet_name]
            
            st.subheader(f"{sheet_name} - Data Preview")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Display preview
            st.dataframe(df.head(20), use_container_width=True)
            
            # Row selection options
            st.subheader("📏 Row Selection")
            
            col1, col2 = st.columns(2)
            
            with col1:
                row_option = st.radio(
                    f"Row selection for {sheet_name}",
                    ["All rows", "Custom range", "First N rows"],
                    key=f"row_option_{sheet_name}"
                )
            
            with col2:
                if row_option == "All rows":
                    start_row = 0
                    end_row = len(df)
                    st.info(f"All rows will be exported ({len(df)} rows)")
                elif row_option == "Custom range":
                    start_row = st.number_input(
                        "Start row (0-based)",
                        min_value=0,
                        max_value=len(df)-1,
                        value=0,
                        key=f"start_{sheet_name}"
                    )
                    end_row = st.number_input(
                        "End row",
                        min_value=start_row+1,
                        max_value=len(df),
                        value=min(1000, len(df)),
                        key=f"end_{sheet_name}"
                    )
                else:  # First N rows
                    n_rows = st.number_input(
                        "Number of rows",
                        min_value=1,
                        max_value=len(df),
                        value=min(1000, len(df)),
                        key=f"nrows_{sheet_name}"
                    )
                    start_row = 0
                    end_row = n_rows
            
            all_row_ranges[sheet_name] = (start_row, end_row)
            
            # Column selection
            st.subheader("🗂️ Column Selection")
            
            available_cols = st.session_state.available_columns.get(sheet_name, df.columns.tolist())
            
            if len(available_cols) > 20:
                st.warning(f"This sheet has {len(available_cols)} columns. Showing first 20.")
                display_cols = available_cols[:20]
            else:
                display_cols = available_cols
            
            # Show column preview
            col_preview_df = pd.DataFrame({
                'Column Name': display_cols,
                'Sample Data': [str(df[col].iloc[0]) if col in df.columns and len(df) > 0 else '' for col in display_cols]
            })
            st.dataframe(col_preview_df, use_container_width=True)
            
            # Column selection multiselect
            if len(available_cols) <= 50:  # Reasonable number for multiselect
                selected_columns = st.multiselect(
                    f"Select columns to export for {sheet_name}",
                    options=available_cols,
                    default=available_cols[:min(10, len(available_cols))],  # Default to first 10 columns
                    key=f"cols_{sheet_name}"
                )
            else:
                # For too many columns, provide checkboxes for common patterns
                st.info("Too many columns for selection. You can:")
                
                col_option = st.radio(
                    f"Column selection method for {sheet_name}",
                    ["All columns", "First N columns", "Columns containing..."],
                    key=f"col_method_{sheet_name}"
                )
                
                if col_option == "All columns":
                    selected_columns = available_cols
                elif col_option == "First N columns":
                    n_cols = st.number_input(
                        "Number of columns",
                        min_value=1,
                        max_value=len(available_cols),
                        value=min(10, len(available_cols)),
                        key=f"ncols_{sheet_name}"
                    )
                    selected_columns = available_cols[:n_cols]
                else:
                    search_term = st.text_input("Search term in column names", key=f"search_{sheet_name}")
                    if search_term:
                        selected_columns = [col for col in available_cols if search_term.lower() in str(col).lower()]
                    else:
                        selected_columns = available_cols
            
            if not selected_columns:
                st.warning("No columns selected. Will export all columns.")
                selected_columns = available_cols
            
            all_selected_columns[sheet_name] = selected_columns
    
    st.divider()
    
    # Export button
    if st.button("🚀 Generate Excel File", type="primary"):
        with st.spinner("Creating Excel file..."):
            # Create temporary Excel file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_excel:
                excel_path = tmp_excel.name
            
            try:
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    for sheet_name in sheet_names:
                        df = st.session_state.extracted_data[sheet_name]
                        start_row, end_row = all_row_ranges[sheet_name]
                        selected_columns = all_selected_columns[sheet_name]
                        
                        # Apply row filtering
                        filtered_df = df.iloc[start_row:end_row]
                        
                        # Apply column selection
                        # Only select columns that exist in the dataframe
                        existing_cols = [col for col in selected_columns if col in df.columns]
                        if existing_cols:
                            filtered_df = filtered_df[existing_cols]
                        
                        # Write to Excel with truncated sheet name if needed
                        safe_sheet_name = sheet_name[:31]  # Excel sheet name limit
                        filtered_df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
                
                # Provide download
                with open(excel_path, 'rb') as f:
                    excel_data = f.read()
                
                st.success("Excel file created successfully!")
                
                # Show export summary
                st.subheader("📋 Export Summary")
                summary_data = []
                for sheet_name in sheet_names:
                    df = st.session_state.extracted_data[sheet_name]
                    start_row, end_row = all_row_ranges[sheet_name]
                    selected_columns = all_selected_columns[sheet_name]
                    
                    exported_rows = min(end_row, len(df)) - start_row
                    exported_cols = len([col for col in selected_columns if col in df.columns])
                    
                    summary_data.append({
                        'Sheet': sheet_name,
                        'Total Rows': len(df),
                        'Exported Rows': exported_rows,
                        'Total Columns': len(df.columns),
                        'Exported Columns': exported_cols
                    })
                
                st.dataframe(pd.DataFrame(summary_data))
                
                # Download button
                output_filename = f"{st.session_state.pdf_name.replace('.pdf', '')}_filtered.xlsx"
                st.download_button(
                    label="📥 Download Excel File",
                    data=excel_data,
                    file_name=output_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
            except Exception as e:
                st.error(f"Error creating Excel file: {e}")
            
            finally:
                # Cleanup
                try:
                    os.unlink(excel_path)
                except:
                    pass

# Instructions
with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    1. **Upload a PDF** file containing tables or structured data
    2. **Wait for processing** - The app will extract data using multiple methods
    3. **Configure each sheet**:
       - **Row Selection**: Choose between:
         - All rows
         - Custom range (e.g., rows 1-1000)
         - First N rows
       - **Column Selection**: Select specific columns to export
    4. **Generate Excel** - Click the "Generate Excel File" button
    5. **Download** your filtered Excel file
    
    **Tips**:
    - The app may extract multiple sheets from your PDF
    - Each sheet can have different row/column selections
    - For PDFs with many columns, use the search feature to find specific columns
    """)
