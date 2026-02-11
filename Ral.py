# app_simple.py - Simplified version
import streamlit as st
import pandas as pd
import tempfile
import os

st.set_page_config(page_title="Simple PDF to Excel", layout="wide")

st.title("📄 Simple PDF to Excel Converter")

# Simple file upload
uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])

if uploaded_file:
    st.info(f"Uploaded: {uploaded_file.name}")
    
    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(uploaded_file.getvalue())
        pdf_path = tmp.name
    
    try:
        # Try different extraction methods
        extracted_data = {}
        
        # Method 1: Try with pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages[:5]):  # First 5 pages
                    text = page.extract_text()
                    if text:
                        lines = text.split('\n')
                        df = pd.DataFrame({'Content': lines})
                        extracted_data[f'Page_{i+1}'] = df
        except:
            pass
        
        # Method 2: Try with tabula
        try:
            from tabula import read_pdf
            tables = read_pdf(pdf_path, pages='all', multiple_tables=True)
            for i, df in enumerate(tables):
                if not df.empty:
                    extracted_data[f'Table_{i+1}'] = df
        except:
            pass
        
        if extracted_data:
            # Save to Excel
            excel_path = pdf_path.replace('.pdf', '.xlsx')
            with pd.ExcelWriter(excel_path) as writer:
                for sheet_name, df in extracted_data.items():
                    df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            
            # Provide download
            with open(excel_path, 'rb') as f:
                excel_data = f.read()
            
            st.download_button(
                "Download Excel",
                excel_data,
                file_name=f"{uploaded_file.name.replace('.pdf', '')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            # Show preview
            st.subheader("Preview")
            for name, df in list(extracted_data.items())[:3]:  # Show first 3
                st.write(f"**{name}** (Rows: {len(df)}, Columns: {len(df.columns)})")
                st.dataframe(df.head(10))
        else:
            st.error("Could not extract any data from PDF")
    
    except Exception as e:
        st.error(f"Error: {e}")
    
    finally:
        # Cleanup
        try:
            os.unlink(pdf_path)
            if 'excel_path' in locals():
                os.unlink(excel_path)
        except:
            pass
