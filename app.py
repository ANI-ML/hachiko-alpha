# app.py
import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path

import time
from tempfile import NamedTemporaryFile
import gc
import json
import csv
from graph import create_graph
from memory_utils import monitor_memory,  reset_run_timestamp, init_memory_log
from utils import OutputTracker
from datetime import datetime

from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager


env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Initialize Langsmith client
langsmith_client = Client()
tracer = LangChainTracer(
    project_name=os.getenv("LANGCHAIN_PROJECT"),  # You can customize this
)

def process_pdf(uploaded_files, query):
    # Create callback manager for tracing
    callback_manager = CallbackManager([tracer])
    # Reset timestamp at the start of each run
    reset_run_timestamp()

    # Save uploaded file to temporary location
    filepaths = []
    results = None
    graph = create_graph(callback_manager=callback_manager)

    for uploaded_file in uploaded_files:
        with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            filepaths.append(tmp_file.name)
    
    try:
        # Add trace metadata
        tracer.metadata = {
            "num_files": len(uploaded_files),
            "file_names": [f.name for f in uploaded_files],
            "query": query
        }
        # Initialize the workflow with necessary parameters
        results = graph.invoke({
            "query": query,
            "filepaths": filepaths,
            'top_k': None
        })
        
        return results.get('generated_summary')
    finally:
        if results is not None:
            monitor_memory("final_state", results)  # Log final memory state
        gc.collect()  # Force garbage collection

        # Clean up large objects
        results = None
        gc.collect()  # Final garbage collection

        # One last memory check after cleanup
        monitor_memory("after_cleanup")
        # Clean up temporary files
        for filepath in filepaths:
            os.unlink(filepath)      

def main():

    # Initialize components
    init_memory_log()
    tracker = OutputTracker()
    st.image("./company_logo.webp", width=300)
    # Initialize session state
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'has_generated' not in st.session_state:
        st.session_state.has_generated = False
    
        # Add sidebar with info
    with st.sidebar:
        st.image("./hachiko_logo.webp", width=600)  
        st.markdown("### About Hachiko")
        st.info("""
        Hachiko is an AI assistant that helps veterinarians by generating 
        structured summaries from medical records. Upload your PDFs to get started.
        """)
        
        st.markdown("### Document Guidelines")
        st.warning("""
        - PDF files only
        - All pages of pdf are oriented properly
        - Maximum 10 files at once
        - Each file < 10MB
        """)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("Hachiko Alpha v1.2")
        
    with col2:
        st.markdown("")  
        st.markdown("") 
        if st.session_state.summary:
            st.download_button(
                "Download Summary",
                st.session_state.summary,
                file_name="medical_summary.txt",
                mime="text/plain"
            )

    # File upload
    uploaded_files = st.file_uploader("Upload a patient's medical record PDFs", type=['pdf'], accept_multiple_files=True)

    # Add a timer display area
    timer_placeholder = st.empty()

    if uploaded_files:
        total_size = sum(file.size for file in uploaded_files)
        if total_size > 100 * 1024 * 1024:  # 100MB limit
            st.error("Total file size exceeds 100MB limit!")
            return
        st.success(f"📎 {len(uploaded_files)} files uploaded successfully")
    
        # Default query
        default_query = "Could you provide a detailed medical history of the patient, including the patient name, breed, all diagnoses, bloodwork, and test results?"
        query = st.text_area("Customize your query (optional)", value=default_query, height=100)
        # Show different buttons based on whether app has generated a summary
        if not st.session_state.has_generated:
            generate_button = st.button("Generate Summary", type="primary", key="generate_button")
        else:
            generate_button = st.button("Generate New Summary", type="secondary", key="regenerate_button")
    
        # Processing logic
        if generate_button:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            start_time = time.time()
            try:
                # Show processing stages
                stages = ["Extracting text", "Processing documents (~3 minutes)", "Generating summary"]
                for i, stage in enumerate(stages):
                    with st.spinner(f" {stage}..."):
                        time.sleep(2)  # Simulate processing time
                        progress_bar.progress((i + 1) * (100 // len(stages)))
                        if i == 1:  # During actual processing
                            summary = process_pdf(uploaded_files, query)
                            st.session_state.summary = summary

                progress_bar.progress(100)
                
                # Time calculation and display
                processing_time = time.time() - start_time
                minutes = int(processing_time // 60)
                seconds = int(processing_time % 60)
                
                time_display = f"{minutes} minute{'s' if minutes != 1 else ''} and {seconds} second{'s' if seconds != 1 else ''}" if minutes > 0 else f"{seconds} second{'s' if seconds != 1 else ''}"
                
                # Log run
                metrics = {
                    'processing_time': processing_time,
                    'success': True
                }

                # Log the run
                run_info = tracker.log_run(uploaded_files, summary, metrics)
                if run_info:
                    st.success(f"✨ Summary generated successfully! (Run ID: {run_info['run_id']})")

                # Display processing time
                timer_placeholder.info(f"⏱️ Processing time: {time_display}")
                
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                return
            finally:
                progress_bar.empty()
                status_text.empty()
    else:
        st.info("👆 Upload PDFs and click 'Generate Summary' to get started!")

    # Show summary 
    if st.session_state.summary:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("### Generated Summary")
        with col2:
            st.download_button(
                label="Download Summary",
                data=st.session_state.summary,
                file_name="medical_summary.txt",
                mime="text/plain",
                key="download_button"
            )
        st.markdown(st.session_state.summary)

if __name__ == "__main__":
    main()