#!/bin/bash
# Simple script to run Streamlit UI
cd /root/projects/test_task
source venv/bin/activate
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --browser.gatherUsageStats false
