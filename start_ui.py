#!/usr/bin/env python3
"""
Startup script for the Streamlit RAG UI.
Ensures the backend API is running before starting the UI.
"""
import subprocess
import time
import requests
import sys
import os

def check_api_health(max_retries=30, delay=2):
    """Check if the API is healthy."""
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    print("✅ API is healthy!")
                    return True
        except requests.exceptions.RequestException:
            pass
        
        print(f"⏳ Waiting for API... ({i+1}/{max_retries})")
        time.sleep(delay)
    
    return False

def start_api():
    """Start the API server if not running."""
    print("🚀 Starting API server...")
    
    # Change to project directory
    os.chdir("/root/projects/test_task")
    
    # Start the API in background
    subprocess.Popen([
        "/root/projects/test_task/venv/bin/python", "start_api.py"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Wait for API to be ready
    if check_api_health():
        return True
    else:
        print("❌ Failed to start API server")
        return False

def start_streamlit():
    """Start the Streamlit app."""
    print("🎨 Starting Streamlit UI...")
    
    # Start Streamlit with virtual environment
    subprocess.run([
        "/root/projects/test_task/venv/bin/streamlit", "run", "streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--browser.gatherUsageStats", "false",
        "--theme.primaryColor", "#667eea",
        "--theme.backgroundColor", "#ffffff",
        "--theme.secondaryBackgroundColor", "#f0f2f6"
    ])

def main():
    """Main function."""
    print("🤖 Starting RAG System UI...")
    
    # Check if API is already running
    if not check_api_health(max_retries=3, delay=1):
        # Start API if not running
        if not start_api():
            print("❌ Cannot start UI without API server")
            sys.exit(1)
    
    # Start Streamlit
    start_streamlit()

if __name__ == "__main__":
    main()
