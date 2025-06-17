#!/usr/bin/env python3
"""
Test script to verify all required packages are installed correctly
"""

def test_imports():
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
        print(f"   Version: {st.__version__}")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
    
    try:
        import google.generativeai as genai
        print("✅ Google GenerativeAI imported successfully")
    except ImportError as e:
        print(f"❌ Google GenerativeAI import failed: {e}")
    
    try:
        import fitz  # PyMuPDF
        print("✅ PyMuPDF imported successfully")
        print(f"   Version: {fitz.version}")
    except ImportError as e:
        print(f"❌ PyMuPDF import failed: {e}")
    
    try:
        from dotenv import load_dotenv
        print("✅ python-dotenv imported successfully")
    except ImportError as e:
        print(f"❌ python-dotenv import failed: {e}")
    
    try:
        import json
        import os
        from pathlib import Path
        from datetime import datetime
        import re
        print("✅ All standard library modules imported successfully")
    except ImportError as e:
        print(f"❌ Standard library import failed: {e}")

if __name__ == "__main__":
    print("Testing PDF Chatbot Dependencies...")
    print("=" * 50)
    test_imports()
    print("=" * 50)
    print("If all imports show ✅, you're ready to run the application!")
    print("\nTo run the app:")
    print("streamlit run main.py")