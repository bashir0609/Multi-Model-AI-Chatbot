# app.py - Main application entry point

# ⚠️ CRITICAL: Import streamlit first and set page config immediately
import streamlit as st
st.set_page_config(page_title="Multi-Model AI Chatbot", layout="wide", page_icon="🧠")

# Now safe to import everything else
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import components after page config is set
try:
    from chat import chat_interface
    from browser import render_model_browser
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Make sure all required files (chat.py, browser.py, models.py, api_utils.py) are in the same directory as app.py")
    st.stop()

# ---- MAIN APP ----
# Main navigation
tab1, tab2 = st.tabs(["💬 Chat", "🔍 Model Browser"])

try:
    with tab1:
        # Main chat interface
        chat_interface()

    with tab2:
        # Model browser interface
        render_model_browser()
except Exception as e:
    st.error(f"Application error: {e}")
    st.error("Please check the console for detailed error information.")
