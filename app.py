# app.py - Main application entry point

import streamlit as st
st.set_page_config(page_title="Multi-Model AI Chatbot", layout="wide", page_icon="🧠")

import os
from dotenv import load_dotenv
load_dotenv()

# Import components
try:
    from chat import chat_interface
    from browser import render_model_browser
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Main navigation
tab1, tab2 = st.tabs(["💬 Chat", "🔍 Model Browser"])

try:
    with tab1:
        chat_interface()
    with tab2:
        render_model_browser()
except Exception as e:
    st.error(f"Application error: {e}")
