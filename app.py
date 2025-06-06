# app.py - Main application entry point

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import components
from chat import chat_interface
from browser import render_model_browser

# ---- MAIN APP ----
st.set_page_config(page_title="Multi-Model AI Chatbot", layout="wide", page_icon="🧠")

# Main navigation
tab1, tab2 = st.tabs(["💬 Chat", "🔍 Model Browser"])

with tab1:
    # Main chat interface
    chat_interface()

with tab2:
    # Model browser interface
    render_model_browser()
