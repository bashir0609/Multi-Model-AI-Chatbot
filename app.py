# app.py - Main application entry point

import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Import components (these imports need to be after load_dotenv)
try:
    from chat import chat_interface
    from browser import render_model_browser
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Make sure all required files (chat.py, browser.py, models.py, api_utils.py) are in the same directory as app.py")
    st.stop()

# ---- MAIN APP ----
def main():
    st.set_page_config(page_title="Multi-Model AI Chatbot", layout="wide", page_icon="🧠")

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

if __name__ == "__main__":
    main()# app.py - Main application entry point

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
