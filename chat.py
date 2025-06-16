# chat.py - Corrected Dynamic Chat Interface

import os
import streamlit as st
from api_utils import validate_api_key, call_model_api, get_available_models

def chat_interface():
    """Main chat interface with dynamic model fetching"""
    st.title("🧠 Multi-Model AI Chatbot (OpenRouter)")

    # Initialize session state for API key and models
    if 'api_key' not in st.session_state:
        st.session_state.api_key = None
    if 'available_models' not in st.session_state:
        st.session_state.available_models = {}

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("🔐 API Access")
        
        # API Key Input and Validation
        if st.session_state.api_key:
            masked_key = st.session_state.api_key[:8] + "..." + st.session_state.api_key[-4:]
            st.success(f"✅ API Key Active: `{masked_key}`")
            if st.button("🗑️ Clear API Key", key="clear_api_key"):
                st.session_state.api_key = None
                st.session_state.available_models = {} # Clear models when key is removed
                st.rerun()
            current_api_key = st.session_state.api_key
        else:
            st.info("Enter your OpenRouter API key to begin.")
            manual_key = st.text_input("API Key:", type="password", placeholder="sk-or-...", key="manual_api_input")
            if st.button("✅ Save Key", key="save_manual_key"):
                is_valid, message = validate_api_key(manual_key)
                if is_valid:
                    st.session_state.api_key = manual_key.strip()
                    st.success("API key saved!")
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
            current_api_key = None

        # This block should be visible only when there is an API key
        if current_api_key:
            st.divider()
            st.header("🤖 Model Management")

            # Button to fetch models - THIS IS THE FIX
            if st.button("🔄 Fetch/Refresh Models", key="fetch_models_btn"):
                with st.spinner("Fetching available models from OpenRouter..."):
                    models, message = get_available_models(current_api_key)
                    if models:
                        st.session_state.available_models = models
                        st.success(f"✅ Loaded {len(models)} models!")
                    else:
                        st.error(f"❌ {message}")
            
            # Model Selection Dropdown
            if st.session_state.available_models:
                all_models = st.session_state.available_models
                default_model = st.session_state.get('selected_model')
                
                if not default_model or default_model not in all_models:
                    default_model = next(iter(all_models), None)
                
                selected_model = st.selectbox(
                    "Choose a model:",
                    options=list(all_models.keys()),
                    format_func=lambda x: all_models.get(x, x),
                    index=list(all_models.keys()).index(default_model) if default_model else 0,
                    key="model_selector"
                )
                st.session_state.selected_model = selected_model
            else:
                st.info("Click 'Fetch/Refresh Models' to load the model list.")
        
        if not current_api_key:
            st.error("Please provide a valid API key to use the chatbot.")
            st.stop()
            
        st.divider()
        st.header("⚙️ Settings")
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
        max_tokens = st.slider("Max tokens", 16, 8192, 1024, 16)
        timeout = st.slider("Timeout (seconds)", 10, 120, 60, 5)

        st.divider()
        if st.button("🗑️ Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    # --- MAIN CHAT AREA ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if 'selected_model' in st.session_state and st.session_state.available_models:
        model_display_name = st.session_state.available_models.get(st.session_state.selected_model, st.session_state.selected_model)
        st.markdown(f"### 💬 Chatting with: **{model_display_name}**")

        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if user_input := st.chat_input("What would you like to discuss?"):
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner(f"🤖 {model_display_name} is thinking..."):
                    api_messages = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_history
                    ]
                    
                    response = call_model_api(
                        st.session_state.selected_model,
                        api_messages,
                        current_api_key,
                        temperature,
                        max_tokens,
                        timeout
                    )
                    message_placeholder.markdown(response)
            
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
    elif not st.session_state.available_models:
         st.info("Please fetch the models using the sidebar to begin chatting.")
