# chat.py - Updated with Gemini-style Chat Input

import os
import streamlit as st
# Make sure to import the new function from api_utils
from api_utils import validate_api_key, call_model_api, get_available_models

def chat_interface():
    """Main chat interface with a clear, dynamic model fetching workflow."""
    st.title("🧠 Multi-Model AI Chatbot (OpenRouter)")

    # --- Custom CSS for Gemini-style Input Box ---
    st.markdown("""
    <style>
    /* Main container for the chat input */
    .stChatInputContainer {
        padding: 1rem;
        background-color: #f0f4f9; /* A light grey background */
        border-radius: 2rem; /* Fully rounded corners */
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    /* The actual text input area */
    .st-emotion-cache-134p1ai textarea {
        background-color: transparent;
        border: none;
        font-size: 1rem;
    }
    
    /* The send button */
    .st-emotion-cache-15zrgzn {
        background-color: #1a73e8; /* A nice blue color */
        border-radius: 50%; /* Make it a circle */
        color: white;
    }
    .st-emotion-cache-15zrgzn:hover {
        background-color: #185abc; /* Darker blue on hover */
        color: white;
    }
    
    /* You may need to adjust these selectors if they change in future Streamlit versions */
    /* To inspect elements, right-click on them in the browser and choose 'Inspect' */

    </style>
    """, unsafe_allow_html=True)


    # --- Initialize Session State ---
    # This ensures all necessary keys exist in the session
    if 'api_key' not in st.session_state:
        st.session_state.api_key = None
    if 'available_models' not in st.session_state:
        st.session_state.available_models = {} # This will store dynamically fetched models
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("🔐 API Access")

        # --- API Key Section ---
        # This section handles entering, validating, and clearing the API key
        if st.session_state.api_key:
            masked_key = st.session_state.api_key[:8] + "..." + st.session_state.api_key[-4:]
            st.success(f"API Key Active: `{masked_key}`")
            if st.button("🗑️ Change API Key"):
                st.session_state.api_key = None
                st.session_state.available_models = {}  # Clear models if key changes
                st.session_state.selected_model = None
                st.rerun()
            current_api_key = st.session_state.api_key
        else:
            st.info("Enter your OpenRouter API key to begin.")
            manual_key = st.text_input("OpenRouter API Key:", type="password", placeholder="sk-or-...")
            if st.button("✅ Save Key"):
                is_valid, message = validate_api_key(manual_key)
                if is_valid:
                    st.session_state.api_key = manual_key.strip()
                    st.success("API key saved! Fetching models...")
                    
                    # --- Automatically fetch models right after saving the key ---
                    with st.spinner("Fetching available models..."):
                        models, message = get_available_models(st.session_state.api_key)
                        if models:
                            st.session_state.available_models = models
                            st.session_state.selected_model = next(iter(models.keys()), None)
                            st.success(f"✅ Loaded {len(models)} models!")
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
                            st.rerun()
                else:
                    st.error(f"❌ {message}")
            current_api_key = None

        # --- Model Management Section ---
        # This section ONLY appears after an API key is provided
        if current_api_key:
            st.divider()
            st.header("🤖 Model Management")

            # **THE FETCHING OPTION**
            # This button will now appear right after the key is saved.
            if st.button("🔄 Fetch Available Models"):
                with st.spinner("Fetching models from OpenRouter..."):
                    models, message = get_available_models(current_api_key)
                    if models:
                        st.session_state.available_models = models
                        # Set a default model after fetching
                        st.session_state.selected_model = next(iter(models.keys()), None)
                        st.success(f"✅ Loaded {len(models)} models!")
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")

            # --- Model Selection Dropdown ---
            # This dropdown ONLY appears if models have been successfully fetched
            if st.session_state.available_models:
                all_models = st.session_state.available_models
                
                # The selectbox for choosing a model among the fetched ones
                selected_model = st.selectbox(
                    "Choose a model:",
                    options=list(all_models.keys()),
                    format_func=lambda x: all_models.get(x, x),
                    # Use the index of the currently selected model
                    index=list(all_models.keys()).index(st.session_state.selected_model) if st.session_state.selected_model in all_models else 0
                )
                # Update the selected model in session state if it changes
                if selected_model != st.session_state.selected_model:
                    st.session_state.selected_model = selected_model
                    st.rerun()
            else:
                 # Instruct the user on the next step
                 st.info("Click the 'Fetch Available Models' button above to load the model list.")
        
        # --- Settings Section ---
        st.divider()
        st.header("⚙️ Settings")
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
        max_tokens = st.slider("Max tokens", 16, 8192, 1024, 16)
        timeout = st.slider("Timeout (seconds)", 10, 120, 60, 5)

        st.divider()
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    # --- MAIN CHAT AREA ---
    # The chat area will only be fully active if an API key is present and a model is selected
    if not current_api_key:
        st.info("⬅️ Please enter your API key in the sidebar to start.")
        st.stop()

    if not st.session_state.selected_model:
        st.info("⬅️ Please fetch and select a model in the sidebar to begin chatting.")
        st.stop()

    # Display the current model name
    model_display_name = st.session_state.available_models.get(st.session_state.selected_model, "Unknown Model")
    st.markdown(f"### 💬 Chatting with: **{model_display_name}**")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input box - This will now be styled by the CSS above
    if user_input := st.chat_input("Type your message..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner(f"🤖 {model_display_name} is thinking..."):
                response = call_model_api(
                    st.session_state.selected_model,
                    st.session_state.chat_history,
                    current_api_key,
                    temperature,
                    max_tokens,
                    timeout
                )
                message_placeholder.markdown(response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
