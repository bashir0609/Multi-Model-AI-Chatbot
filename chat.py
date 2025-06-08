# chat.py - Simple chat interface (no caching, no duplicate keys)

import os
import streamlit as st
from models import MODEL_OPTIONS, get_cost_info
from api_utils import validate_api_key, call_model_api

def chat_interface():
    """Main chat interface - simple version"""
    st.title("🧠 Multi-Model AI Chatbot (OpenRouter)")
    
    st.warning("⚠️ **IMPORTANT**: Many free models change frequently. If you get 'model not found' errors:")
    with st.expander("🔧 How to find working models", expanded=False):
        st.markdown("""
        1. Go to [OpenRouter Models](https://openrouter.ai/models) in another tab
        2. Use the filter → Set "Prompt pricing" to "FREE" 
        3. Copy the exact model ID (like `provider/model-name:free`)
        4. Add it manually in the model selection below
        5. Test it with the connection test button
        """)

    # Check for transferred models from browser
    if 'transfer_models' in st.session_state:
        st.success(f"✅ Model transferred from browser!")
        default_model = st.session_state.transfer_models[0]
        del st.session_state.transfer_models
    else:
        default_model = "meta-llama/llama-3.1-8b-instruct"

    # CSS
    st.markdown("""
    <style>
        .model-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            text-align: center;
            font-weight: bold;
            font-size: 1.1em;
        }
    </style>
    """, unsafe_allow_html=True)

    # SIDEBAR
    with st.sidebar:
        st.header("🔐 API Access")
        st.info("⚠️ Session storage only - API key lost on refresh")
        
        # Initialize API key session state
        if 'api_key' not in st.session_state:
            st.session_state.api_key = None
        
        # Check if we have an API key
        if st.session_state.api_key:
            # Show current API key
            masked_key = st.session_state.api_key[:8] + "..." + st.session_state.api_key[-4:] if len(st.session_state.api_key) > 12 else "sk-..."
            st.success(f"✅ API Key Active: `{masked_key}`")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear", key="clear_api_key_btn"):
                    st.session_state.api_key = None
                    st.rerun()
            with col2:
                if st.button("🔄 Change", key="change_api_key_btn"):
                    st.session_state.api_key = None
                    st.rerun()
            
            current_api_key = st.session_state.api_key
            
        else:
            # No API key - show input
            st.info("🔑 No API key active. Please enter below:")
            
            # Check environment
            env_api_key = os.getenv("OPENROUTER_API_KEY")
            
            source = st.radio(
                "API Key Source:",
                ["Environment Variable", "Manual Input"],
                key="api_source_radio"
            )
            
            if source == "Environment Variable":
                if env_api_key:
                    env_api_key = env_api_key.strip()
                    is_valid, message = validate_api_key(env_api_key)
                    
                    if is_valid:
                        masked_key = env_api_key[:8] + "..." + env_api_key[-4:]
                        st.info(f"🔍 Environment key: `{masked_key}`")
                        
                        if st.button("✅ Use Environment Key", key="use_env_key_btn"):
                            st.session_state.api_key = env_api_key
                            st.success("✅ Environment key loaded!")
                            st.rerun()
                    else:
                        st.error(f"❌ Environment key issue: {message}")
                else:
                    st.warning("⚠️ No OPENROUTER_API_KEY found in environment")
                    st.info("💡 Create a `.env` file or switch to Manual Input")
            
            else:  # Manual Input
                manual_key = st.text_input(
                    "API Key:",
                    type="password",
                    placeholder="sk-or-v1-...",
                    key="manual_api_input"
                )
                
                if manual_key:
                    manual_key = manual_key.strip()
                    is_valid, message = validate_api_key(manual_key)
                    if is_valid:
                        masked_key = manual_key[:8] + "..." + manual_key[-4:]
                        st.info(f"✅ Key validated: `{masked_key}`")
                        
                        if st.button("✅ Use This Key", key="use_manual_key_btn"):
                            st.session_state.api_key = manual_key
                            st.success("✅ API key loaded!")
                            st.rerun()
                    else:
                        st.error(f"❌ {message}")
            
            current_api_key = None
        
        # Stop if no API key
        if not current_api_key:
            st.error("❌ No API key available. Please configure above.")
            st.stop()

        st.divider()
        st.header("🔧 Connection Test")
        
        if st.button("🧪 Test Connection", key="test_conn_btn"):
            with st.spinner("Testing..."):
                test_response = call_model_api(
                    "meta-llama/llama-3.1-8b-instruct",
                    [{"role": "user", "content": "Hi"}],
                    current_api_key,
                    0.1, 10, 30, ""
                )
                if test_response.startswith("🔐") or test_response.startswith("❌"):
                    st.error("❌ Connection failed")
                    st.error(test_response)
                else:
                    st.success("✅ Connection successful!")

        st.divider()
        st.header("🤖 Model Selection")
        
        # Initialize model session state
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = default_model
        if 'custom_models' not in st.session_state:
            st.session_state.custom_models = {}
        
        # Combine models
        all_models = MODEL_OPTIONS.copy()
        all_models.update(st.session_state.custom_models)
        
        # Model selection
        selected_model = st.selectbox(
            "Choose a model:",
            options=list(all_models.keys()),
            format_func=lambda x: all_models[x],
            index=list(all_models.keys()).index(st.session_state.selected_model) if st.session_state.selected_model in all_models else 0,
            key="model_selector"
        )
        
        # Update when selection changes
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model

        # Show model info
        if selected_model in MODEL_OPTIONS:
            cost_info = get_cost_info(selected_model, MODEL_OPTIONS[selected_model])
            if cost_info['type'] == 'free':
                st.success(f"🆓 {cost_info['cost']}")
            else:
                st.info(f"💰 {cost_info['cost']}")

        # Add custom model
        st.subheader("➕ Add Custom Model")
        
        custom_model = st.text_input(
            "Model ID:",
            placeholder="e.g., deepseek/deepseek-chat:free",
            key="custom_model_input"
        )
        
        if st.button("➕ Add", key="add_custom_model_btn"):
            if custom_model and custom_model.strip():
                custom_id = custom_model.strip()
                if custom_id not in all_models:
                    if ':free' in custom_id:
                        display_name = f"{custom_id.replace(':free', '')} (FREE - Custom)"
                    else:
                        display_name = f"{custom_id} (Custom)"
                    
                    st.session_state.custom_models[custom_id] = display_name
                    st.session_state.selected_model = custom_id
                    st.success(f"✅ Added: {custom_id}")
                    st.rerun()
                else:
                    st.session_state.selected_model = custom_id
                    st.info(f"✅ Switched to: {custom_id}")
                    st.rerun()

        # Quick select
        st.subheader("⚡ Quick Select")
        quick_models = [
            ("🦙 Llama Basic", "meta-llama/llama-3.1-8b-instruct"),
            ("🧠 DeepSeek", "deepseek/deepseek-chat"),
            ("🆓 Free Llama", "meta-llama/llama-3.1-8b-instruct:free"),
        ]
        
        for name, model_id in quick_models:
            if st.button(name, key=f"quick_{model_id}", use_container_width=True):
                st.session_state.selected_model = model_id
                st.rerun()

        st.divider()
        st.header("💬 Chat Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", key="clear_chat_btn"):
                if 'chat_history' in st.session_state:
                    st.session_state.chat_history = []
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear All", key="clear_all_btn"):
                if 'chat_history' in st.session_state:
                    st.session_state.chat_history = []
                if 'api_key' in st.session_state:
                    st.session_state.api_key = None
                st.rerun()

        st.divider()
        with st.expander("⚙️ Settings"):
            temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
            max_tokens = st.slider("Max tokens", 16, 2048, 512, 16)
            timeout = st.slider("Timeout (seconds)", 10, 120, 60, 5)

    # Main chat area
    if not selected_model:
        st.warning("Please select a model.")
        st.stop()

    # SESSION STATE FOR CHAT
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # DISPLAY CURRENT MODEL
    st.markdown(f'<div class="model-header">💬 Chatting with: {MODEL_OPTIONS.get(selected_model, selected_model)}</div>', unsafe_allow_html=True)

    # DISPLAY CHAT HISTORY
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.chat_message("user").markdown(msg["content"])
            else:
                st.chat_message("assistant").markdown(msg["content"])

    # CHAT INPUT
    user_input = st.chat_input("Type your message...")

    if user_input:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Show user message
        with chat_container:
            st.chat_message("user").markdown(user_input)
        
        # Get AI response
        with st.spinner(f"🤖 {MODEL_OPTIONS.get(selected_model, selected_model)} is thinking..."):
            response = call_model_api(
                selected_model,
                st.session_state.chat_history,
                current_api_key,
                temperature,
                max_tokens,
                timeout,
                ""  # No system message for simplicity
            )
        
        # Add and show AI response
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        with chat_container:
            st.chat_message("assistant").markdown(response)
        
        st.rerun()

    # FOOTER
    if st.session_state.chat_history:
        with st.expander("📊 Session Info"):
            total_messages = len(st.session_state.chat_history)
            user_messages = len([msg for msg in st.session_state.chat_history if msg["role"] == "user"])
            assistant_messages = len([msg for msg in st.session_state.chat_history if msg["role"] == "assistant"])
            
            st.write(f"**Current model:** {MODEL_OPTIONS.get(selected_model, selected_model)}")
            st.write(f"**Total messages:** {total_messages}")
            st.write(f"**Your messages:** {user_messages}")
            st.write(f"**AI responses:** {assistant_messages}")
