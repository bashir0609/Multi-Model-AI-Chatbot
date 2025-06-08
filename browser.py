# browser.py - Fixed Model Browser (No API key required)

import streamlit as st

# Current working models
CURRENT_FREE_MODELS = {
    "deepseek/deepseek-chat:free": "DeepSeek Chat (FREE)",
    "deepseek/deepseek-r1:free": "DeepSeek R1 Reasoning (FREE)",
    "deepseek/deepseek-coder:free": "DeepSeek Coder (FREE)",
    "meta-llama/llama-3.1-8b-instruct:free": "Llama 3.1 8B Instruct (FREE)",
    "meta-llama/llama-3.1-70b-instruct:free": "Llama 3.1 70B Instruct (FREE)",
    "meta-llama/llama-3.2-1b-instruct:free": "Llama 3.2 1B Instruct (FREE)",
    "meta-llama/llama-3.2-3b-instruct:free": "Llama 3.2 3B Instruct (FREE)",
    "meta-llama/llama-3.2-11b-vision-instruct:free": "Llama 3.2 11B Vision (FREE)",
    "meta-llama/llama-3.3-70b-instruct:free": "Llama 3.3 70B Instruct (FREE)",
    "google/gemini-2.0-flash-experimental:free": "Gemini 2.0 Flash Experimental (FREE)",
    "google/gemini-pro:free": "Gemini Pro (FREE)",
    "mistralai/mistral-7b-instruct:free": "Mistral 7B Instruct (FREE)",
    "qwen/qwen2.5-72b-instruct:free": "Qwen 2.5 72B Instruct (FREE)",
    "qwen/qwen2.5-coder-32b-instruct:free": "Qwen 2.5 Coder 32B (FREE)",
}

ULTRA_CHEAP_MODELS = {
    "meta-llama/llama-3.2-3b-instruct": "Llama 3.2 3B ($0.02/1M)",
    "deepseek/deepseek-chat": "DeepSeek Chat ($0.02/1M)",
    "meta-llama/llama-3.1-8b-instruct": "Llama 3.1 8B ($0.05/1M)",
    "mistralai/mistral-7b-instruct": "Mistral 7B ($0.07/1M)",
    "qwen/qwen2.5-7b-instruct": "Qwen 2.5 7B ($0.07/1M)",
}

def render_model_browser():
    """Model browser - NO DUPLICATE KEYS"""
    st.title("🔍 Model Browser - Find Working Models")
    
    st.info("💡 **Browse and select models to use in chat. No API key required for browsing!**")
    st.warning("⚠️ **IMPORTANT**: Free model availability changes frequently. Always test models in the Chat tab!")
    
    # Model selection state
    if 'browser_selected_models' not in st.session_state:
        st.session_state.browser_selected_models = []
    
    # Quick access links (no buttons, just HTML)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <a href="https://openrouter.ai/models" target="_blank" style="text-decoration: none;">
            <div style="background: linear-gradient(45deg, #667eea, #764ba2); color: white; padding: 10px; border-radius: 8px; text-align: center; margin-bottom: 10px;">
                🔍 Check OpenRouter Live
            </div>
        </a>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <a href="https://openrouter.ai/models?pricing=free" target="_blank" style="text-decoration: none;">
            <div style="background: linear-gradient(45deg, #28a745, #20c997); color: white; padding: 10px; border-radius: 8px; text-align: center; margin-bottom: 10px;">
                🆓 Free Models Only
            </div>
        </a>
        """, unsafe_allow_html=True)
    
    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🆓 Free Models", len(CURRENT_FREE_MODELS))
    with col2:
        st.metric("💰 Ultra-Cheap", len(ULTRA_CHEAP_MODELS))
    with col3:
        st.metric("📋 Selected", len(st.session_state.browser_selected_models))
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🆓 Free Models", "💰 Ultra-Cheap", "📋 Selected"])
    
    with tab1:
        st.subheader("🆓 Free Models")
        
        # Just show the models - NO BUTTONS AT ALL
        for model_id, display_name in CURRENT_FREE_MODELS.items():
            st.write(f"**{display_name}**")
            st.caption(f"`{model_id}`")
            st.success("FREE")
            st.divider()
        
        # Single selectbox to add models
        st.subheader("Add a Free Model")
        selected_free = st.selectbox(
            "Choose a free model to add:",
            options=list(CURRENT_FREE_MODELS.keys()),
            format_func=lambda x: CURRENT_FREE_MODELS[x],
            key="free_model_select"
        )
        
        # Single button to add
        if st.button("➕ Add This Free Model", key="add_free_btn"):
            if selected_free not in st.session_state.browser_selected_models:
                st.session_state.browser_selected_models.append(selected_free)
                st.success("Added!")
                st.rerun()
    
    with tab2:
        st.subheader("💰 Ultra-Cheap Models")
        
        # Just show the models - NO BUTTONS AT ALL
        for model_id, display_name in ULTRA_CHEAP_MODELS.items():
            st.write(f"**{display_name}**")
            st.caption(f"`{model_id}`")
            price = display_name.split("($")[1].split(")")[0] if "($" in display_name else "Cheap"
            st.info(price)
            st.divider()
        
        # Single selectbox to add models
        st.subheader("Add a Cheap Model")
        selected_cheap = st.selectbox(
            "Choose a cheap model to add:",
            options=list(ULTRA_CHEAP_MODELS.keys()),
            format_func=lambda x: ULTRA_CHEAP_MODELS[x],
            key="cheap_model_select"
        )
        
        # Single button to add
        if st.button("➕ Add This Cheap Model", key="add_cheap_btn"):
            if selected_cheap not in st.session_state.browser_selected_models:
                st.session_state.browser_selected_models.append(selected_cheap)
                st.success("Added!")
                st.rerun()
    
    with tab3:
        st.subheader("📋 Selected Models")
        
        if st.session_state.browser_selected_models:
            st.success(f"✅ You have {len(st.session_state.browser_selected_models)} models selected")
            
            # Show selected models - NO REMOVE BUTTONS
            for model_id in st.session_state.browser_selected_models:
                display_name = CURRENT_FREE_MODELS.get(model_id) or ULTRA_CHEAP_MODELS.get(model_id) or model_id
                st.write(f"**{display_name}**")
                st.caption(f"`{model_id}`")
                if ':free' in model_id:
                    st.success("FREE")
                else:
                    st.info("Paid")
                st.divider()
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🚀 Send First Model to Chat", key="send_to_chat_btn"):
                    if st.session_state.browser_selected_models:
                        st.session_state.transfer_models = [st.session_state.browser_selected_models[0]]
                        st.balloons()
                        st.success(f"✅ Sent: {st.session_state.browser_selected_models[0]}")
            
            with col2:
                if st.button("📋 Show All IDs", key="show_ids_btn"):
                    model_list = "\n".join(st.session_state.browser_selected_models)
                    st.text_area("Model IDs:", model_list, height=100, key="ids_display")
            
            with col3:
                if st.button("🗑️ Clear All Selected", key="clear_all_btn"):
                    st.session_state.browser_selected_models = []
                    st.success("Cleared!")
                    st.rerun()
            
            # Remove individual model
            if len(st.session_state.browser_selected_models) > 1:
                st.subheader("Remove Individual Model")
                to_remove = st.selectbox(
                    "Select model to remove:",
                    options=st.session_state.browser_selected_models,
                    key="remove_select"
                )
                if st.button("🗑️ Remove Selected Model", key="remove_btn"):
                    st.session_state.browser_selected_models.remove(to_remove)
                    st.success("Removed!")
                    st.rerun()
        else:
            st.info("No models selected yet. Use the tabs above to add models.")
    
    # Manual input
    st.divider()
    st.subheader("➕ Add Any Model Manually")
    
    custom_model = st.text_input(
        "Enter model ID:",
        placeholder="e.g., anthropic/claude-3-sonnet",
        key="custom_input"
    )
    
    if st.button("➕ Add Custom Model", key="add_custom_btn"):
        if custom_model and custom_model.strip():
            model_id = custom_model.strip()
            if model_id not in st.session_state.browser_selected_models:
                st.session_state.browser_selected_models.append(model_id)
                st.success(f"Added: {model_id}")
                st.rerun()
            else:
                st.info("Already added!")
        else:
            st.error("Please enter a model ID")
    
    # Instructions
    st.divider()
    with st.expander("📖 How to Use"):
        st.markdown("""
        1. Browse models in the tabs above
        2. Select models and click "Add" 
        3. Go to "Selected" tab to manage your list
        4. Click "Send to Chat" to use a model
        5. Switch to Chat tab to start chatting
        """)
