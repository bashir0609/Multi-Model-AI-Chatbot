# browser.py - Guide users to find their own working models

import streamlit as st
from models import MODEL_OPTIONS

def render_model_browser():
    """Guide users to find working models themselves."""
    st.header("🔍 Model Browser - Find Working Models")
    
    # Big warning
    st.error("⚠️ **Model availability changes frequently!** Follow the guide below to find current working models.")
    
    # Step-by-step guide
    st.subheader("📋 Step-by-Step Guide")
    
    with st.container():
        st.markdown("### 🔗 Step 1: Open OpenRouter")
        st.markdown("Go to **[OpenRouter Models](https://openrouter.ai/models)** in a new tab")
        
        st.markdown("### 🔍 Step 2: Filter for Free Models")
        st.markdown("""
        1. Look for **"Prompt pricing"** filter on the left
        2. Set it to **"FREE"** or drag slider to 0
        3. You'll see only free models
        """)
        
        st.markdown("### 📋 Step 3: Copy Model IDs")
        st.markdown("""
        1. Find models that say **":free"** at the end
        2. Copy the **exact model ID** (e.g., `meta-llama/llama-3.1-8b-instruct:free`)
        3. Common patterns:
           - `provider/model-name:free`
           - `deepseek/deepseek-chat:free`
           - `meta-llama/llama-3.x-xxxb-instruct:free`
        """)
        
        st.markdown("### ✅ Step 4: Test the Model")
        st.markdown("""
        1. Go back to the **Chat** tab
        2. Add the model ID to the selection
        3. Use **"Test API Connection"** button
        4. If it works ✅, you're good to go!
        5. If it fails ❌, try another model
        """)
    
    # Manual model input
    st.subheader("🔧 Manual Model Input")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        manual_model = st.text_input(
            "Enter a model ID you found on OpenRouter:",
            placeholder="e.g., meta-llama/llama-3.1-8b-instruct:free",
            help="Copy the exact model ID from OpenRouter"
        )
    
    with col2:
        if st.button("Add Model", type="primary"):
            if manual_model:
                if 'selected_models_browser' not in st.session_state:
                    st.session_state.selected_models_browser = []
                if manual_model not in st.session_state.selected_models_browser:
                    st.session_state.selected_models_browser.append(manual_model)
                    st.success(f"✅ Added: {manual_model}")
                else:
                    st.info("Already added!")
            else:
                st.error("Please enter a model ID")
    
    # Current basic models
    st.subheader("📦 Basic Models (May Need Credits)")
    st.info("These are conservative choices that should exist, but may require paid credits:")
    
    for model_id, display_name in MODEL_OPTIONS.items():
        col1, col2, col3 = st.columns([4, 1, 1])
        with col1:
            st.write(f"**{display_name}**")
            st.caption(f"`{model_id}`")
        with col2:
            if ':free' in model_id:
                st.success("FREE?")
            else:
                st.warning("PAID")
        with col3:
            if st.button(f"Add", key=f"add_{hash(model_id)}"):
                if 'selected_models_browser' not in st.session_state:
                    st.session_state.selected_models_browser = []
                if model_id not in st.session_state.selected_models_browser:
                    st.session_state.selected_models_browser.append(model_id)
                    st.success("Added!")
                else:
                    st.info("Already added!")
    
    # Selected models
    if 'selected_models_browser' in st.session_state and st.session_state.selected_models_browser:
        st.subheader("✅ Selected Models")
        
        for model_id in st.session_state.selected_models_browser:
            st.write(f"🔹 `{model_id}`")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Use in Chat", type="primary"):
                st.session_state.transfer_models = st.session_state.selected_models_browser
                st.success("✅ Models transferred! Go to Chat tab.")
        
        with col2:
            if st.button("🗑️ Clear All"):
                st.session_state.selected_models_browser = []
                st.rerun()
    
    # Links
    st.subheader("🔗 Useful Links")
    st.markdown("""
    - [OpenRouter Models](https://openrouter.ai/models) - Find current models
    - [OpenRouter API Keys](https://openrouter.ai/keys) - Get your API key  
    - [OpenRouter Docs](https://openrouter.ai/docs) - Documentation
    """)
    
    # Tips
    with st.expander("💡 Pro Tips", expanded=False):
        st.markdown("""
        **Finding Free Models:**
        - Look for models ending in `:free`
        - DeepSeek often has free models
        - Meta Llama sometimes has free versions
        - Google Gemini may have free experimental models
        
        **If Free Models Don't Work:**
        - Buy $5-10 credits on OpenRouter
        - Use ultra-cheap models (under $0.10/1M tokens)
        - Many good models cost only a few cents per conversation
        
        **Model Naming Patterns:**
        - `deepseek/deepseek-chat:free`
        - `meta-llama/llama-3.1-8b-instruct:free`
        - `google/gemini-2.0-flash-exp:free`
        - `mistralai/mistral-7b-instruct:free`
        """)
    
    st.warning("🔄 **Remember**: Model availability changes daily. Always check OpenRouter for the latest free models!")
