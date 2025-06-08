# browser.py - Improved Model Browser with Current Models

import streamlit as st
from models import MODEL_OPTIONS

# Current working models based on recent OpenRouter data
CURRENT_FREE_MODELS = {
    # DeepSeek Models (Usually Reliable)
    "deepseek/deepseek-chat:free": "DeepSeek Chat (FREE)",
    "deepseek/deepseek-r1:free": "DeepSeek R1 Reasoning (FREE)",
    "deepseek/deepseek-coder:free": "DeepSeek Coder (FREE)",
    
    # Meta Llama Models (Most Reliable Free Options)
    "meta-llama/llama-3.1-8b-instruct:free": "Llama 3.1 8B Instruct (FREE)",
    "meta-llama/llama-3.1-70b-instruct:free": "Llama 3.1 70B Instruct (FREE)",
    "meta-llama/llama-3.2-1b-instruct:free": "Llama 3.2 1B Instruct (FREE)",
    "meta-llama/llama-3.2-3b-instruct:free": "Llama 3.2 3B Instruct (FREE)",
    "meta-llama/llama-3.2-11b-vision-instruct:free": "Llama 3.2 11B Vision (FREE)",
    "meta-llama/llama-3.3-70b-instruct:free": "Llama 3.3 70B Instruct (FREE)",
    "meta-llama/llama-4-maverick:free": "Llama 4 Maverick 400B MoE (FREE)",
    
    # Google Models (Experimental but Often Free)
    "google/gemini-2.0-flash-experimental:free": "Gemini 2.0 Flash Experimental (FREE)",
    "google/gemini-pro:free": "Gemini Pro (FREE)",
    
    # Mistral Models
    "mistralai/mistral-7b-instruct:free": "Mistral 7B Instruct (FREE)",
    "mistralai/mistral-small-3.1:free": "Mistral Small 3.1 (FREE)",
    
    # Qwen Models (Often Available Free)
    "qwen/qwen2.5-72b-instruct:free": "Qwen 2.5 72B Instruct (FREE)",
    "qwen/qwen2.5-coder-32b-instruct:free": "Qwen 2.5 Coder 32B (FREE)",
    
    # Other Promising Free Models
    "nvidia/llama-3.1-nemotron-8b-instruct:free": "NVIDIA Nemotron 8B (FREE)",
    "kimi/kimi-vl-a3b-thinking:free": "Kimi VL-A3B Thinking (FREE)",
}

ULTRA_CHEAP_MODELS = {
    # Under $0.10 per 1M tokens
    "meta-llama/llama-3.2-3b-instruct": "Llama 3.2 3B ($0.02/1M)",
    "mistralai/ministral-3b": "Ministral 3B ($0.04/1M)", 
    "meta-llama/llama-3.1-8b-instruct": "Llama 3.1 8B ($0.05/1M)",
    "qwen/qwen2.5-7b-instruct": "Qwen 2.5 7B ($0.07/1M)",
}

def categorize_current_models():
    """Categorize models by capability and use case"""
    categories = {
        '🧠 Reasoning & Problem Solving': [
            ("deepseek/deepseek-r1:free", "DeepSeek R1 - Advanced reasoning capabilities"),
            ("meta-llama/llama-3.3-70b-instruct:free", "Llama 3.3 70B - Best overall reasoning"),
            ("qwen/qwen2.5-72b-instruct:free", "Qwen 2.5 72B - Mathematical reasoning"),
        ],
        
        '💻 Code & Programming': [
            ("deepseek/deepseek-coder:free", "DeepSeek Coder - Specialized for coding"),
            ("qwen/qwen2.5-coder-32b-instruct:free", "Qwen 2.5 Coder - Multi-language programming"),
            ("meta-llama/llama-3.1-70b-instruct:free", "Llama 3.1 70B - General coding tasks"),
        ],
        
        '👁️ Vision & Multimodal': [
            ("meta-llama/llama-3.2-11b-vision-instruct:free", "Llama 3.2 11B Vision - Image understanding"),
            ("kimi/kimi-vl-a3b-thinking:free", "Kimi VL-A3B - Visual reasoning"),
            ("google/gemini-2.0-flash-experimental:free", "Gemini 2.0 Flash - Multimodal capabilities"),
        ],
        
        '🚀 Speed & Efficiency': [
            ("meta-llama/llama-3.2-1b-instruct:free", "Llama 3.2 1B - Fastest responses"),
            ("meta-llama/llama-3.2-3b-instruct:free", "Llama 3.2 3B - Balance of speed and quality"),
            ("deepseek/deepseek-chat:free", "DeepSeek Chat - Efficient general use"),
        ],
        
        '🦣 Large Scale & Advanced': [
            ("meta-llama/llama-4-maverick:free", "Llama 4 Maverick - 400B MoE model"),
            ("meta-llama/llama-3.3-70b-instruct:free", "Llama 3.3 70B - High capability"),
            ("qwen/qwen2.5-72b-instruct:free", "Qwen 2.5 72B - Large scale reasoning"),
        ],
        
        '🧪 Experimental & Latest': [
            ("google/gemini-2.0-flash-experimental:free", "Gemini 2.0 Flash - Google's latest"),
            ("mistralai/mistral-small-3.1:free", "Mistral Small 3.1 - Latest from Mistral"),
            ("nvidia/llama-3.1-nemotron-8b-instruct:free", "NVIDIA Nemotron - Optimized"),
        ]
    }
    return categories

def render_model_browser():
    """Enhanced model browser with current working models"""
    st.title("🔍 Model Browser - Current Working Models")
    
    # Status and warnings
    col1, col2 = st.columns([2, 1])
    with col1:
        st.success("✅ Updated with current OpenRouter models (June 2025)")
    with col2:
        if st.button("🔄 Refresh OpenRouter", help="Open OpenRouter in new tab"):
            st.markdown("""
            <a href="https://openrouter.ai/models" target="_blank">
                <button>🔄 Check Latest Models</button>
            </a>
            """, unsafe_allow_html=True)
    
    st.warning("⚠️ **Rate Limits**: Free models have 50 requests/day (basic) or 1000/day (with $10+ credits)")
    
    # Quick stats
    total_free = len(CURRENT_FREE_MODELS)
    total_cheap = len(ULTRA_CHEAP_MODELS)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🆓 Free Models", total_free)
    with col2:
        st.metric("💰 Ultra-Cheap", total_cheap)
    with col3:
        st.metric("🏷️ Categories", 6)
    with col4:
        st.metric("📊 Total Available", total_free + total_cheap)
    
    # Model selection state
    if 'browser_selected_models' not in st.session_state:
        st.session_state.browser_selected_models = []
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 By Category", "🆓 All Free Models", "💰 Ultra-Cheap", "🔍 Search"])
    
    with tab1:
        st.subheader("🎯 Browse Models by Use Case")
        
        categories = categorize_current_models()
        
        for category_name, models in categories.items():
            with st.expander(f"{category_name} ({len(models)} models)", expanded=False):
                for model_id, description in models:
                    col1, col2, col3 = st.columns([4, 1, 1])
                    
                    with col1:
                        st.write(f"**{description}**")
                        st.caption(f"`{model_id}`")
                    
                    with col2:
                        if ':free' in model_id:
                            st.success("FREE")
                        else:
                            st.info("Cheap")
                    
                    with col3:
                        if st.button("➕", key=f"cat_add_{hash(model_id)}", help=f"Add {model_id}"):
                            if model_id not in st.session_state.browser_selected_models:
                                st.session_state.browser_selected_models.append(model_id)
                                st.success("Added!")
                                st.rerun()
    
    with tab2:
        st.subheader("🆓 All Free Models")
        st.info("These models should be available for free (subject to rate limits)")
        
        for model_id, display_name in CURRENT_FREE_MODELS.items():
            col1, col2, col3 = st.columns([4, 1, 1])
            
            with col1:
                st.write(f"**{display_name}**")
                st.caption(f"`{model_id}`")
            
            with col2:
                st.success("FREE")
            
            with col3:
                if st.button("➕", key=f"free_add_{hash(model_id)}", help=f"Add {model_id}"):
                    if model_id not in st.session_state.browser_selected_models:
                        st.session_state.browser_selected_models.append(model_id)
                        st.success("Added!")
                        st.rerun()
    
    with tab3:
        st.subheader("💰 Ultra-Cheap Models (Under $0.10/1M tokens)")
        st.info("These models cost very little - great for high-volume use")
        
        for model_id, display_name in ULTRA_CHEAP_MODELS.items():
            col1, col2, col3 = st.columns([4, 1, 1])
            
            with col1:
                st.write(f"**{display_name}**")
                st.caption(f"`{model_id}`")
            
            with col2:
                price = display_name.split("($")[1].split(")")[0] if "($" in display_name else "Cheap"
                st.info(price)
            
            with col3:
                if st.button("➕", key=f"cheap_add_{hash(model_id)}", help=f"Add {model_id}"):
                    if model_id not in st.session_state.browser_selected_models:
                        st.session_state.browser_selected_models.append(model_id)
                        st.success("Added!")
                        st.rerun()
    
    with tab4:
        st.subheader("🔍 Search Models")
        
        search_term = st.text_input(
            "Search by name, capability, or provider:",
            placeholder="e.g., deepseek, vision, coding, free"
        )
        
        if search_term:
            search_lower = search_term.lower()
            matching_models = []
            
            # Search in free models
            for model_id, display_name in CURRENT_FREE_MODELS.items():
                if (search_lower in model_id.lower() or 
                    search_lower in display_name.lower()):
                    matching_models.append((model_id, display_name, "FREE"))
            
            # Search in cheap models
            for model_id, display_name in ULTRA_CHEAP_MODELS.items():
                if (search_lower in model_id.lower() or 
                    search_lower in display_name.lower()):
                    price = display_name.split("($")[1].split(")")[0] if "($" in display_name else "Cheap"
                    matching_models.append((model_id, display_name, price))
            
            st.write(f"🔍 Found {len(matching_models)} matching models:")
            
            for model_id, display_name, price_info in matching_models:
                col1, col2, col3 = st.columns([4, 1, 1])
                
                with col1:
                    st.write(f"**{display_name}**")
                    st.caption(f"`{model_id}`")
                
                with col2:
                    if price_info == "FREE":
                        st.success("FREE")
                    else:
                        st.info(price_info)
                
                with col3:
                    if st.button("➕", key=f"search_add_{hash(model_id)}", help=f"Add {model_id}"):
                        if model_id not in st.session_state.browser_selected_models:
                            st.session_state.browser_selected_models.append(model_id)
                            st.success("Added!")
                            st.rerun()
    
    # Selected models section
    st.divider()
    
    if st.session_state.browser_selected_models:
        st.subheader("✅ Selected Models")
        
        for i, model_id in enumerate(st.session_state.browser_selected_models):
            col1, col2, col3 = st.columns([4, 1, 1])
            
            with col1:
                # Get display name
                display_name = CURRENT_FREE_MODELS.get(model_id) or ULTRA_CHEAP_MODELS.get(model_id) or model_id
                st.write(f"**{display_name}**")
                st.caption(f"`{model_id}`")
            
            with col2:
                if ':free' in model_id:
                    st.success("FREE")
                elif model_id in ULTRA_CHEAP_MODELS:
                    st.info("Cheap")
                else:
                    st.warning("Custom")
            
            with col3:
                if st.button("🗑️", key=f"remove_{i}", help=f"Remove {model_id}"):
                    st.session_state.browser_selected_models.remove(model_id)
                    st.rerun()
        
        # Transfer actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Transfer to Chat", type="primary", use_container_width=True):
                # Transfer first model to chat (since chat now uses single model)
                if st.session_state.browser_selected_models:
                    st.session_state.transfer_models = [st.session_state.browser_selected_models[0]]
                    st.success(f"✅ Transferred: {st.session_state.browser_selected_models[0]}")
                    st.info("💡 Go to Chat tab to use the model!")
        
        with col2:
            if st.button("📋 Copy Model IDs", use_container_width=True):
                model_list = "\n".join(st.session_state.browser_selected_models)
                st.text_area("Copy these model IDs:", model_list, height=100)
        
        with col3:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.browser_selected_models = []
                st.rerun()
    
    else:
        st.info("👆 Select models above to transfer to chat or copy IDs")
    
    # Manual model input
    st.divider()
    st.subheader("➕ Add Custom Model")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        custom_model = st.text_input(
            "Enter any OpenRouter model ID:",
            placeholder="e.g., openai/gpt-4, anthropic/claude-3-sonnet",
            help="You can add any model from OpenRouter, not just the ones listed above"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        if st.button("➕ Add Custom", use_container_width=True):
            if custom_model and custom_model.strip():
                model_id = custom_model.strip()
                if model_id not in st.session_state.browser_selected_models:
                    st.session_state.browser_selected_models.append(model_id)
                    st.success(f"✅ Added custom model: {model_id}")
                    st.rerun()
                else:
                    st.info("Model already in list!")
            else:
                st.error("Please enter a model ID")
    
    # Help section
    with st.expander("💡 Tips & Information", expanded=False):
        st.markdown("""
        **Free Model Limits:**
        - 50 requests/day (basic account)
        - 1000 requests/day (after purchasing $10+ credits)
        - 20 requests/minute rate limit
        
        **Finding New Models:**
        - Visit [OpenRouter Models](https://openrouter.ai/models)
        - Filter by "FREE" pricing
        - Look for models ending in `:free`
        
        **Model Recommendations:**
        - **General Use**: Llama 3.3 70B or DeepSeek Chat
        - **Coding**: DeepSeek Coder or Qwen 2.5 Coder
        - **Reasoning**: DeepSeek R1 or Llama 3.3 70B
        - **Speed**: Llama 3.2 1B or 3B
        - **Vision**: Llama 3.2 11B Vision
        
        **Cost Saving:**
        - Use FREE models for testing and light use
        - Switch to ultra-cheap models for production
        - Most conversations cost under $0.01 with cheap models
        """)
    
    # Footer links
    st.divider()
    st.markdown("""
    **🔗 Useful Links:**
    - [OpenRouter Models](https://openrouter.ai/models) - Browse all available models
    - [OpenRouter Pricing](https://openrouter.ai/models?pricing=free) - Free models filter
    - [API Documentation](https://openrouter.ai/docs) - Integration guides
    - [Rate Limits Info](https://openrouter.ai/docs/faq) - Usage limits and pricing
    """)
