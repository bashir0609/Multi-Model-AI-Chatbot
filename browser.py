# browser.py - Fixed Model Browser (No API key required)

import streamlit as st

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
    
    # Google Models (Experimental but Often Free)
    "google/gemini-2.0-flash-experimental:free": "Gemini 2.0 Flash Experimental (FREE)",
    "google/gemini-pro:free": "Gemini Pro (FREE)",
    
    # Mistral Models
    "mistralai/mistral-7b-instruct:free": "Mistral 7B Instruct (FREE)",
    "mistralai/mistral-small-3.1:free": "Mistral Small 3.1 (FREE)",
    
    # Qwen Models (Often Available Free)
    "qwen/qwen2.5-72b-instruct:free": "Qwen 2.5 72B Instruct (FREE)",
    "qwen/qwen2.5-coder-32b-instruct:free": "Qwen 2.5 Coder 32B (FREE)",
}

ULTRA_CHEAP_MODELS = {
    # Under $0.10 per 1M tokens
    "meta-llama/llama-3.2-3b-instruct": "Llama 3.2 3B ($0.02/1M)",
    "deepseek/deepseek-chat": "DeepSeek Chat ($0.02/1M)",
    "meta-llama/llama-3.1-8b-instruct": "Llama 3.1 8B ($0.05/1M)",
    "mistralai/mistral-7b-instruct": "Mistral 7B ($0.07/1M)",
    "qwen/qwen2.5-7b-instruct": "Qwen 2.5 7B ($0.07/1M)",
}

def render_model_browser():
    """Model browser - No API key required"""
    st.title("🔍 Model Browser - Find Working Models")
    
    st.info("💡 **Browse and select models to use in chat. No API key required for browsing!**")
    
    # Warning about model availability
    st.warning("⚠️ **IMPORTANT**: Free model availability changes frequently. Always test models in the Chat tab!")
    
    # Initialize button counter for unique keys
    if 'button_counter' not in st.session_state:
        st.session_state.button_counter = 0
    
    # Quick access links
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
    
    # Model selection state
    if 'browser_selected_models' not in st.session_state:
        st.session_state.browser_selected_models = []
    
    # Stats
    total_free = len(CURRENT_FREE_MODELS)
    total_cheap = len(ULTRA_CHEAP_MODELS)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🆓 Free Models", total_free)
    with col2:
        st.metric("💰 Ultra-Cheap", total_cheap)
    with col3:
        st.metric("📋 Selected", len(st.session_state.browser_selected_models))
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["🆓 Free Models", "💰 Ultra-Cheap", "📋 Selected"])
    
    with tab1:
        st.subheader("🆓 Free Models (Subject to Rate Limits)")
        st.caption("Rate limits: 50/day (basic) or 1000/day (with $10+ credits)")
        
        # Organize by provider
        providers = {}
        for model_id, display_name in CURRENT_FREE_MODELS.items():
            provider = model_id.split('/')[0]
            if provider not in providers:
                providers[provider] = []
            providers[provider].append((model_id, display_name))
        
        for provider, models in providers.items():
            with st.expander(f"🏢 {provider.title()} ({len(models)} models)", expanded=provider=='deepseek'):
                for idx, (model_id, display_name) in enumerate(models):
                    col1, col2, col3 = st.columns([4, 1, 1])
                    
                    with col1:
                        st.write(f"**{display_name}**")
                        st.caption(f"`{model_id}`")
                        
                        # Add capability hints
                        if 'r1' in model_id.lower():
                            st.caption("🧠 Advanced Reasoning")
                        elif 'coder' in model_id.lower():
                            st.caption("💻 Code Specialist")
                        elif 'vision' in model_id.lower():
                            st.caption("👁️ Vision/Image Understanding")
                        elif '1b' in model_id.lower():
                            st.caption("⚡ Ultra Fast")
                        elif '70b' in model_id.lower() or '72b' in model_id.lower():
                            st.caption("🦣 Large Scale")
                    
                    with col2:
                        st.success("FREE")
                    
                    with col3:
                        # Create unique key using counter
                        st.session_state.button_counter += 1
                        unique_key = f"free_add_{st.session_state.button_counter}_{provider}_{idx}"
                        
                        if st.button("➕", key=unique_key, help=f"Add {model_id}"):
                            if model_id not in st.session_state.browser_selected_models:
                                st.session_state.browser_selected_models.append(model_id)
                                st.success("Added!")
                                st.rerun()
                            else:
                                st.info("Already added!")
    
    with tab2:
        st.subheader("💰 Ultra-Cheap Models (Under $0.10/1M tokens)")
        st.caption("Perfect for production use - no rate limits!")
        
        for idx, (model_id, display_name) in enumerate(ULTRA_CHEAP_MODELS.items()):
            col1, col2, col3 = st.columns([4, 1, 1])
            
            with col1:
                st.write(f"**{display_name}**")
                st.caption(f"`{model_id}`")
            
            with col2:
                # Extract price
                price = display_name.split("($")[1].split(")")[0] if "($" in display_name else "Cheap"
                st.info(price)
            
            with col3:
                # Create unique key using counter
                st.session_state.button_counter += 1
                unique_key = f"cheap_add_{st.session_state.button_counter}_{idx}"
                
                if st.button("➕", key=unique_key, help=f"Add {model_id}"):
                    if model_id not in st.session_state.browser_selected_models:
                        st.session_state.browser_selected_models.append(model_id)
                        st.success("Added!")
                        st.rerun()
                    else:
                        st.info("Already added!")
    
    with tab3:
        st.subheader("📋 Selected Models")
        
        if st.session_state.browser_selected_models:
            st.success(f"✅ You have {len(st.session_state.browser_selected_models)} models selected")
            
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
                    # Create unique remove key using counter
                    st.session_state.button_counter += 1
                    unique_remove_key = f"remove_{st.session_state.button_counter}_{i}"
                    
                    if st.button("🗑️", key=unique_remove_key, help=f"Remove {model_id}"):
                        st.session_state.browser_selected_models.remove(model_id)
                        st.rerun()
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.session_state.button_counter += 1
                if st.button("🚀 Send to Chat", type="primary", use_container_width=True, key=f"send_to_chat_btn_{st.session_state.button_counter}"):
                    # Transfer the first selected model to chat
                    if st.session_state.browser_selected_models:
                        st.session_state.transfer_models = [st.session_state.browser_selected_models[0]]
                        st.balloons()
                        st.success(f"✅ Sent to Chat: {st.session_state.browser_selected_models[0]}")
                        st.info("💡 Go to Chat tab to use the model!")
            
            with col2:
                st.session_state.button_counter += 1
                if st.button("📋 Copy IDs", use_container_width=True, key=f"copy_model_ids_btn_{st.session_state.button_counter}"):
                    model_list = "\n".join(st.session_state.browser_selected_models)
                    st.text_area("📋 Copy these model IDs:", model_list, height=100, key=f"model_ids_textarea_{st.session_state.button_counter}")
            
            with col3:
                st.session_state.button_counter += 1
                if st.button("🗑️ Clear All", use_container_width=True, key=f"clear_all_models_btn_{st.session_state.button_counter}"):
                    st.session_state.browser_selected_models = []
                    st.rerun()
        
        else:
            st.info("👆 No models selected yet. Browse the tabs above to add models!")
    
    # Manual model input section
    st.divider()
    st.subheader("➕ Add Any Model")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.session_state.button_counter += 1
        custom_model = st.text_input(
            "Enter any OpenRouter model ID:",
            placeholder="e.g., anthropic/claude-3-sonnet, openai/gpt-4",
            help="You can add any model from OpenRouter, not just the ones listed above",
            key=f"custom_model_input_field_{st.session_state.button_counter}"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        st.session_state.button_counter += 1
        if st.button("➕ Add", use_container_width=True, type="secondary", key=f"add_custom_model_btn_{st.session_state.button_counter}"):
            if custom_model and custom_model.strip():
                model_id = custom_model.strip()
                if model_id not in st.session_state.browser_selected_models:
                    st.session_state.browser_selected_models.append(model_id)
                    st.success(f"✅ Added: {model_id}")
                    st.rerun()
                else:
                    st.info("Already in list!")
            else:
                st.error("Please enter a model ID")
    
    # Instructions
    st.divider()
    with st.expander("📖 How to Use", expanded=False):
        st.markdown("""
        **Step 1**: Browse free models or ultra-cheap options above
        
        **Step 2**: Click ➕ to add models to your selection
        
        **Step 3**: Click "🚀 Send to Chat" to transfer a model to the Chat tab
        
        **Step 4**: Go to Chat tab and start using the model!
        
        **💡 Tips:**
        - Free models have daily limits (50 or 1000 requests)
        - Ultra-cheap models cost ~$0.02-0.07 per 1000 words
        - Always test models in Chat before using extensively
        - Check OpenRouter live for latest model availability
        """)
    
    # Footer
    st.caption("🔄 Model availability updated June 2025. Always verify current models on OpenRouter.")
