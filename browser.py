# browser.py - Model browser interface (Updated)

import streamlit as st
import pandas as pd
import json
from models import MODEL_OPTIONS, categorize_models, analyze_model_capabilities, get_model_stats, get_cost_info

def render_model_browser():
    """Render the complete model browser interface."""
    st.header("🔍 Model Browser & Explorer")
    st.markdown("Explore **FREE** models and **ultra-cheap** options with detailed capabilities")
    
    # Model statistics
    stats = get_model_stats()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Models", stats['total'])
    with col2:
        st.metric("FREE Models", stats['free'])
        st.success("No cost!")
    with col3:
        st.metric("Ultra-Cheap", stats['cheap'])
        st.info("Under $0.50/1M")
    with col4:
        st.metric("Providers", len(stats['providers']))
    
    # Cost breakdown info
    with st.expander("💡 Understanding Costs & Limits", expanded=False):
        st.markdown("""
        ### Free Models (:free)
        - **Cost**: Completely FREE
        - **Rate Limits**: 20 requests/minute
        - **Daily Limits**: 
          - 50 requests/day (basic account)  
          - 1000 requests/day (after buying $10+ credits)
        
        ### Ultra-Cheap Models
        - **Cost**: $0.02 - $0.50 per million tokens
        - **Comparison**: ~750 words = 1000 tokens = $0.02-0.50
        - **Example**: A 1000-word article costs $0.02-0.50 to generate
        
        ### How to Get Large Free Limits
        1. **Use FREE models** - Completely free with rate limits
        2. **Buy $10 credits** - Increases free model limits to 1000 req/day
        3. **Use ultra-cheap models** - Tiny costs, no rate limits
        """)
    
    # Provider distribution
    with st.expander("📊 Provider Distribution", expanded=False):
        provider_df = pd.DataFrame(list(stats['providers'].items()), columns=['Provider', 'Models'])
        provider_df = provider_df.sort_values('Models', ascending=False)
        st.dataframe(provider_df, use_container_width=True)
    
    # Quick recommendations (Updated with current models)
    st.subheader("⭐ Quick Recommendations")
    
    recommendations = {
        "🚀 Fastest Free": ("meta-llama/llama-3.2-1b-instruct:free", "Smallest, fastest responses"),
        "🧠 Best Reasoning": ("deepseek/deepseek-r1:free", "Advanced reasoning capabilities"),
        "🏆 Most Capable": ("meta-llama/llama-3.3-70b-instruct:free", "Best overall performance"),
        "🔬 Latest Google": ("google/gemini-2.5-flash:free", "Cutting-edge multimodal"),
        "💻 Code Expert": ("qwen/qwen2.5-coder-32b-instruct:free", "32B coding specialist"),
        "👁️ Vision": ("meta-llama/llama-3.2-11b-vision-instruct:free", "Image understanding"),
        "🦣 Largest Free": ("meta-llama/llama-4-maverick:free", "400B MoE parameters"),
        "💰 Cheapest Paid": ("mistralai/ministral-3b", "Only $0.04 per million tokens")
    }
    
    cols = st.columns(4)
    for i, (rec_name, (model_id, description)) in enumerate(recommendations.items()):
        with cols[i % 4]:
            display_name = MODEL_OPTIONS.get(model_id, model_id)
            cost_info = get_cost_info(model_id, display_name)
            
            st.write(f"**{rec_name}**")
            st.caption(description)
            
            # Show cost badge
            if cost_info['color'] == 'success':
                st.success(f"🆓 {cost_info['cost']}")
            elif cost_info['color'] == 'info':
                st.info(f"💰 {cost_info['cost']}")
            else:
                st.warning(f"💳 {cost_info['cost']}")
            
            if st.button(f"Select", key=f"rec_quick_{i}", use_container_width=True):
                if 'selected_models_browser' not in st.session_state:
                    st.session_state.selected_models_browser = []
                if model_id not in st.session_state.selected_models_browser:
                    st.session_state.selected_models_browser.append(model_id)
                    st.success(f"✅ Added {display_name.split(' (')[0]}")
                else:
                    st.info("Already selected!")
    
    # Search functionality
    st.subheader("🔍 Search Models")
    search_term = st.text_input("Search by name or capability", placeholder="e.g., reasoning, vision, free, cheap, llama")
    
    if search_term:
        matching_models = []
        search_lower = search_term.lower()
        
        for model_id, display_name in MODEL_OPTIONS.items():
            if (search_lower in model_id.lower() or 
                search_lower in display_name.lower()):
                matching_models.append((model_id, display_name))
        
        st.write(f"Found {len(matching_models)} matching models:")
        
        for search_idx, (model_id, display_name) in enumerate(matching_models):
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                st.write(f"**{display_name}**")
                st.caption(f"`{model_id}`")
            with col2:
                cost_info = get_cost_info(model_id, display_name)
                if cost_info['color'] == 'success':
                    st.success("FREE")
                elif cost_info['color'] == 'info':
                    st.info("Ultra-Cheap")
                else:
                    st.warning("Paid")
            with col3:
                if st.button(f"Add", key=f"search_{search_idx}_{hash(model_id)}"):
                    if 'selected_models_browser' not in st.session_state:
                        st.session_state.selected_models_browser = []
                    if model_id not in st.session_state.selected_models_browser:
                        st.session_state.selected_models_browser.append(model_id)
                        st.success("Added!")
                    else:
                        st.info("Already added!")
    
    # Categories view
    st.subheader("🏷️ Browse by Provider")
    categories = categorize_models()
    
    for category_name, models in categories.items():
        if models:  # Only show categories with models
            with st.expander(f"{category_name} ({len(models)} models)", expanded=False):
                for cat_idx, (model_id, display_name) in enumerate(models):
                    col1, col2, col3 = st.columns([4, 1, 1])
                    with col1:
                        st.write(f"**{display_name}**")
                        st.caption(f"Model ID: `{model_id}`")
                    with col2:
                        cost_info = get_cost_info(model_id, display_name)
                        if cost_info['color'] == 'success':
                            st.success(cost_info['cost'])
                        elif cost_info['color'] == 'info':
                            st.info(cost_info['cost'])
                        else:
                            st.warning(cost_info['cost'])
                    with col3:
                        category_key = category_name.replace(" ", "_").replace("(", "").replace(")", "")
                        if st.button(f"Add", key=f"cat_{category_key}_{cat_idx}"):
                            if 'selected_models_browser' not in st.session_state:
                                st.session_state.selected_models_browser = []
                            if model_id not in st.session_state.selected_models_browser:
                                st.session_state.selected_models_browser.append(model_id)
                                st.success("Added!")
                            else:
                                st.info("Already added!")
    
    # Capabilities view
    st.subheader("🎯 Browse by Capability")
    capabilities = analyze_model_capabilities()
    
    capability_names = {
        'reasoning': '🧠 Advanced Reasoning',
        'thinking': '💭 Thinking Models',
        'vision': '👁️ Vision/Multimodal', 
        'coding': '💻 Code Specialization',
        'large_scale': '🦣 Large Scale (70B+)',
        'experimental': '🧪 Experimental/Latest',
        'ultra_cheap': '💰 Ultra-Cheap Models'
    }
    
    for cap_key, cap_name in capability_names.items():
        models = capabilities.get(cap_key, [])
        if models:
            with st.expander(f"{cap_name} ({len(models)} models)", expanded=False):
                for cap_idx, (model_id, display_name) in enumerate(models):
                    col1, col2, col3 = st.columns([4, 1, 1])
                    with col1:
                        st.write(f"**{display_name}**")
                        st.caption(f"`{model_id}`")
                    with col2:
                        cost_info = get_cost_info(model_id, display_name)
                        if cost_info['color'] == 'success':
                            st.success("FREE")
                        elif cost_info['color'] == 'info':
                            st.info(cost_info['cost'])
                        else:
                            st.warning("Paid")
                    with col3:
                        if st.button(f"Add", key=f"cap_{cap_key}_{cap_idx}"):
                            if 'selected_models_browser' not in st.session_state:
                                st.session_state.selected_models_browser = []
                            if model_id not in st.session_state.selected_models_browser:
                                st.session_state.selected_models_browser.append(model_id)
                                st.success("Added!")
                            else:
                                st.info("Already added!")
    
    # Selected models summary
    if 'selected_models_browser' in st.session_state and st.session_state.selected_models_browser:
        st.subheader("✅ Selected Models for Comparison")
        
        cols = st.columns([4, 1])
        with cols[0]:
            for model_id in st.session_state.selected_models_browser:
                display_name = MODEL_OPTIONS.get(model_id, model_id)
                cost_info = get_cost_info(model_id, display_name)
                cost_badge = "🆓" if cost_info['type'] == 'free' else "💰" if cost_info['type'] == 'ultra_cheap' else "💳"
                st.write(f"{cost_badge} **{display_name}**")
        
        with cols[1]:
            if st.button("🗑️ Clear All", key="clear_browser_selection"):
                st.session_state.selected_models_browser = []
                st.rerun()
        
        if st.button("🚀 Use These Models in Chat", type="primary", use_container_width=True):
            # Transfer selected models to main chat
            st.session_state.transfer_models = st.session_state.selected_models_browser
            st.success("✅ Models transferred! Switch to the Chat tab to start chatting.")
    
    # Export functionality
    with st.expander("📤 Export Model Information", expanded=False):
        if st.button("📋 Copy All Models as JSON"):
            model_json = json.dumps(MODEL_OPTIONS, indent=2)
            st.code(model_json, language="json")
            st.success("Model list displayed above - copy as needed!")
        
        if st.button("📊 Download Model Statistics CSV"):
            # Create a comprehensive model DataFrame
            model_data = []
            for model_id, display_name in MODEL_OPTIONS.items():
                provider = model_id.split('/')[0] if '/' in model_id else 'other'
                cost_info = get_cost_info(model_id, display_name)
                
                size = 'Unknown'
                if any(x in display_name.lower() for x in ['0.5b', '1b', '2b', '3b', '7b', '8b']):
                    size = 'Small (≤8B)'
                elif any(x in display_name.lower() for x in ['11b', '12b', '14b', '24b', '27b', '30b', '32b']):
                    size = 'Medium (9B-32B)'
                elif any(x in display_name.lower() for x in ['70b', '72b']):
                    size = 'Large (70B+)'
                elif any(x in display_name.lower() for x in ['109b', '235b', '253b', '400b', '405b']):
                    size = 'Ultra Large (100B+)'
                
                model_data.append({
                    'Model ID': model_id,
                    'Display Name': display_name,
                    'Provider': provider.title(),
                    'Cost': cost_info['cost'],
                    'Type': cost_info['type'].title(),
                    'Size Category': size,
                    'Limits': cost_info['limits']
                })
            
            df = pd.DataFrame(model_data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="openrouter_models_enhanced.csv",
                mime="text/csv"
            )
