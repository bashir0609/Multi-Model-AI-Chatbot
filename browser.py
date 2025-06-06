# browser.py - Simple model browser with verified models only

import streamlit as st
import pandas as pd
import json
from models import MODEL_OPTIONS, categorize_models, analyze_model_capabilities, get_model_stats, get_cost_info

def render_model_browser():
    """Render a simple model browser interface with verified models."""
    st.header("🔍 Model Browser - Verified Working Models")
    st.markdown("**Only verified working models** - no more errors!")
    
    # Model statistics
    stats = get_model_stats()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Models", stats['total'])
    with col2:
        st.metric("FREE Models", stats['free'])
        st.success("No cost!")
    with col3:
        st.metric("Ultra-Cheap", stats['cheap'])
        st.info("Under $0.20/1M")
    
    # Quick recommendations
    st.subheader("⭐ Verified Working Models")
    
    recommendations = {
        "🧠 Best Free Reasoning": ("deepseek/deepseek-r1-distill-llama-70b:free", "70B distilled reasoning model"),
        "🔬 Google Thinking": ("google/gemini-2.0-flash-thinking-exp:free", "Latest thinking model"),
        "💰 Ultra-Cheap": ("deepseek/deepseek-r1-distill-llama-8b", "Only ~$0.04/1M tokens"),
        "🚀 Latest": ("deepseek/deepseek-r1-0528", "Updated reasoning model")
    }
    
    cols = st.columns(2)
    for i, (rec_name, (model_id, description)) in enumerate(recommendations.items()):
        with cols[i % 2]:
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
    
    # All models list
    st.subheader("📋 All Verified Models")
    
    for model_id, display_name in MODEL_OPTIONS.items():
        col1, col2, col3 = st.columns([4, 1, 1])
        with col1:
            st.write(f"**{display_name}**")
            st.caption(f"`{model_id}`")
        with col2:
            cost_info = get_cost_info(model_id, display_name)
            if cost_info['color'] == 'success':
                st.success("FREE")
            elif cost_info['color'] == 'info':
                st.info("Cheap")
            else:
                st.warning("Paid")
        with col3:
            if st.button(f"Add", key=f"add_{hash(model_id)}"):
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
    
    # Important note
    st.info("💡 **Note**: This list contains only verified working models. If you get a 'model not found' error, the model may have been discontinued.")
    
    # Export functionality
    with st.expander("📤 Export Model Information", expanded=False):
        if st.button("📋 Copy All Models as JSON"):
            model_json = json.dumps(MODEL_OPTIONS, indent=2)
            st.code(model_json, language="json")
            st.success("Model list displayed above - copy as needed!")
