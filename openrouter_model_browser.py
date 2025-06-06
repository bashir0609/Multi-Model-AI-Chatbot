"""
Model Browser for OpenRouter Free Models
Run this script to explore all available free models and their capabilities
"""

import streamlit as st
from settings import OPENROUTER_MODELS
import pandas as pd

def categorize_models():
    """Categorize models by provider and type."""
    categories = {
        'DeepSeek (Reasoning & Performance)': [],
        'Meta Llama (General Purpose)': [],
        'Qwen (Multilingual & Vision)': [],
        'Google (Research & Latest)': [],
        'Microsoft (Enterprise & Reasoning)': [],
        'Mistral (European AI)': [],
        'NVIDIA (Optimized Performance)': [],
        'Specialized (Vision, Code, Multimodal)': [],
        'Low-Cost Paid (Under $1/1M tokens)': [],
        'OpenAI (Premium Paid)': []
    }
    
    for model_id, display_name in OPENROUTER_MODELS.items():
        if model_id.startswith('deepseek/'):
            categories['DeepSeek (Reasoning & Performance)'].append((model_id, display_name))
        elif model_id.startswith('meta-llama/'):
            categories['Meta Llama (General Purpose)'].append((model_id, display_name))
        elif model_id.startswith('qwen/'):
            categories['Qwen (Multilingual & Vision)'].append((model_id, display_name))
        elif model_id.startswith('google/'):
            categories['Google (Research & Latest)'].append((model_id, display_name))
        elif model_id.startswith('microsoft/'):
            categories['Microsoft (Enterprise & Reasoning)'].append((model_id, display_name))
        elif model_id.startswith('mistralai/'):
            categories['Mistral (European AI)'].append((model_id, display_name))
        elif model_id.startswith('nvidia/'):
            categories['NVIDIA (Optimized Performance)'].append((model_id, display_name))
        elif "Cheap" in display_name and ":free" not in model_id:
            categories['Low-Cost Paid (Under $1/1M tokens)'].append((model_id, display_name))
        elif model_id.startswith('gpt-'):
            categories['OpenAI (Premium Paid)'].append((model_id, display_name))
        else:
            categories['Specialized (Vision, Code, Multimodal)'].append((model_id, display_name))
    
    return categories

def analyze_model_capabilities():
    """Analyze and categorize model capabilities."""
    capabilities = {
        'reasoning': [],
        'vision': [],
        'coding': [],
        'multilingual': [],
        'large_scale': [],
        'multimodal': [],
        'experimental': []
    }
    
    for model_id, display_name in OPENROUTER_MODELS.items():
        name_lower = display_name.lower()
        
        # Reasoning models
        if any(word in name_lower for word in ['reasoning', 'r1', 'prover', 'thinking', 'qwq']):
            capabilities['reasoning'].append((model_id, display_name))
        
        # Vision/Multimodal models
        if any(word in name_lower for word in ['vision', 'vl', 'multimodal', 'kimi']):
            capabilities['vision'].append((model_id, display_name))
        
        # Coding specialized
        if any(word in name_lower for word in ['coder', 'code', 'deepcoder', 'olympiccoder', 'devstral']):
            capabilities['coding'].append((model_id, display_name))
        
        # Large scale (70B+)
        if any(word in name_lower for word in ['70b', '405b', '235b', '253b']):
            capabilities['large_scale'].append((model_id, display_name))
        
        # Experimental/Latest
        if any(word in name_lower for word in ['experimental', 'preview', 'maverick', 'scout']):
            capabilities['experimental'].append((model_id, display_name))
    
    return capabilities

def get_model_stats():
    """Get statistics about available models."""
    total_models = len(OPENROUTER_MODELS)
    free_models = len([m for m in OPENROUTER_MODELS.keys() if ':free' in m])
    paid_models = total_models - free_models
    
    # Provider distribution
    providers = {}
    for model_id in OPENROUTER_MODELS.keys():
        provider = model_id.split('/')[0] if '/' in model_id else 'openai'
        providers[provider] = providers.get(provider, 0) + 1
    
    return {
        'total': total_models,
        'free': free_models,
        'paid': paid_models,
        'providers': providers
    }

def render_model_browser():
    """Render the complete model browser interface."""
    st.title("🔍 OpenRouter Model Browser")
    st.markdown("Explore all available FREE and low-cost models with detailed capabilities")
    
    # Model statistics
    stats = get_model_stats()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Models", stats['total'])
    with col2:
        st.metric("FREE Models", stats['free'])
        st.success("No cost!")
    with col3:
        st.metric("Paid Models", stats['paid'])
        st.info("Under $1.50/1M")
    with col4:
        st.metric("Providers", len(stats['providers']))
    
    # Provider distribution
    st.subheader("📊 Provider Distribution")
    provider_df = pd.DataFrame(list(stats['providers'].items()), columns=['Provider', 'Models'])
    provider_df = provider_df.sort_values('Models', ascending=False)
    st.dataframe(provider_df, use_container_width=True)
    
    # Categories view
    st.subheader("🏷️ Browse by Category")
    categories = categorize_models()
    
    for category_name, models in categories.items():
        if models:  # Only show categories with models
            with st.expander(f"{category_name} ({len(models)} models)", expanded=False):
                for model_id, display_name in models:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        # Determine cost
                        if ":free" in model_id:
                            cost_badge = "🆓 FREE"
                            cost_color = "success"
                        elif "Cheap" in display_name:
                            cost_match = display_name.split("$")[1].split("/")[0] if "$" in display_name else "Low"
                            cost_badge = f"💰 ${cost_match}/1M"
                            cost_color = "info"
                        elif model_id.startswith('gpt-'):
                            cost_badge = "💸 $1.50/1M"
                            cost_color = "warning"
                        else:
                            cost_badge = "❓ Variable"
                            cost_color = "secondary"
                        
                        # Display model info
                        st.write(f"**{display_name}**")
                        st.caption(f"Model ID: `{model_id}`")
                        
                        if cost_color == "success":
                            st.success(cost_badge)
                        elif cost_color == "info":
                            st.info(cost_badge)
                        elif cost_color == "warning":
                            st.warning(cost_badge)
                        else:
                            st.write(cost_badge)
                    
                    with col2:
                        if st.button(f"Select", key=f"select_{model_id}"):
                            st.session_state.selected_model = model_id
                            st.success(f"Selected: {display_name}")
    
    # Capabilities view
    st.subheader("🎯 Browse by Capability")
    capabilities = analyze_model_capabilities()
    
    capability_names = {
        'reasoning': '🧠 Advanced Reasoning',
        'vision': '👁️ Vision/Multimodal',
        'coding': '💻 Code Specialization',
        'large_scale': '🦣 Large Scale (70B+)',
        'experimental': '🧪 Experimental/Latest'
    }
    
    for cap_key, cap_name in capability_names.items():
        models = capabilities.get(cap_key, [])
        if models:
            with st.expander(f"{cap_name} ({len(models)} models)", expanded=False):
                for model_id, display_name in models:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"**{display_name}**")
                        st.caption(f"`{model_id}`")
                    with col2:
                        if ":free" in model_id:
                            st.success("FREE")
                        else:
                            st.info("Paid")
    
    # Search functionality
    st.subheader("🔍 Search Models")
    search_term = st.text_input("Search by name or capability", placeholder="e.g., reasoning, vision, free, llama")
    
    if search_term:
        matching_models = []
        search_lower = search_term.lower()
        
        for model_id, display_name in OPENROUTER_MODELS.items():
            if (search_lower in model_id.lower() or 
                search_lower in display_name.lower()):
                matching_models.append((model_id, display_name))
        
        st.write(f"Found {len(matching_models)} matching models:")
        
        for model_id, display_name in matching_models:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{display_name}**")
                st.caption(f"`{model_id}`")
            with col2:
                if ":free" in model_id:
                    st.success("FREE")
                elif "Cheap" in display_name:
                    st.info("Cheap")
                else:
                    st.warning("Paid")
            with col3:
                if st.button(f"Use", key=f"use_{model_id}"):
                    st.session_state.selected_model = model_id
                    st.success("Model selected!")
    
    # Quick recommendations
    st.subheader("⭐ Quick Recommendations")
    
    recommendations = {
        "🚀 Fastest Free": "meta-llama/llama-3.2-1b-instruct:free",
        "🧠 Best Reasoning": "deepseek/deepseek-r1:free",
        "🏆 Most Capable": "meta-llama/llama-3.3-70b-instruct:free",
        "🔬 Latest Research": "google/gemini-2.0-flash-experimental:free",
        "💻 Best for Code": "qwen/qwen2.5-coder-32b-instruct:free",
        "👁️ Vision Tasks": "meta-llama/llama-3.2-11b-vision-instruct:free",
        "🦣 Largest Scale": "meta-llama/llama-3.1-405b-base:free",
        "💰 Ultra Cheap": "meta-llama/llama-3.2-3b-instruct"
    }
    
    cols = st.columns(4)
    for i, (rec_name, model_id) in enumerate(recommendations.items()):
        with cols[i % 4]:
            display_name = OPENROUTER_MODELS.get(model_id, model_id)
            st.write(f"**{rec_name}**")
            st.caption(display_name.split(' (')[0])
            if st.button(f"Use {rec_name.split(' ')[1]}", key=f"rec_{i}"):
                st.session_state.selected_model = model_id
                st.success(f"Selected {rec_name}!")
    
    # Export model list
    st.subheader("📤 Export Model List")
    if st.button("📋 Copy All Models as JSON"):
        import json
        model_json = json.dumps(OPENROUTER_MODELS, indent=2)
        st.code(model_json, language="json")
        st.success("Model list displayed above - copy as needed!")

if __name__ == "__main__":
    # This can be run as a standalone Streamlit app
    render_model_browser()