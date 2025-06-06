import os
import streamlit as st
import requests
import asyncio
import aiohttp
import time
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# --- Load API key from .env file ---
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# ---- MODEL LIST ----
MODEL_OPTIONS = {
    # DeepSeek
    "deepseek/deepseek-r1-0528-qwen3-8b:free": "DeepSeek R1 0528 Qwen3 8B (FREE)",
    "deepseek/deepseek-r1-0528:free": "DeepSeek R1 0528 (FREE)",
    "deepseek/deepseek-prover-v2:free": "DeepSeek Prover V2 (FREE)",
    "deepseek/deepseek-r1t-chimera:free": "DeepSeek R1T Chimera (FREE)",
    "deepseek/deepseek-v3-base:free": "DeepSeek V3 Base (FREE)",
    "deepseek/deepseek-v3-0324:free": "DeepSeek V3 0324 (FREE)",
    "deepseek/deepseek-r1-zero:free": "DeepSeek R1 Zero (FREE)",
    "deepseek/deepseek-r1-distill-qwen-32b:free": "DeepSeek R1 Distill Qwen 32B (FREE)",
    "deepseek/deepseek-r1-distill-qwen-14b:free": "DeepSeek R1 Distill Qwen 14B (FREE)",
    "deepseek/deepseek-r1-distill-llama-70b:free": "DeepSeek R1 Distill Llama 70B (FREE)",
    "deepseek/deepseek-r1:free": "DeepSeek R1 (FREE)",
    "deepseek/deepseek-v3:free": "DeepSeek V3 (FREE)",
    # Meta Llama
    "meta-llama/llama-3.3-8b-instruct:free": "Llama 3.3 8B Instruct (FREE)",
    "meta-llama/llama-3.3-70b-instruct:free": "Llama 3.3 70B Instruct (FREE)",
    "meta-llama/llama-3.2-1b-instruct:free": "Llama 3.2 1B Instruct (FREE)",
    "meta-llama/llama-3.2-3b-instruct:free": "Llama 3.2 3B Instruct (FREE)",
    "meta-llama/llama-3.2-11b-vision-instruct:free": "Llama 3.2 11B Vision Instruct (FREE)",
    "meta-llama/llama-3.1-405b-base:free": "Llama 3.1 405B Base (FREE)",
    "meta-llama/llama-3.1-8b-instruct:free": "Llama 3.1 8B Instruct (FREE)",
    "meta-llama/llama-4-maverick:free": "Llama 4 Maverick (FREE)",
    "meta-llama/llama-4-scout:free": "Llama 4 Scout (FREE)",
    # Qwen
    "qwen/qwen3-30b-a3b:free": "Qwen3 30B A3B (FREE)",
    "qwen/qwen3-8b:free": "Qwen3 8B (FREE)",
    "qwen/qwen3-14b:free": "Qwen3 14B (FREE)",
    "qwen/qwen3-32b:free": "Qwen3 32B (FREE)",
    "qwen/qwen3-235b-a22b:free": "Qwen3 235B A22B (FREE)",
    "qwen/qwen2.5-vl-3b-instruct:free": "Qwen2.5 VL 3B Instruct (FREE)",
    "qwen/qwen2.5-vl-32b-instruct:free": "Qwen2.5 VL 32B Instruct (FREE)",
    "qwen/qwen2.5-vl-72b-instruct:free": "Qwen2.5 VL 72B Instruct (FREE)",
    "qwen/qwen2.5-vl-7b-instruct:free": "Qwen2.5 VL 7B Instruct (FREE)",
    "qwen/qwen2.5-coder-32b-instruct:free": "Qwen2.5 Coder 32B Instruct (FREE)",
    "qwen/qwen2.5-7b-instruct:free": "Qwen2.5 7B Instruct (FREE)",
    "qwen/qwen2.5-72b-instruct:free": "Qwen2.5 72B Instruct (FREE)",
    "qwen/qwq-32b:free": "QwQ 32B (FREE)",
    # Google
    "google/gemma-3n-4b:free": "Gemma 3n 4B (FREE)",
    "google/gemma-3-1b:free": "Gemma 3 1B (FREE)",
    "google/gemma-3-4b:free": "Gemma 3 4B (FREE)",
    "google/gemma-3-12b:free": "Gemma 3 12B (FREE)",
    "google/gemma-3-27b:free": "Gemma 3 27B (FREE)",
    "google/gemma-2-9b:free": "Gemma 2 9B (FREE)",
    "google/gemini-2.5-pro-experimental:free": "Gemini 2.5 Pro Experimental (FREE)",
    "google/gemini-2.0-flash-experimental:free": "Gemini 2.0 Flash Experimental (FREE)",
    "google/gemma-7b-it:free": "Gemma 7B IT (FREE)",
    # Mistral
    "mistralai/devstral-small:free": "Devstral Small (FREE)",
    "mistralai/mistral-small-3.1-24b:free": "Mistral Small 3.1 24B (FREE)",
    "mistralai/mistral-small-3:free": "Mistral Small 3 (FREE)",
    "mistralai/mistral-nemo:free": "Mistral Nemo (FREE)",
    "mistralai/mistral-7b-instruct:free": "Mistral 7B Instruct (FREE)",
    # Microsoft
    "microsoft/phi-4-reasoning-plus:free": "Phi 4 Reasoning Plus (FREE)",
    "microsoft/phi-4-reasoning:free": "Phi 4 Reasoning (FREE)",
    "microsoft/mai-ds-r1:free": "MAI DS R1 (FREE)",
    # Nous Research
    "nousresearch/deephermes-3-mistral-24b-preview:free": "DeepHermes 3 Mistral 24B Preview (FREE)",
    "nousresearch/deephermes-3-llama-3-8b-preview:free": "DeepHermes 3 Llama 3 8B Preview (FREE)",
    # OpenGVLab
    "opengvlab/internvl3-14b:free": "InternVL3 14B (FREE)",
    "opengvlab/internvl3-2b:free": "InternVL3 2B (FREE)",
    # THUDM
    "thudm/glm-z1-32b:free": "GLM Z1 32B (FREE)",
    "thudm/glm-4-32b:free": "GLM 4 32B (FREE)",
    # Specialized AI
    "sarvamai/sarvam-m:free": "Sarvam-M (FREE)",
    "shisa-ai/shisa-v2-llama-3.3-70b:free": "Shisa V2 Llama 3.3 70B (FREE)",
    "arliai/qwq-32b-rpr-v1:free": "QwQ 32B RpR v1 (FREE)",
    "agentica-org/deepcoder-14b-preview:free": "Deepcoder 14B Preview (FREE)",
    "moonshotai/kimi-vl-a3b-thinking:free": "Kimi VL A3B Thinking (FREE)",
    "moonshotai/moonlight-16b-a3b-instruct:free": "Moonlight 16B A3B Instruct (FREE)",
    # NVIDIA
    "nvidia/llama-3.3-nemotron-super-49b-v1:free": "Llama 3.3 Nemotron Super 49B v1 (FREE)",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1:free": "Llama 3.1 Nemotron Ultra 253B v1 (FREE)",
    # Other Specialized
    "featherless/qwerky-72b:free": "Qwerky 72B (FREE)",
    "open-r1/olympiccoder-32b:free": "OlympicCoder 32B (FREE)",
    "rekaai/flash-3:free": "Reka Flash 3 (FREE)",
    "cognitivecomputations/dolphin3.0-r1-mistral-24b:free": "Dolphin3.0 R1 Mistral 24B (FREE)",
    "cognitivecomputations/dolphin3.0-mistral-24b:free": "Dolphin3.0 Mistral 24B (FREE)",
    "tngtech/deepseek-r1t-chimera:free": "TNG DeepSeek R1T Chimera (FREE)",
}

# ---- API KEY VALIDATION ----
def validate_api_key(key):
    """Validate the API key format and basic structure"""
    if not key:
        return False, "API key is empty or None"
    
    # Remove any whitespace
    key = key.strip()
    
    # Basic validation - OpenRouter keys typically start with "sk-or-"
    if not key.startswith(('sk-or-', 'sk-')):
        return False, "API key doesn't appear to be a valid OpenRouter key format"
    
    if len(key) < 20:
        return False, "API key appears to be too short"
    
    return True, "API key format looks valid"

# ---- IMPROVED API CALL FUNCTION ----
def call_model_api(model_id, messages, api_key, temperature, max_tokens, timeout=60, system_message=""):
    """Enhanced API call with better error handling and logging"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    # Clean and validate API key one more time
    if not api_key or not api_key.strip():
        return "❌ API key is empty. Please check your configuration."
    
    api_key = api_key.strip()
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://streamlit.app",  # For OpenRouter analytics
        "X-Title": "Multi-Model Chatbot"
    }
    
    # Add system message if provided
    if system_message.strip():
        messages_with_system = [{"role": "system", "content": system_message.strip()}] + messages
    else:
        messages_with_system = messages
    
    data = {
        "model": model_id,
        "messages": messages_with_system,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=timeout)
        
        # Enhanced error handling with specific messages
        if response.status_code == 401:
            error_detail = ""
            try:
                error_json = response.json()
                if 'error' in error_json:
                    error_detail = f": {error_json['error'].get('message', 'Unknown auth error')}"
            except:
                pass
            return f"🔐 Authentication failed{error_detail}. Please check your API key at https://openrouter.ai/keys"
        
        elif response.status_code == 429:
            return "⏳ Rate limit exceeded. Please wait a moment and try again."
        elif response.status_code == 402:
            return "💳 Insufficient credits. Please check your OpenRouter account."
        elif response.status_code == 400:
            try:
                error_json = response.json()
                error_msg = error_json.get('error', {}).get('message', 'Bad request')
                return f"❌ Bad request: {error_msg}"
            except:
                return "❌ Bad request. Please check your input."
        elif response.status_code == 404:
            return f"❌ Model '{model_id}' not found. It may be unavailable or discontinued."
        elif response.status_code != 200:
            try:
                error_json = response.json()
                error_msg = error_json.get('error', {}).get('message', f'HTTP {response.status_code}')
                return f"❌ {error_msg}"
            except:
                return f"❌ HTTP {response.status_code}: {response.text[:100]}"
        
        result = response.json()
        
        if 'choices' not in result or not result['choices']:
            return "❌ No response generated."
        
        content = result['choices'][0]['message']['content']
        
        # Add usage info if available
        if 'usage' in result:
            usage = result['usage']
            content += f"\n\n*Tokens: {usage.get('total_tokens', 'N/A')}*"
        
        return content
        
    except requests.exceptions.Timeout:
        return f"⏰ Request timed out after {timeout} seconds."
    except requests.exceptions.ConnectionError:
        return "🌐 Connection error. Please check your internet connection."
    except requests.exceptions.RequestException as e:
        return f"❌ Request error: {str(e)}"
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"

def call_models_parallel(models, messages, api_key, temperature, max_tokens, timeout, system_message=""):
    """Call multiple models in parallel for faster responses"""
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {}
        
        for model in models:
            future = executor.submit(
                call_model_api, model, messages, api_key, temperature, max_tokens, timeout, system_message
            )
            futures[model] = future
        
        results = {}
        for model, future in futures.items():
            try:
                results[model] = future.result(timeout=timeout + 10)
            except Exception as e:
                results[model] = f"❌ Error: {str(e)}"
        
        return results

# ---- MODEL BROWSER FUNCTIONS ----
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
    }
    
    for model_id, display_name in MODEL_OPTIONS.items():
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
        'experimental': []
    }
    
    for model_id, display_name in MODEL_OPTIONS.items():
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
    total_models = len(MODEL_OPTIONS)
    free_models = len([m for m in MODEL_OPTIONS.keys() if ':free' in m])
    
    # Provider distribution
    providers = {}
    for model_id in MODEL_OPTIONS.keys():
        provider = model_id.split('/')[0] if '/' in model_id else 'other'
        providers[provider] = providers.get(provider, 0) + 1
    
    return {
        'total': total_models,
        'free': free_models,
        'providers': providers
    }

def render_model_browser():
    """Render the complete model browser interface."""
    st.header("🔍 Model Browser & Explorer")
    st.markdown("Explore all available FREE models with detailed capabilities")
    
    # Model statistics
    stats = get_model_stats()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Models", stats['total'])
    with col2:
        st.metric("FREE Models", stats['free'])
        st.success("No cost!")
    with col3:
        most_models_provider = max(stats['providers'].items(), key=lambda x: x[1])
        st.metric("Top Provider", most_models_provider[0].title())
        st.info(f"{most_models_provider[1]} models")
    with col4:
        st.metric("Providers", len(stats['providers']))
    
    # Provider distribution
    with st.expander("📊 Provider Distribution", expanded=False):
        provider_df = pd.DataFrame(list(stats['providers'].items()), columns=['Provider', 'Models'])
        provider_df = provider_df.sort_values('Models', ascending=False)
        st.dataframe(provider_df, use_container_width=True)
    
    # Quick recommendations
    st.subheader("⭐ Quick Recommendations")
    
    recommendations = {
        "🚀 Fastest": ("meta-llama/llama-3.2-1b-instruct:free", "Smallest, fastest responses"),
        "🧠 Best Reasoning": ("deepseek/deepseek-r1:free", "Advanced reasoning capabilities"),
        "🏆 Most Capable": ("meta-llama/llama-3.3-70b-instruct:free", "Best overall performance"),
        "🔬 Latest": ("google/gemini-2.0-flash-experimental:free", "Cutting-edge research"),
        "💻 Code Expert": ("qwen/qwen2.5-coder-32b-instruct:free", "Specialized for programming"),
        "👁️ Vision": ("meta-llama/llama-3.2-11b-vision-instruct:free", "Image understanding"),
        "🦣 Largest": ("meta-llama/llama-3.1-405b-base:free", "Most parameters"),
        "🌐 Multilingual": ("qwen/qwen2.5-72b-instruct:free", "Multiple languages")
    }
    
    cols = st.columns(4)
    for i, (rec_name, (model_id, description)) in enumerate(recommendations.items()):
        with cols[i % 4]:
            display_name = MODEL_OPTIONS.get(model_id, model_id)
            st.write(f"**{rec_name}**")
            st.caption(description)
            if st.button(f"Select", key=f"rec_{i}", use_container_width=True):
                if 'selected_models_browser' not in st.session_state:
                    st.session_state.selected_models_browser = []
                if model_id not in st.session_state.selected_models_browser:
                    st.session_state.selected_models_browser.append(model_id)
                    st.success(f"✅ Added {display_name.split(' (')[0]}")
                else:
                    st.info("Already selected!")
    
    # Search functionality
    st.subheader("🔍 Search Models")
    search_term = st.text_input("Search by name or capability", placeholder="e.g., reasoning, vision, llama, deepseek")
    
    if search_term:
        matching_models = []
        search_lower = search_term.lower()
        
        for model_id, display_name in MODEL_OPTIONS.items():
            if (search_lower in model_id.lower() or 
                search_lower in display_name.lower()):
                matching_models.append((model_id, display_name))
        
        st.write(f"Found {len(matching_models)} matching models:")
        
        for model_id, display_name in matching_models:
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                st.write(f"**{display_name}**")
                st.caption(f"`{model_id}`")
            with col2:
                st.success("FREE")
            with col3:
                if st.button(f"Add", key=f"search_{model_id}"):
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
                for model_id, display_name in models:
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.write(f"**{display_name}**")
                        st.caption(f"Model ID: `{model_id}`")
                    with col2:
                        if st.button(f"Add", key=f"cat_{model_id}"):
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
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.write(f"**{display_name}**")
                        st.caption(f"`{model_id}`")
                    with col2:
                        if st.button(f"Add", key=f"cap_{model_id}"):
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
                st.write(f"• **{display_name}**")
        
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
                size = 'Unknown'
                if any(x in display_name.lower() for x in ['1b', '3b', '7b', '8b']):
                    size = 'Small (≤8B)'
                elif any(x in display_name.lower() for x in ['11b', '12b', '14b', '24b', '27b', '30b', '32b']):
                    size = 'Medium (9B-32B)'
                elif any(x in display_name.lower() for x in ['70b', '72b']):
                    size = 'Large (70B+)'
                elif any(x in display_name.lower() for x in ['235b', '253b', '405b']):
                    size = 'Ultra Large (200B+)'
                
                model_data.append({
                    'Model ID': model_id,
                    'Display Name': display_name,
                    'Provider': provider.title(),
                    'Cost': 'FREE',
                    'Size Category': size
                })
            
            df = pd.DataFrame(model_data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="openrouter_models.csv",
                mime="text/csv"
            )

def chat_interface():
    """Main chat interface function"""
    st.title("🧠 Multi-Model AI Chatbot (OpenRouter)")

    # Check for transferred models from browser
    if 'transfer_models' in st.session_state:
        st.success(f"✅ {len(st.session_state.transfer_models)} models transferred from browser!")
        # Auto-select the transferred models
        default_models = st.session_state.transfer_models
        del st.session_state.transfer_models  # Clean up
    else:
        default_models = ["deepseek/deepseek-v3-base:free"]

    # Custom CSS for better styling
    st.markdown("""
    <style>
        .model-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            text-align: center;
            font-weight: bold;
        }
        .status-box {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            margin: 5px 0;
            background-color: #f8f9fa;
        }
        .error-box {
            border: 1px solid #dc3545;
            background-color: #f8d7da;
            color: #721c24;
        }
        .success-box {
            border: 1px solid #28a745;
            background-color: #d4edda;
            color: #155724;
        }
    </style>
    """, unsafe_allow_html=True)

    # ---- SIDEBAR ----
    with st.sidebar:
        st.header("🔐 API Access")
        
        # Check if environment API key exists
        env_api_key = api_key
        
        # API Key source selection
        api_source = st.radio(
            "Choose API Key Source:",
            ["Environment Variable", "Manual Input"],
            help="Select how you want to provide your OpenRouter API key"
        )
        
        final_api_key = None
        
        if api_source == "Environment Variable":
            if env_api_key:
                env_api_key = env_api_key.strip()
                is_valid, message = validate_api_key(env_api_key)
                
                if is_valid:
                    # Show partial key for verification
                    masked_key = env_api_key[:8] + "..." + env_api_key[-4:] if len(env_api_key) > 12 else "sk-..."
                    st.success(f"✅ Environment API key loaded: `{masked_key}`")
                    final_api_key = env_api_key
                    
                    # Option to view/edit the key
                    with st.expander("🔍 View/Edit Environment Key", expanded=False):
                        edited_key = st.text_input(
                            "Current environment key:",
                            value=env_api_key,
                            type="password",
                            help="Edit if needed"
                        )
                        if edited_key != env_api_key:
                            is_valid_edited, message_edited = validate_api_key(edited_key)
                            if is_valid_edited:
                                st.info("✅ Using edited key")
                                final_api_key = edited_key
                            else:
                                st.error(f"❌ Edited key invalid: {message_edited}")
                else:
                    st.error(f"❌ Environment API key issue: {message}")
                    st.info("💡 Switch to 'Manual Input' or fix your .env file")
            else:
                st.warning("⚠️ No OPENROUTER_API_KEY found in environment")
                st.info("💡 Create a `.env` file with: `OPENROUTER_API_KEY=your-key-here`")
                st.info("💡 Or switch to 'Manual Input' below")
        
        elif api_source == "Manual Input":
            st.info("🔑 Enter your OpenRouter API key manually")
            
            manual_key = st.text_input(
                "API Key:",
                type="password",
                help="Get your key from https://openrouter.ai/keys",
                placeholder="sk-or-v1-..."
            )
            
            if manual_key:
                manual_key = manual_key.strip()
                is_valid, message = validate_api_key(manual_key)
                if is_valid:
                    masked_key = manual_key[:8] + "..." + manual_key[-4:] if len(manual_key) > 12 else "sk-..."
                    st.success(f"✅ Manual API key validated: `{masked_key}`")
                    final_api_key = manual_key
                else:
                    st.error(f"❌ {message}")
            else:
                st.warning("⚠️ Please enter your API key above")
        
        # Final validation and assignment
        if final_api_key:
            current_api_key = final_api_key
            
            # Quick reference links
            with st.expander("🔗 Quick Links", expanded=False):
                st.markdown("""
                - [Get API Key](https://openrouter.ai/keys) 🗝️
                - [View Usage](https://openrouter.ai/usage) 📊  
                - [Check Credits](https://openrouter.ai/credits) 💳
                - [Documentation](https://openrouter.ai/docs) 📚
                """)
        else:
            st.error("❌ No valid API key available")
            st.stop()

        st.divider()
        st.header("🔧 Connection Test")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧪 Test API Connection", help="Test if your API key works", use_container_width=True):
                with st.spinner("Testing connection..."):
                    # Use a simple test model for the connection test
                    test_model = "deepseek/deepseek-v3-base:free"
                    test_response = call_model_api(
                        test_model,
                        [{"role": "user", "content": "Hi"}],
                        current_api_key,
                        0.1,
                        10,
                        30,
                        ""  # No system message for test
                    )
                    if test_response.startswith("🔐") or test_response.startswith("❌"):
                        st.error(f"❌ Connection failed")
                        st.error(test_response)
                    else:
                        st.success("✅ API connection successful!")
                        st.info(f"Test response: {test_response[:100]}...")
        
        with col2:
            if st.button("📋 Copy API Setup", help="Copy .env file format", use_container_width=True):
                if current_api_key:
                    env_format = f"OPENROUTER_API_KEY={current_api_key}"
                    st.code(env_format, language="bash")
                    st.info("📝 Copy this to your .env file")

        st.divider()
        st.header("🤖 Model Selection")
        selected_models = st.multiselect(
            "Choose one or more models to compare:",
            options=list(MODEL_OPTIONS.keys()),
            format_func=lambda x: MODEL_OPTIONS[x],
            default=default_models,
            help="Select multiple models to compare their responses."
        )

        # Quick model selection buttons
        st.subheader("⚡ Quick Select")
        quick_models = {
            "🧠 Reasoning": "deepseek/deepseek-r1:free",
            "🏆 Best Overall": "meta-llama/llama-3.3-70b-instruct:free", 
            "🚀 Fastest": "meta-llama/llama-3.2-1b-instruct:free",
            "💻 Coding": "qwen/qwen2.5-coder-32b-instruct:free"
        }
        
        cols = st.columns(2)
        for i, (name, model_id) in enumerate(quick_models.items()):
            with cols[i % 2]:
                if st.button(name, key=f"quick_{i}", use_container_width=True):
                    if model_id not in selected_models:
                        selected_models.append(model_id)
                        st.rerun()

        # Layout options
        st.subheader("📱 Layout")
        if len(selected_models) > 2:
            layout_mode = st.radio(
                "Choose layout for multiple models:",
                ["Tabs", "Columns", "Stacked"],
                help="Tabs are better for 3+ models"
            )
        else:
            layout_mode = "Columns"

        st.divider()
        st.header("💬 Conversation")
        system_message = st.text_area(
            "System Message (Optional):",
            placeholder="You are a helpful assistant...",
            help="Set the behavior/personality of the AI models"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear All", help="Clear all conversations"):
                st.session_state.chat_history = {}
                st.rerun()
        
        with col2:
            if st.button("📋 Export", help="Copy conversation to clipboard"):
                if "chat_history" in st.session_state:
                    # Create export text
                    export_text = "# AI Chatbot Conversation Export\n\n"
                    for model in selected_models:
                        if model in st.session_state.chat_history:
                            export_text += f"## {MODEL_OPTIONS[model]}\n\n"
                            for msg in st.session_state.chat_history[model]:
                                role = "**User**" if msg["role"] == "user" else "**Assistant**"
                                export_text += f"{role}: {msg['content']}\n\n"
                            export_text += "---\n\n"
                    st.text_area("Copy this text:", export_text, height=100)

        st.divider()
        with st.expander("⚙️ Advanced Settings", expanded=False):
            temperature = st.slider(
                "Temperature",
                0.0, 1.5, 0.7, 0.05,
                help="Higher = more creative, lower = more focused."
            )
            max_tokens = st.slider(
                "Max tokens",
                16, 2048, 512, 16,
                help="Maximum length of the model's response."
            )
            timeout = st.slider(
                "Timeout (seconds)",
                10, 120, 60, 5,
                help="Request timeout for API calls."
            )
            
            if st.button("Restore Defaults"):
                st.session_state.temperature = 0.7
                st.session_state.max_tokens = 512
                st.session_state.timeout = 60
                st.rerun()

        st.markdown("---")
        st.caption("Made with ❤️ using Streamlit and OpenRouter")

    if not selected_models:
        st.warning("Please select at least one model to continue.")
        st.info("💡 Use the Model Browser tab to explore and select models!")
        st.stop()

    # Final API key check before proceeding
    if not current_api_key or not current_api_key.strip():
        st.error("❌ No valid API key available. Please configure your API key in the sidebar.")
        st.stop()

    # ---- SESSION STATE FOR CHAT ----
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}

    if "model_status" not in st.session_state:
        st.session_state.model_status = {}

    for model in selected_models:
        if model not in st.session_state.chat_history:
            st.session_state.chat_history[model] = []
        if model not in st.session_state.model_status:
            st.session_state.model_status[model] = "Ready"

    # ---- MAIN CHAT INTERFACE ----
    user_input = st.chat_input("Type your message and press Enter...")

    if user_input:
        # Add user message to all selected models
        for model in selected_models:
            st.session_state.chat_history[model].append({"role": "user", "content": user_input})
        
        # Show progress
        progress_container = st.container()
        with progress_container:
            st.info("🤖 Getting responses from selected models...")
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # Update status for all models
        for model in selected_models:
            st.session_state.model_status[model] = "Generating..."
        
        if len(selected_models) == 1:
            # Single model - simple call
            model = selected_models[0]
            status_text.text(f"Calling {MODEL_OPTIONS[model]}...")
            response = call_model_api(
                model,
                st.session_state.chat_history[model],
                current_api_key,
                temperature,
                max_tokens,
                timeout,
                system_message
            )
            st.session_state.chat_history[model].append({"role": "assistant", "content": response})
            st.session_state.model_status[model] = "Complete"
            progress_bar.progress(1.0)
        else:
            # Multiple models - parallel calls
            status_text.text("Calling multiple models in parallel...")
            
            # Get messages for parallel call (excluding the system message part)
            messages_for_api = st.session_state.chat_history[selected_models[0]]
            
            results = call_models_parallel(
                selected_models, messages_for_api, current_api_key, temperature, max_tokens, timeout, system_message
            )
            
            # Add responses to chat history
            for i, model in enumerate(selected_models):
                st.session_state.chat_history[model].append({
                    "role": "assistant", 
                    "content": results[model]
                })
                st.session_state.model_status[model] = "Complete"
                progress_bar.progress((i + 1) / len(selected_models))
        
        # Clear progress indicators
        progress_container.empty()
        st.rerun()

    # ---- DISPLAY CHAT BASED ON LAYOUT ----
    if layout_mode == "Tabs" and len(selected_models) > 1:
        # Tab layout for better readability with many models
        tabs = st.tabs([MODEL_OPTIONS[model] for model in selected_models])
        
        for idx, model in enumerate(selected_models):
            with tabs[idx]:
                # Model status
                status = st.session_state.model_status.get(model, "Ready")
                if status == "Generating...":
                    st.info("🤖 Generating response...")
                elif status == "Complete":
                    st.success("✅ Response ready")
                
                # Chat history
                chat_container = st.container()
                with chat_container:
                    for msg in st.session_state.chat_history.get(model, []):
                        if msg["role"] == "user":
                            st.chat_message("user").markdown(msg["content"])
                        else:
                            st.chat_message("assistant").markdown(msg["content"])
                
                # Individual model controls
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Clear {MODEL_OPTIONS[model][:20]}...", key=f"clear_{model}"):
                        st.session_state.chat_history[model] = []
                        st.rerun()

    elif layout_mode == "Stacked":
        # Stacked layout - one model per row
        for model in selected_models:
            st.markdown(f'<div class="model-header">{MODEL_OPTIONS[model]}</div>', unsafe_allow_html=True)
            
            # Status indicator
            status = st.session_state.model_status.get(model, "Ready")
            if status == "Generating...":
                st.info("🤖 Generating response...")
            
            # Chat messages
            for msg in st.session_state.chat_history.get(model, []):
                if msg["role"] == "user":
                    st.chat_message("user").markdown(msg["content"])
                else:
                    st.chat_message("assistant").markdown(msg["content"])
            
            st.markdown("---")

    else:
        # Column layout (default for 1-2 models)
        cols = st.columns(len(selected_models))
        
        for idx, model in enumerate(selected_models):
            with cols[idx]:
                st.markdown(f'<div class="model-header">{MODEL_OPTIONS[model]}</div>', unsafe_allow_html=True)
                
                # Status indicator
                status = st.session_state.model_status.get(model, "Ready")
                if status == "Generating...":
                    st.info("🤖 Generating...")
                
                # Chat history
                for msg in st.session_state.chat_history.get(model, []):
                    if msg["role"] == "user":
                        st.chat_message("user").markdown(msg["content"])
                    else:
                        st.chat_message("assistant").markdown(msg["content"])
                
                # Individual clear button
                if st.button(f"Clear", key=f"clear_{model}", help=f"Clear {MODEL_OPTIONS[model]}"):
                    st.session_state.chat_history[model] = []
                    st.rerun()

    # ---- FOOTER INFO ----
    if st.session_state.chat_history:
        with st.expander("📊 Session Info"):
            total_messages = sum(len(history) for history in st.session_state.chat_history.values())
            st.write(f"**Total messages:** {total_messages}")
            st.write(f"**Active models:** {len(selected_models)}")
            
            for model in selected_models:
                model_messages = len(st.session_state.chat_history.get(model, []))
                st.write(f"- {MODEL_OPTIONS[model]}: {model_messages} messages")

# ---- MAIN APP ----
st.set_page_config(page_title="Multi-Model AI Chatbot", layout="wide", page_icon="🧠")

# Main navigation
tab1, tab2 = st.tabs(["💬 Chat", "🔍 Model Browser"])

with tab1:
    # Main chat interface
    chat_interface()

with tab2:
    # Model browser interface
    render_model_browser()

def chat_interface():
    """Main chat interface function"""
    st.title("🧠 Multi-Model AI Chatbot (OpenRouter)")

def chat_interface():
    """Main chat interface function"""
    st.title("🧠 Multi-Model AI Chatbot (OpenRouter)")

    # Check for transferred models from browser
    if 'transfer_models' in st.session_state:
        st.success(f"✅ {len(st.session_state.transfer_models)} models transferred from browser!")
        # Auto-select the transferred models
        default_models = st.session_state.transfer_models
        del st.session_state.transfer_models  # Clean up
    else:
        default_models = ["deepseek/deepseek-v3-base:free"]

    # Custom CSS for better styling
    st.markdown("""
    <style>
        .model-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            text-align: center;
            font-weight: bold;
        }
        .status-box {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            margin: 5px 0;
            background-color: #f8f9fa;
        }
        .error-box {
            border: 1px solid #dc3545;
            background-color: #f8d7da;
            color: #721c24;
        }
        .success-box {
            border: 1px solid #28a745;
            background-color: #d4edda;
            color: #155724;
        }
    </style>
    """, unsafe_allow_html=True)

    # ---- SIDEBAR ----
    with st.sidebar:
        st.header("🔐 API Access")
        
        # Check if environment API key exists
        env_api_key = api_key
        
        # API Key source selection
        api_source = st.radio(
            "Choose API Key Source:",
            ["Environment Variable", "Manual Input"],
            help="Select how you want to provide your OpenRouter API key"
        )
        
        final_api_key = None
        
        if api_source == "Environment Variable":
            if env_api_key:
                env_api_key = env_api_key.strip()
                is_valid, message = validate_api_key(env_api_key)
                
                if is_valid:
                    # Show partial key for verification
                    masked_key = env_api_key[:8] + "..." + env_api_key[-4:] if len(env_api_key) > 12 else "sk-..."
                    st.success(f"✅ Environment API key loaded: `{masked_key}`")
                    final_api_key = env_api_key
                    
                    # Option to view/edit the key
                    with st.expander("🔍 View/Edit Environment Key", expanded=False):
                        edited_key = st.text_input(
                            "Current environment key:",
                            value=env_api_key,
                            type="password",
                            help="Edit if needed"
                        )
                        if edited_key != env_api_key:
                            is_valid_edited, message_edited = validate_api_key(edited_key)
                            if is_valid_edited:
                                st.info("✅ Using edited key")
                                final_api_key = edited_key
                            else:
                                st.error(f"❌ Edited key invalid: {message_edited}")
                else:
                    st.error(f"❌ Environment API key issue: {message}")
                    st.info("💡 Switch to 'Manual Input' or fix your .env file")
            else:
                st.warning("⚠️ No OPENROUTER_API_KEY found in environment")
                st.info("💡 Create a `.env` file with: `OPENROUTER_API_KEY=your-key-here`")
                st.info("💡 Or switch to 'Manual Input' below")
        
        elif api_source == "Manual Input":
            st.info("🔑 Enter your OpenRouter API key manually")
            
            manual_key = st.text_input(
                "API Key:",
                type="password",
                help="Get your key from https://openrouter.ai/keys",
                placeholder="sk-or-v1-..."
            )
            
            if manual_key:
                manual_key = manual_key.strip()
                is_valid, message = validate_api_key(manual_key)
                if is_valid:
                    masked_key = manual_key[:8] + "..." + manual_key[-4:] if len(manual_key) > 12 else "sk-..."
                    st.success(f"✅ Manual API key validated: `{masked_key}`")
                    final_api_key = manual_key
                else:
                    st.error(f"❌ {message}")
            else:
                st.warning("⚠️ Please enter your API key above")
        
        # Final validation and assignment
        if final_api_key:
            current_api_key = final_api_key
            
            # Quick reference links
            with st.expander("🔗 Quick Links", expanded=False):
                st.markdown("""
                - [Get API Key](https://openrouter.ai/keys) 🗝️
                - [View Usage](https://openrouter.ai/usage) 📊  
                - [Check Credits](https://openrouter.ai/credits) 💳
                - [Documentation](https://openrouter.ai/docs) 📚
                """)
        else:
            st.error("❌ No valid API key available")
            st.stop()

        st.divider()
        st.header("🔧 Connection Test")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧪 Test API Connection", help="Test if your API key works", use_container_width=True):
                with st.spinner("Testing connection..."):
                    # Use a simple test model for the connection test
                    test_model = "deepseek/deepseek-v3-base:free"
                    test_response = call_model_api(
                        test_model,
                        [{"role": "user", "content": "Hi"}],
                        current_api_key,
                        0.1,
                        10,
                        30,
                        ""  # No system message for test
                    )
                    if test_response.startswith("🔐") or test_response.startswith("❌"):
                        st.error(f"❌ Connection failed")
                        st.error(test_response)
                    else:
                        st.success("✅ API connection successful!")
                        st.info(f"Test response: {test_response[:100]}...")
        
        with col2:
            if st.button("📋 Copy API Setup", help="Copy .env file format", use_container_width=True):
                if current_api_key:
                    env_format = f"OPENROUTER_API_KEY={current_api_key}"
                    st.code(env_format, language="bash")
                    st.info("📝 Copy this to your .env file")

        st.divider()
        st.header("🤖 Model Selection")
        selected_models = st.multiselect(
            "Choose one or more models to compare:",
            options=list(MODEL_OPTIONS.keys()),
            format_func=lambda x: MODEL_OPTIONS[x],
            default=default_models,
            help="Select multiple models to compare their responses."
        )

        # Quick model selection buttons
        st.subheader("⚡ Quick Select")
        quick_models = {
            "🧠 Reasoning": "deepseek/deepseek-r1:free",
            "🏆 Best Overall": "meta-llama/llama-3.3-70b-instruct:free", 
            "🚀 Fastest": "meta-llama/llama-3.2-1b-instruct:free",
            "💻 Coding": "qwen/qwen2.5-coder-32b-instruct:free"
        }
        
        cols = st.columns(2)
        for i, (name, model_id) in enumerate(quick_models.items()):
            with cols[i % 2]:
                if st.button(name, key=f"quick_{i}", use_container_width=True):
                    if model_id not in selected_models:
                        selected_models.append(model_id)
                        st.rerun()

        # Layout options
        st.subheader("📱 Layout")
        if len(selected_models) > 2:
            layout_mode = st.radio(
                "Choose layout for multiple models:",
                ["Tabs", "Columns", "Stacked"],
                help="Tabs are better for 3+ models"
            )
        else:
            layout_mode = "Columns"

        st.divider()
        st.header("💬 Conversation")
        system_message = st.text_area(
            "System Message (Optional):",
            placeholder="You are a helpful assistant...",
            help="Set the behavior/personality of the AI models"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear All", help="Clear all conversations"):
                st.session_state.chat_history = {}
                st.rerun()
        
        with col2:
            if st.button("📋 Export", help="Copy conversation to clipboard"):
                if "chat_history" in st.session_state:
                    # Create export text
                    export_text = "# AI Chatbot Conversation Export\n\n"
                    for model in selected_models:
                        if model in st.session_state.chat_history:
                            export_text += f"## {MODEL_OPTIONS[model]}\n\n"
                            for msg in st.session_state.chat_history[model]:
                                role = "**User**" if msg["role"] == "user" else "**Assistant**"
                                export_text += f"{role}: {msg['content']}\n\n"
                            export_text += "---\n\n"
                    st.text_area("Copy this text:", export_text, height=100)

        st.divider()
        with st.expander("⚙️ Advanced Settings", expanded=False):
            temperature = st.slider(
                "Temperature",
                0.0, 1.5, 0.7, 0.05,
                help="Higher = more creative, lower = more focused."
            )
            max_tokens = st.slider(
                "Max tokens",
                16, 2048, 512, 16,
                help="Maximum length of the model's response."
            )
            timeout = st.slider(
                "Timeout (seconds)",
                10, 120, 60, 5,
                help="Request timeout for API calls."
            )
            
            if st.button("Restore Defaults"):
                st.session_state.temperature = 0.7
                st.session_state.max_tokens = 512
                st.session_state.timeout = 60
                st.rerun()

        st.markdown("---")
        st.caption("Made with ❤️ using Streamlit and OpenRouter")

    if not selected_models:
        st.warning("Please select at least one model to continue.")
        st.info("💡 Use the Model Browser tab to explore and select models!")
        st.stop()

    # Final API key check before proceeding
    if not current_api_key or not current_api_key.strip():
        st.error("❌ No valid API key available. Please configure your API key in the sidebar.")
        st.stop()

    # ---- SESSION STATE FOR CHAT ----
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}

    if "model_status" not in st.session_state:
        st.session_state.model_status = {}

    for model in selected_models:
        if model not in st.session_state.chat_history:
            st.session_state.chat_history[model] = []
        if model not in st.session_state.model_status:
            st.session_state.model_status[model] = "Ready"

    # ---- MAIN CHAT INTERFACE ----
    user_input = st.chat_input("Type your message and press Enter...")

    if user_input:
        # Add user message to all selected models
        for model in selected_models:
            st.session_state.chat_history[model].append({"role": "user", "content": user_input})
        
        # Show progress
        progress_container = st.container()
        with progress_container:
            st.info("🤖 Getting responses from selected models...")
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # Update status for all models
        for model in selected_models:
            st.session_state.model_status[model] = "Generating..."
        
        if len(selected_models) == 1:
            # Single model - simple call
            model = selected_models[0]
            status_text.text(f"Calling {MODEL_OPTIONS[model]}...")
            response = call_model_api(
                model,
                st.session_state.chat_history[model],
                current_api_key,
                temperature,
                max_tokens,
                timeout,
                system_message
            )
            st.session_state.chat_history[model].append({"role": "assistant", "content": response})
            st.session_state.model_status[model] = "Complete"
            progress_bar.progress(1.0)
        else:
            # Multiple models - parallel calls
            status_text.text("Calling multiple models in parallel...")
            
            # Get messages for parallel call (excluding the system message part)
            messages_for_api = st.session_state.chat_history[selected_models[0]]
            
            results = call_models_parallel(
                selected_models, messages_for_api, current_api_key, temperature, max_tokens, timeout, system_message
            )
            
            # Add responses to chat history
            for i, model in enumerate(selected_models):
                st.session_state.chat_history[model].append({
                    "role": "assistant", 
                    "content": results[model]
                })
                st.session_state.model_status[model] = "Complete"
                progress_bar.progress((i + 1) / len(selected_models))
        
        # Clear progress indicators
        progress_container.empty()
        st.rerun()

    # ---- DISPLAY CHAT BASED ON LAYOUT ----
    if layout_mode == "Tabs" and len(selected_models) > 1:
        # Tab layout for better readability with many models
        tabs = st.tabs([MODEL_OPTIONS[model] for model in selected_models])
        
        for idx, model in enumerate(selected_models):
            with tabs[idx]:
                # Model status
                status = st.session_state.model_status.get(model, "Ready")
                if status == "Generating...":
                    st.info("🤖 Generating response...")
                elif status == "Complete":
                    st.success("✅ Response ready")
                
                # Chat history
                chat_container = st.container()
                with chat_container:
                    for msg in st.session_state.chat_history.get(model, []):
                        if msg["role"] == "user":
                            st.chat_message("user").markdown(msg["content"])
                        else:
                            st.chat_message("assistant").markdown(msg["content"])
                
                # Individual model controls
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Clear {MODEL_OPTIONS[model][:20]}...", key=f"clear_{model}"):
                        st.session_state.chat_history[model] = []
                        st.rerun()

    elif layout_mode == "Stacked":
        # Stacked layout - one model per row
        for model in selected_models:
            st.markdown(f'<div class="model-header">{MODEL_OPTIONS[model]}</div>', unsafe_allow_html=True)
            
            # Status indicator
            status = st.session_state.model_status.get(model, "Ready")
            if status == "Generating...":
                st.info("🤖 Generating response...")
            
            # Chat messages
            for msg in st.session_state.chat_history.get(model, []):
                if msg["role"] == "user":
                    st.chat_message("user").markdown(msg["content"])
                else:
                    st.chat_message("assistant").markdown(msg["content"])
            
            st.markdown("---")

    else:
        # Column layout (default for 1-2 models)
        cols = st.columns(len(selected_models))
        
        for idx, model in enumerate(selected_models):
            with cols[idx]:
                st.markdown(f'<div class="model-header">{MODEL_OPTIONS[model]}</div>', unsafe_allow_html=True)
                
                # Status indicator
                status = st.session_state.model_status.get(model, "Ready")
                if status == "Generating...":
                    st.info("🤖 Generating...")
                
                # Chat history
                for msg in st.session_state.chat_history.get(model, []):
                    if msg["role"] == "user":
                        st.chat_message("user").markdown(msg["content"])
                    else:
                        st.chat_message("assistant").markdown(msg["content"])
                
                # Individual clear button
                if st.button(f"Clear", key=f"clear_{model}", help=f"Clear {MODEL_OPTIONS[model]}"):
                    st.session_state.chat_history[model] = []
                    st.rerun()

    # ---- FOOTER INFO ----
    if st.session_state.chat_history:
        with st.expander("📊 Session Info"):
            total_messages = sum(len(history) for history in st.session_state.chat_history.values())
            st.write(f"**Total messages:** {total_messages}")
            st.write(f"**Active models:** {len(selected_models)}")
            
            for model in selected_models:
                model_messages = len(st.session_state.chat_history.get(model, []))
                st.write(f"- {MODEL_OPTIONS[model]}: {model_messages} messages")

# Custom CSS for better styling
st.markdown("""
<style>
    .model-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        text-align: center;
        font-weight: bold;
    }
    .status-box {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        background-color: #f8f9fa;
    }
    .error-box {
        border: 1px solid #dc3545;
        background-color: #f8d7da;
        color: #721c24;
    }
    .success-box {
        border: 1px solid #28a745;
        background-color: #d4edda;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Multi-Model AI Chatbot (OpenRouter)")

# ---- SIDEBAR ----
with st.sidebar:
    st.header("🔐 API Access")
    
    # Check if environment API key exists
    env_api_key = api_key
    
    # API Key source selection
    api_source = st.radio(
        "Choose API Key Source:",
        ["Environment Variable", "Manual Input"],
        help="Select how you want to provide your OpenRouter API key"
    )
    
    final_api_key = None
    
    if api_source == "Environment Variable":
        if env_api_key:
            env_api_key = env_api_key.strip()
            is_valid, message = validate_api_key(env_api_key)
            
            if is_valid:
                # Show partial key for verification
                masked_key = env_api_key[:8] + "..." + env_api_key[-4:] if len(env_api_key) > 12 else "sk-..."
                st.success(f"✅ Environment API key loaded: `{masked_key}`")
                final_api_key = env_api_key
                
                # Option to view/edit the key
                with st.expander("🔍 View/Edit Environment Key", expanded=False):
                    edited_key = st.text_input(
                        "Current environment key:",
                        value=env_api_key,
                        type="password",
                        help="Edit if needed"
                    )
                    if edited_key != env_api_key:
                        is_valid_edited, message_edited = validate_api_key(edited_key)
                        if is_valid_edited:
                            st.info("✅ Using edited key")
                            final_api_key = edited_key
                        else:
                            st.error(f"❌ Edited key invalid: {message_edited}")
            else:
                st.error(f"❌ Environment API key issue: {message}")
                st.info("💡 Switch to 'Manual Input' or fix your .env file")
        else:
            st.warning("⚠️ No OPENROUTER_API_KEY found in environment")
            st.info("💡 Create a `.env` file with: `OPENROUTER_API_KEY=your-key-here`")
            st.info("💡 Or switch to 'Manual Input' below")
    
    elif api_source == "Manual Input":
        st.info("🔑 Enter your OpenRouter API key manually")
        
        manual_key = st.text_input(
            "API Key:",
            type="password",
            help="Get your key from https://openrouter.ai/keys",
            placeholder="sk-or-v1-..."
        )
        
        if manual_key:
            manual_key = manual_key.strip()
            is_valid, message = validate_api_key(manual_key)
            if is_valid:
                masked_key = manual_key[:8] + "..." + manual_key[-4:] if len(manual_key) > 12 else "sk-..."
                st.success(f"✅ Manual API key validated: `{masked_key}`")
                final_api_key = manual_key
            else:
                st.error(f"❌ {message}")
        else:
            st.warning("⚠️ Please enter your API key above")
    
    # Final validation and assignment
    if final_api_key:
        api_key = final_api_key
        
        # Quick reference links
        with st.expander("🔗 Quick Links", expanded=False):
            st.markdown("""
            - [Get API Key](https://openrouter.ai/keys) 🗝️
            - [View Usage](https://openrouter.ai/usage) 📊  
            - [Check Credits](https://openrouter.ai/credits) 💳
            - [Documentation](https://openrouter.ai/docs) 📚
            """)
    else:
        st.error("❌ No valid API key available")
        st.stop()

    st.divider()
    st.header("🔧 Connection Test")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧪 Test API Connection", help="Test if your API key works", use_container_width=True):
            with st.spinner("Testing connection..."):
                # Use a simple test model for the connection test
                test_model = "deepseek/deepseek-v3-base:free"
                test_response = call_model_api(
                    test_model,
                    [{"role": "user", "content": "Hi"}],
                    api_key,
                    0.1,
                    10,
                    30,
                    ""  # No system message for test
                )
                if test_response.startswith("🔐") or test_response.startswith("❌"):
                    st.error(f"❌ Connection failed")
                    st.error(test_response)
                else:
                    st.success("✅ API connection successful!")
                    st.info(f"Test response: {test_response[:100]}...")
    
    with col2:
        if st.button("📋 Copy API Setup", help="Copy .env file format", use_container_width=True):
            if api_key:
                env_format = f"OPENROUTER_API_KEY={api_key}"
                st.code(env_format, language="bash")
                st.info("📝 Copy this to your .env file")

    st.divider()
    st.header("🤖 Model Selection")
    selected_models = st.multiselect(
        "Choose one or more models to compare:",
        options=list(MODEL_OPTIONS.keys()),
        format_func=lambda x: MODEL_OPTIONS[x],
        default=["deepseek/deepseek-v3-base:free"],
        help="Select multiple models to compare their responses."
    )

    # Layout options
    st.subheader("📱 Layout")
    if len(selected_models) > 2:
        layout_mode = st.radio(
            "Choose layout for multiple models:",
            ["Tabs", "Columns", "Stacked"],
            help="Tabs are better for 3+ models"
        )
    else:
        layout_mode = "Columns"

    st.divider()
    st.header("💬 Conversation")
    system_message = st.text_area(
        "System Message (Optional):",
        placeholder="You are a helpful assistant...",
        help="Set the behavior/personality of the AI models"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear All", help="Clear all conversations"):
            st.session_state.chat_history = {}
            st.rerun()
    
    with col2:
        if st.button("📋 Export", help="Copy conversation to clipboard"):
            if "chat_history" in st.session_state:
                # Create export text
                export_text = "# AI Chatbot Conversation Export\n\n"
                for model in selected_models:
                    if model in st.session_state.chat_history:
                        export_text += f"## {MODEL_OPTIONS[model]}\n\n"
                        for msg in st.session_state.chat_history[model]:
                            role = "**User**" if msg["role"] == "user" else "**Assistant**"
                            export_text += f"{role}: {msg['content']}\n\n"
                        export_text += "---\n\n"
                st.text_area("Copy this text:", export_text, height=100)

    st.divider()
    with st.expander("⚙️ Advanced Settings", expanded=False):
        temperature = st.slider(
            "Temperature",
            0.0, 1.5, 0.7, 0.05,
            help="Higher = more creative, lower = more focused."
        )
        max_tokens = st.slider(
            "Max tokens",
            16, 2048, 512, 16,
            help="Maximum length of the model's response."
        )
        timeout = st.slider(
            "Timeout (seconds)",
            10, 120, 60, 5,
            help="Request timeout for API calls."
        )
        
        if st.button("Restore Defaults"):
            st.session_state.temperature = 0.7
            st.session_state.max_tokens = 512
            st.session_state.timeout = 60
            st.rerun()

    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit and OpenRouter")

if not selected_models:
    st.warning("Please select at least one model to continue.")
    st.stop()

# Final API key check before proceeding
if not api_key or not api_key.strip():
    st.error("❌ No valid API key available. Please configure your API key in the sidebar.")
    st.stop()

# ---- SESSION STATE FOR CHAT ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

if "model_status" not in st.session_state:
    st.session_state.model_status = {}

for model in selected_models:
    if model not in st.session_state.chat_history:
        st.session_state.chat_history[model] = []
    if model not in st.session_state.model_status:
        st.session_state.model_status[model] = "Ready"

# ---- MAIN CHAT INTERFACE ----
user_input = st.chat_input("Type your message and press Enter...")

if user_input:
    # Add user message to all selected models
    for model in selected_models:
        st.session_state.chat_history[model].append({"role": "user", "content": user_input})
    
    # Show progress
    progress_container = st.container()
    with progress_container:
        st.info("🤖 Getting responses from selected models...")
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    # Update status for all models
    for model in selected_models:
        st.session_state.model_status[model] = "Generating..."
    
    if len(selected_models) == 1:
        # Single model - simple call
        model = selected_models[0]
        status_text.text(f"Calling {MODEL_OPTIONS[model]}...")
        response = call_model_api(
            model,
            st.session_state.chat_history[model],
            api_key,
            temperature,
            max_tokens,
            timeout,
            system_message
        )
        st.session_state.chat_history[model].append({"role": "assistant", "content": response})
        st.session_state.model_status[model] = "Complete"
        progress_bar.progress(1.0)
    else:
        # Multiple models - parallel calls
        status_text.text("Calling multiple models in parallel...")
        
        # Get messages for parallel call (excluding the system message part)
        messages_for_api = st.session_state.chat_history[selected_models[0]]
        
        results = call_models_parallel(
            selected_models, messages_for_api, api_key, temperature, max_tokens, timeout, system_message
        )
        
        # Add responses to chat history
        for i, model in enumerate(selected_models):
            st.session_state.chat_history[model].append({
                "role": "assistant", 
                "content": results[model]
            })
            st.session_state.model_status[model] = "Complete"
            progress_bar.progress((i + 1) / len(selected_models))
    
    # Clear progress indicators
    progress_container.empty()
    st.rerun()

# ---- DISPLAY CHAT BASED ON LAYOUT ----
if layout_mode == "Tabs" and len(selected_models) > 1:
    # Tab layout for better readability with many models
    tabs = st.tabs([MODEL_OPTIONS[model] for model in selected_models])
    
    for idx, model in enumerate(selected_models):
        with tabs[idx]:
            # Model status
            status = st.session_state.model_status.get(model, "Ready")
            if status == "Generating...":
                st.info("🤖 Generating response...")
            elif status == "Complete":
                st.success("✅ Response ready")
            
            # Chat history
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state.chat_history.get(model, []):
                    if msg["role"] == "user":
                        st.chat_message("user").markdown(msg["content"])
                    else:
                        st.chat_message("assistant").markdown(msg["content"])
            
            # Individual model controls
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Clear {MODEL_OPTIONS[model][:20]}...", key=f"clear_{model}"):
                    st.session_state.chat_history[model] = []
                    st.rerun()

elif layout_mode == "Stacked":
    # Stacked layout - one model per row
    for model in selected_models:
        st.markdown(f'<div class="model-header">{MODEL_OPTIONS[model]}</div>', unsafe_allow_html=True)
        
        # Status indicator
        status = st.session_state.model_status.get(model, "Ready")
        if status == "Generating...":
            st.info("🤖 Generating response...")
        
        # Chat messages
        for msg in st.session_state.chat_history.get(model, []):
            if msg["role"] == "user":
                st.chat_message("user").markdown(msg["content"])
            else:
                st.chat_message("assistant").markdown(msg["content"])
        
        st.markdown("---")

else:
    # Column layout (default for 1-2 models)
    cols = st.columns(len(selected_models))
    
    for idx, model in enumerate(selected_models):
        with cols[idx]:
            st.markdown(f'<div class="model-header">{MODEL_OPTIONS[model]}</div>', unsafe_allow_html=True)
            
            # Status indicator
            status = st.session_state.model_status.get(model, "Ready")
            if status == "Generating...":
                st.info("🤖 Generating...")
            
            # Chat history
            for msg in st.session_state.chat_history.get(model, []):
                if msg["role"] == "user":
                    st.chat_message("user").markdown(msg["content"])
                else:
                    st.chat_message("assistant").markdown(msg["content"])
            
            # Individual clear button
            if st.button(f"Clear", key=f"clear_{model}", help=f"Clear {MODEL_OPTIONS[model]}"):
                st.session_state.chat_history[model] = []
                st.rerun()

# ---- FOOTER INFO ----
if st.session_state.chat_history:
    with st.expander("📊 Session Info"):
        total_messages = sum(len(history) for history in st.session_state.chat_history.values())
        st.write(f"**Total messages:** {total_messages}")
        st.write(f"**Active models:** {len(selected_models)}")
        
        for model in selected_models:
            model_messages = len(st.session_state.chat_history.get(model, []))
            st.write(f"- {MODEL_OPTIONS[model]}: {model_messages} messages")
