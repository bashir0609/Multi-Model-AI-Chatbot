# chat.py - Single model chat interface with persistent API key caching

import os
import streamlit as st
from models import MODEL_OPTIONS, get_cost_info
from api_utils import validate_api_key, call_model_api

# Import persistent cache with better error handling
try:
    from persistent_cache import get_api_cache
    CACHE_AVAILABLE = True
    # Test cache initialization immediately
    try:
        _test_cache = get_api_cache()
        cache_type = type(_test_cache).__name__
        st.write(f"🔧 Cache system: {cache_type}")
    except Exception as cache_init_error:
        st.error(f"⚠️ Cache initialization issue: {cache_init_error}")
        CACHE_AVAILABLE = False
except ImportError:
    CACHE_AVAILABLE = False
    st.warning("⚠️ Persistent caching not available. Install 'cryptography' for API key persistence across restarts.")
except Exception as e:
    CACHE_AVAILABLE = False
    st.error(f"❌ Cache system error: {e}")

# Replace the load_cached_api_key function (around lines 16-27) with this enhanced version:

def load_cached_api_key():
    """Load API key from persistent cache with enhanced error handling"""
    if not CACHE_AVAILABLE:
        st.sidebar.info("ℹ️ Cache not available")
        return None
    
    try:
        cache = get_api_cache()
        cached_data = cache.load_api_key()
        
        if cached_data:
            # Show cache status in sidebar
            method = cached_data.get('method', 'unknown')
            source = cached_data.get('source', 'unknown')
            st.sidebar.success(f"✅ Cache loaded: {method} method, source: {source}")
            
            # Add cache info to sidebar
            with st.sidebar.expander("🔍 Cache Details", expanded=False):
                st.json({
                    "method": method,
                    "source": source,
                    "timestamp": cached_data.get('timestamp'),
                    "key_preview": cached_data['key'][:8] + "..." + cached_data['key'][-4:] if cached_data['key'] else "N/A"
                })
        else:
            st.sidebar.info("ℹ️ No cached API key found")
        
        return cached_data
    except Exception as e:
        st.sidebar.error(f"❌ Cache load error: {e}")
        # Show detailed error in expander
        with st.sidebar.expander("🐛 Cache Error Details", expanded=False):
            import traceback
            st.code(traceback.format_exc())
        return None

def load_cached_api_key():
    """Load API key from persistent cache"""
    if not CACHE_AVAILABLE:
        return None
    
    try:
        cache = get_api_cache()
        cached_data = cache.load_api_key()
        return cached_data
    except Exception as e:
        st.error(f"Error loading cached API key: {e}")
        return None

def save_api_key_to_cache(api_key, source="manual"):
    """Save API key to persistent cache with enhanced error handling"""
    if not CACHE_AVAILABLE:
        st.sidebar.warning("⚠️ Cache not available for saving")
        return False
    
    if not api_key or not api_key.strip():
        st.sidebar.error("❌ Cannot save empty API key")
        return False
    
    try:
        cache = get_api_cache()
        result = cache.save_api_key(api_key.strip(), source)
        
        if result:
            st.sidebar.success(f"✅ API key saved to cache (source: {source})")
            
            # Show cache info after save
            info = cache.get_cache_info() if hasattr(cache, 'get_cache_info') else {}
            if info:
                with st.sidebar.expander("💾 Cache Status", expanded=False):
                    st.json(info)
        else:
            st.sidebar.error("❌ Failed to save API key to cache")
        
        return result
    except Exception as e:
        st.sidebar.error(f"❌ Cache save error: {e}")
        with st.sidebar.expander("🐛 Save Error Details", expanded=False):
            import traceback
            st.code(traceback.format_exc())
        return False

def clear_api_key_cache():
    """Clear persistent API key cache with enhanced feedback"""
    if not CACHE_AVAILABLE:
        st.sidebar.warning("⚠️ Cache not available")
        return False
    
    try:
        cache = get_api_cache()
        result = cache.clear_cache()
        
        if result:
            st.sidebar.success("✅ Cache cleared successfully")
        else:
            st.sidebar.warning("⚠️ Cache clear may have failed")
        
        return result
    except Exception as e:
        st.sidebar.error(f"❌ Cache clear error: {e}")
        with st.sidebar.expander("🐛 Clear Error Details", expanded=False):
            import traceback
            st.code(traceback.format_exc())
        return False

def test_cache_functionality():
    """Test cache functionality and show results"""
    if not CACHE_AVAILABLE:
        st.sidebar.error("❌ Cache not available for testing")
        return False
    
    try:
        cache = get_api_cache()
        test_key = "sk-test-key-12345"
        
        # Test save
        st.sidebar.info("🧪 Testing cache save...")
        save_result = cache.save_api_key(test_key, "test")
        
        if not save_result:
            st.sidebar.error("❌ Cache save test failed")
            return False
        
        # Test load
        st.sidebar.info("🧪 Testing cache load...")
        load_result = cache.load_api_key()
        
        if not load_result or load_result.get('key') != test_key:
            st.sidebar.error("❌ Cache load test failed")
            return False
        
        # Test clear
        st.sidebar.info("🧪 Testing cache clear...")
        clear_result = cache.clear_cache()
        
        if clear_result:
            st.sidebar.success("✅ All cache tests passed!")
            return True
        else:
            st.sidebar.warning("⚠️ Cache clear test failed")
            return False
            
    except Exception as e:
        st.sidebar.error(f"❌ Cache test error: {e}")
        with st.sidebar.expander("🐛 Test Error Details", expanded=False):
            import traceback
            st.code(traceback.format_exc())
        return False

def chat_interface():
    """Main chat interface function - single model only"""
    st.title("🧠 Multi-Model AI Chatbot (OpenRouter)")
    
    # Important instructions
    st.warning("⚠️ **IMPORTANT**: Many free models change frequently. If you get 'model not found' errors:")
    with st.expander("🔧 How to find working models", expanded=False):
        st.markdown("""
        1. **Go to [OpenRouter Models](https://openrouter.ai/models)** in another tab
        2. **Use the filter** → Set "Prompt pricing" to "FREE" 
        3. **Copy the exact model ID** (like `provider/model-name:free`)
        4. **Add it manually** in the model selection below
        5. **Test it** with the connection test button
        """)
    st.info("💡 **Tip**: The models below are conservative choices that *should* work, but you may need paid credits.")

    # Check for transferred models from browser
    if 'transfer_models' in st.session_state:
        st.success(f"✅ Model transferred from browser!")
        # Auto-select the first transferred model
        default_model = st.session_state.transfer_models[0]
        del st.session_state.transfer_models  # Clean up
    else:
        default_model = "meta-llama/llama-3.1-8b-instruct"

    # Custom CSS for better styling
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
        .chat-container {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # ---- SIDEBAR ----
    with st.sidebar:
        st.header("🔐 API Access")
        
        # Initialize API key session state
        if 'cached_api_key' not in st.session_state:
            st.session_state.cached_api_key = None
        
        # Try to load from persistent cache on first run
        if st.session_state.cached_api_key is None and CACHE_AVAILABLE:
            cached_data = load_cached_api_key()
            if cached_data and cached_data.get('key'):
                st.session_state.cached_api_key = cached_data['key']
                st.session_state.api_key_source = cached_data.get('source', 'cached')
        
        # Check if we have a cached API key
        if st.session_state.cached_api_key:
            # Show current cached API key status
            masked_key = st.session_state.cached_api_key[:8] + "..." + st.session_state.cached_api_key[-4:] if len(st.session_state.cached_api_key) > 12 else "sk-..."
            st.success(f"✅ API Key Active: `{masked_key}`")
            
            # Show cache status
            if CACHE_AVAILABLE:
                cache = get_api_cache()
                if cache.is_cached():
                    st.caption("🔒 Persistently cached - survives app restarts")
                else:
                    st.caption("⚠️ Session only - will be lost on restart")
            else:
                st.caption("⚠️ Session only - install 'cryptography' for persistence")
            
            # Cache control buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("💾 Save", help="Save to persistent cache", use_container_width=True):
                    if CACHE_AVAILABLE:
                        if save_api_key_to_cache(st.session_state.cached_api_key, "manual"):
                            st.success("✅ Saved to persistent cache!")
                        else:
                            st.error("❌ Failed to save to cache")
                    else:
                        st.error("❌ Persistent cache not available")
            
            with col2:
                if st.button("🗑️ Clear", help="Clear API key completely", use_container_width=True):
                    # Clear from both session and persistent cache
                    st.session_state.cached_api_key = None
                    if CACHE_AVAILABLE:
                        if clear_api_key_cache():
                            st.success("✅ Cleared completely!")
                        else:
                            st.warning("⚠️ Session cleared, cache clear failed")
                    else:
                        st.success("✅ Session cleared!")
                    st.rerun()
            
            with col3:
                if st.button("🔄 Change", help="Enter a different API key", use_container_width=True):
                    st.session_state.cached_api_key = None
                    st.rerun()
            
            current_api_key = st.session_state.cached_api_key
            
        else:
            # No cached key - show input options
            st.info("🔑 No API key active. Please configure below:")
            
            # Load API key from environment
            env_api_key = os.getenv("OPENROUTER_API_KEY")
            
            # API Key source selection
            if 'api_key_source' not in st.session_state:
                st.session_state.api_key_source = "Environment Variable"
            
            api_source = st.radio(
                "Choose API Key Source:",
                ["Environment Variable", "Manual Input"],
                index=0 if st.session_state.api_key_source == "Environment Variable" else 1,
                help="Select how you want to provide your OpenRouter API key"
            )
            
            st.session_state.api_key_source = api_source
            
            if api_source == "Environment Variable":
                if env_api_key:
                    env_api_key = env_api_key.strip()
                    is_valid, message = validate_api_key(env_api_key)
                    
                    if is_valid:
                        # Show partial key for verification
                        masked_key = env_api_key[:8] + "..." + env_api_key[-4:] if len(env_api_key) > 12 else "sk-..."
                        st.info(f"🔍 Environment key found: `{masked_key}`")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("✅ Use & Save", help="Use and save to persistent cache", use_container_width=True):
                                st.session_state.cached_api_key = env_api_key
                                if CACHE_AVAILABLE:
                                    if save_api_key_to_cache(env_api_key, "environment"):
                                        st.success("✅ Environment key saved persistently!")
                                    else:
                                        st.warning("✅ Using environment key (save failed)")
                                else:
                                    st.success("✅ Using environment key!")
                                st.rerun()
                        
                        with col2:
                            if st.button("✅ Use Only", help="Use without saving", use_container_width=True):
                                st.session_state.cached_api_key = env_api_key
                                st.success("✅ Environment API key loaded!")
                                st.rerun()
                        
                        # Option to edit the environment key
                        with st.expander("🔍 Edit Environment Key", expanded=False):
                            edited_key = st.text_input(
                                "Edit environment key:",
                                value=env_api_key,
                                type="password",
                                help="Modify if needed",
                                key="edit_env_key"
                            )
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Use Edited & Save", key="use_edited_save"):
                                    if edited_key.strip():
                                        is_valid_edited, message_edited = validate_api_key(edited_key.strip())
                                        if is_valid_edited:
                                            st.session_state.cached_api_key = edited_key.strip()
                                            if CACHE_AVAILABLE:
                                                save_api_key_to_cache(edited_key.strip(), "edited_environment")
                                            st.success("✅ Edited key saved!")
                                            st.rerun()
                                        else:
                                            st.error(f"❌ Invalid: {message_edited}")
                            with col2:
                                if st.button("Use Edited Only", key="use_edited_only"):
                                    if edited_key.strip():
                                        is_valid_edited, message_edited = validate_api_key(edited_key.strip())
                                        if is_valid_edited:
                                            st.session_state.cached_api_key = edited_key.strip()
                                            st.success("✅ Edited key loaded!")
                                            st.rerun()
                                        else:
                                            st.error(f"❌ Invalid: {message_edited}")
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
                    placeholder="sk-or-v1-...",
                    key="manual_api_input"
                )
                
                if manual_key:
                    manual_key = manual_key.strip()
                    is_valid, message = validate_api_key(manual_key)
                    if is_valid:
                        masked_key = manual_key[:8] + "..." + manual_key[-4:] if len(manual_key) > 12 else "sk-..."
                        st.info(f"✅ Key validated: `{masked_key}`")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("💾 Save & Use", help="Save persistently and use", use_container_width=True):
                                st.session_state.cached_api_key = manual_key
                                if CACHE_AVAILABLE:
                                    if save_api_key_to_cache(manual_key, "manual"):
                                        st.success("✅ API key saved persistently!")
                                        st.balloons()
                                    else:
                                        st.warning("✅ Using key (save failed)")
                                else:
                                    st.success("✅ API key cached for session!")
                                st.rerun()
                        
                        with col2:
                            if st.button("🔓 Use Only", help="Use without saving", use_container_width=True):
                                st.session_state.cached_api_key = manual_key
                                st.success("✅ API key loaded!")
                                st.rerun()
                    else:
                        st.error(f"❌ {message}")
                else:
                    st.warning("⚠️ Please enter your API key above")
            
            # No valid key available yet
            current_api_key = None
        
        # Show cache status and instructions
        if CACHE_AVAILABLE:
            cache = get_api_cache()
            with st.expander("💾 Cache Info", expanded=False):
                if cache.is_cached():
                    st.success("✅ API key is persistently cached")
                    st.caption("Your API key will survive app restarts!")
                else:
                    st.info("ℹ️ No persistent cache found")
                    st.caption("Use 'Save & Use' to enable persistence")
                
                st.markdown("""
                **Cache Features:**
                - 🔒 Encrypted storage in your home directory
                - 🔄 Survives app restarts and refreshes
                - 🗑️ Easy to clear when needed
                - 🛡️ Secure file permissions
                """)
        else:
            with st.expander("⚠️ Install for Persistence", expanded=False):
                st.warning("Persistent caching not available")
                st.code("pip install cryptography")
                st.caption("Install the above package to enable API key persistence across app restarts")
        
        # Show quick links if we have a key
        if current_api_key:
            with st.expander("🔗 Quick Links", expanded=False):
                st.markdown("""
                - [Get API Key](https://openrouter.ai/keys) 🗝️
                - [View Usage](https://openrouter.ai/usage) 📊  
                - [Check Credits](https://openrouter.ai/credits) 💳
                - [Documentation](https://openrouter.ai/docs) 📚
                """)
        
        # Stop if no API key
        if not current_api_key:
            st.error("❌ No API key available. Please configure your API key above.")
            st.stop()

        st.divider()
        st.header("🔧 Connection Test")
        
        if st.button("🧪 Test API Connection", help="Test if your API key works", use_container_width=True):
            with st.spinner("Testing connection..."):
                # Use a simple test model for the connection test
                test_model = "meta-llama/llama-3.1-8b-instruct"
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

        st.divider()
        st.header("🤖 Model Selection")
        
        # Initialize session states
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = default_model
        if 'custom_models' not in st.session_state:
            st.session_state.custom_models = {}
        
        # Combine predefined and custom models
        all_models = MODEL_OPTIONS.copy()
        all_models.update(st.session_state.custom_models)
        
        # Single model selection
        selected_model = st.selectbox(
            "Choose a model:",
            options=list(all_models.keys()),
            format_func=lambda x: all_models[x],
            index=list(all_models.keys()).index(st.session_state.selected_model) if st.session_state.selected_model in all_models else 0,
            help="Select one model to chat with.",
            key="model_selector"
        )
        
        # Update session state when selection changes
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model

        # Show model info
        if selected_model in MODEL_OPTIONS:
            cost_info = get_cost_info(selected_model, MODEL_OPTIONS[selected_model])
            if cost_info['type'] == 'free':
                st.success(f"🆓 {cost_info['cost']}")
            else:
                st.info(f"💰 {cost_info['cost']}")
        else:
            # Custom model
            if ':free' in selected_model:
                st.success("🆓 Custom FREE model")
            else:
                st.info("💰 Custom model (check OpenRouter for pricing)")

        # Manual model input
        st.subheader("➕ Add Custom Model")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            manual_model = st.text_input(
                "Model ID:",
                placeholder="e.g., deepseek/deepseek-chat:free",
                help="Enter a model ID from OpenRouter",
                key="custom_model_input"
            )
        
        with col2:
            st.write("")  # Empty space for alignment
            st.write("")  # Empty space for alignment
            add_button = st.button("➕ Add", use_container_width=True, type="primary")
        
        if add_button:
            if manual_model and manual_model.strip():
                custom_id = manual_model.strip()
                # Create a display name for the custom model
                if custom_id not in all_models:
                    # Try to create a nice display name
                    if ':free' in custom_id:
                        display_name = f"{custom_id.replace(':free', '')} (FREE - Custom)"
                    else:
                        display_name = f"{custom_id} (Custom)"
                    
                    # Add to custom models
                    st.session_state.custom_models[custom_id] = display_name
                    st.session_state.selected_model = custom_id
                    st.success(f"✅ Added and selected: {custom_id}")
                    st.rerun()
                else:
                    st.session_state.selected_model = custom_id
                    st.info(f"✅ Switched to existing model: {custom_id}")
                    st.rerun()
            else:
                st.error("Please enter a model ID")
        
        # Show custom models if any
        if st.session_state.custom_models:
            with st.expander("🗂️ Your Custom Models", expanded=False):
                for model_id, display_name in st.session_state.custom_models.items():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{display_name}**")
                        st.caption(f"`{model_id}`")
                    with col2:
                        if st.button("🗑️", key=f"remove_{hash(model_id)}", help=f"Remove {model_id}"):
                            del st.session_state.custom_models[model_id]
                            if st.session_state.selected_model == model_id:
                                # Switch to a default model if current was deleted
                                st.session_state.selected_model = "meta-llama/llama-3.1-8b-instruct"
                            st.rerun()

        # Check OpenRouter button with direct links
        st.subheader("🔍 Find Working Models")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <a href="https://openrouter.ai/models" target="_blank">
                <button style="
                    background: linear-gradient(45deg, #667eea, #764ba2);
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 6px;
                    cursor: pointer;
                    width: 100%;
                    font-weight: bold;
                ">🔍 Check OpenRouter</button>
            </a>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <a href="https://openrouter.ai/models?pricing=free" target="_blank">
                <button style="
                    background: linear-gradient(45deg, #28a745, #20c997);
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 6px;
                    cursor: pointer;
                    width: 100%;
                    font-weight: bold;
                ">🆓 Free Models</button>
            </a>
            """, unsafe_allow_html=True)
        
        st.caption("💡 Click above to open OpenRouter and find current working models")

        # Quick model selection buttons
        st.subheader("⚡ Quick Select")
        quick_models = {
            "🦙 Llama Basic": "meta-llama/llama-3.1-8b-instruct",
            "🌟 Mistral": "mistralai/mistral-7b-instruct", 
            "🧠 DeepSeek": "deepseek/deepseek-chat",
            "🆓 Free Llama": "meta-llama/llama-3.1-8b-instruct:free",
            "🆓 Free Mistral": "mistralai/mistral-7b-instruct:free",
        }
        
        for name, model_id in quick_models.items():
            if st.button(name, key=f"quick_{name}", use_container_width=True):
                st.session_state.selected_model = model_id
                st.rerun()

        st.divider()
        st.header("💬 Conversation")
        system_message = st.text_area(
            "System Message (Optional):",
            placeholder="You are a helpful assistant...",
            help="Set the behavior/personality of the AI model"
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🗑️ Clear Chat", help="Clear conversation history"):
                if 'chat_history' in st.session_state:
                    st.session_state.chat_history = []
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear All", help="Clear chat + API key cache", type="secondary"):
                # Clear everything
                if 'chat_history' in st.session_state:
                    st.session_state.chat_history = []
                if 'cached_api_key' in st.session_state:
                    st.session_state.cached_api_key = None
                st.success("✅ Everything cleared!")
                st.rerun()
        
        with col3:
            if st.button("📋 Export", help="Copy conversation to clipboard"):
                if 'chat_history' in st.session_state and st.session_state.chat_history:
                    # Create export text
                    export_text = f"# AI Chat with {MODEL_OPTIONS.get(selected_model, selected_model)}\n\n"
                    for msg in st.session_state.chat_history:
                        role = "**User**" if msg["role"] == "user" else "**Assistant**"
                        export_text += f"{role}: {msg['content']}\n\n"
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

        st.markdown("---")
        st.caption("Made with ❤️ using Streamlit and OpenRouter")

    # Use the selected model
    if not selected_model:
        st.warning("Please select a model to continue.")
        st.stop()

    # ---- SESSION STATE FOR CHAT ----
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ---- DISPLAY CURRENT MODEL ----
    st.markdown(f'<div class="model-header">💬 Chatting with: {MODEL_OPTIONS.get(selected_model, selected_model)}</div>', unsafe_allow_html=True)

    # ---- DISPLAY CHAT HISTORY ----
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.chat_message("user").markdown(msg["content"])
            else:
                st.chat_message("assistant").markdown(msg["content"])

    # ---- CHAT INPUT AT THE BOTTOM ----
    user_input = st.chat_input("Type your message and press Enter...")

    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Show user message immediately
        with chat_container:
            st.chat_message("user").markdown(user_input)
        
        # Show progress
        with st.spinner(f"🤖 {MODEL_OPTIONS.get(selected_model, selected_model)} is thinking..."):
            # Call the API
            response = call_model_api(
                selected_model,
                st.session_state.chat_history,
                current_api_key,
                temperature,
                max_tokens,
                timeout,
                system_message
            )
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Show assistant response
        with chat_container:
            st.chat_message("assistant").markdown(response)
        
        st.rerun()

    # ---- FOOTER INFO ----
    if st.session_state.chat_history:
        with st.expander("📊 Session Info"):
            total_messages = len(st.session_state.chat_history)
            user_messages = len([msg for msg in st.session_state.chat_history if msg["role"] == "user"])
            assistant_messages = len([msg for msg in st.session_state.chat_history if msg["role"] == "assistant"])
            
            st.write(f"**Current model:** {MODEL_OPTIONS.get(selected_model, selected_model)}")
            st.write(f"**Total messages:** {total_messages}")
            st.write(f"**Your messages:** {user_messages}")
            st.write(f"**AI responses:** {assistant_messages}")
