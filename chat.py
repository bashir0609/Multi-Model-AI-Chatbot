# chat.py - Single model chat interface

import os
import streamlit as st
from models import MODEL_OPTIONS, get_cost_info
from api_utils import validate_api_key, call_model_api

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
        if 'api_key_source' not in st.session_state:
            st.session_state.api_key_source = "Environment Variable"
        
        # Check if we have a cached API key
        if st.session_state.cached_api_key:
            # Show current cached API key status
            masked_key = st.session_state.cached_api_key[:8] + "..." + st.session_state.cached_api_key[-4:] if len(st.session_state.cached_api_key) > 12 else "sk-..."
            st.success(f"✅ API Key Active: `{masked_key}`")
            
            # Cache control buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Cache", help="Clear stored API key", use_container_width=True):
                    st.session_state.cached_api_key = None
                    st.success("✅ API key cache cleared!")
                    st.rerun()
            
            with col2:
                if st.button("🔄 Change Key", help="Enter a different API key", use_container_width=True):
                    st.session_state.cached_api_key = None
                    st.rerun()
            
            current_api_key = st.session_state.cached_api_key
            
        else:
            # No cached key - show input options
            # Load API key from environment
            env_api_key = os.getenv("OPENROUTER_API_KEY")
            
            # API Key source selection
            api_source = st.radio(
                "Choose API Key Source:",
                ["Environment Variable", "Manual Input"],
                index=0 if st.session_state.api_key_source == "Environment Variable" else 1,
                help="Select how you want to provide your OpenRouter API key"
            )
            
            st.session_state.api_key_source = api_source
            final_api_key = None
            
            if api_source == "Environment Variable":
                if env_api_key:
                    env_api_key = env_api_key.strip()
                    is_valid, message = validate_api_key(env_api_key)
                    
                    if is_valid:
                        # Show partial key for verification
                        masked_key = env_api_key[:8] + "..." + env_api_key[-4:] if len(env_api_key) > 12 else "sk-..."
                        st.info(f"🔍 Environment key found: `{masked_key}`")
                        
                        if st.button("✅ Use Environment Key", use_container_width=True):
                            st.session_state.cached_api_key = env_api_key
                            st.success("✅ Environment API key cached!")
                            st.rerun()
                        
                        # Option to view/edit the key
                        with st.expander("🔍 Edit Environment Key", expanded=False):
                            edited_key = st.text_input(
                                "Edit environment key:",
                                value=env_api_key,
                                type="password",
                                help="Modify if needed"
                            )
                            if st.button("Use Edited Key"):
                                if edited_key.strip():
                                    is_valid_edited, message_edited = validate_api_key(edited_key.strip())
                                    if is_valid_edited:
                                        st.session_state.cached_api_key = edited_key.strip()
                                        st.success("✅ Edited API key cached!")
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
                        
                        if st.button("💾 Save & Use Key", use_container_width=True):
                            st.session_state.cached_api_key = manual_key
                            st.success("✅ API key cached!")
                            st.rerun()
                    else:
                        st.error(f"❌ {message}")
                else:
                    st.warning("⚠️ Please enter your API key above")
            
            # No valid key available yet
            current_api_key = None
        
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
        
        # Initialize selected model in session state
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = default_model
        
        # Single model selection
        selected_model = st.selectbox(
            "Choose a model:",
            options=list(MODEL_OPTIONS.keys()),
            format_func=lambda x: MODEL_OPTIONS[x],
            index=list(MODEL_OPTIONS.keys()).index(st.session_state.selected_model) if st.session_state.selected_model in MODEL_OPTIONS else 0,
            help="Select one model to chat with.",
            key="model_selector"
        )
        
        # Update session state when selection changes
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model

        # Show model info
        cost_info = get_cost_info(selected_model, MODEL_OPTIONS[selected_model])
        if cost_info['type'] == 'free':
            st.success(f"🆓 {cost_info['cost']}")
        else:
            st.info(f"💰 {cost_info['cost']}")

        # Manual model input
        st.subheader("➕ Add Custom Model")
        manual_model = st.text_input(
            "Model ID:",
            placeholder="e.g., deepseek/deepseek-chat:free",
            help="Enter a model ID from OpenRouter"
        )
        
        if st.button("Use Custom Model", use_container_width=True):
            if manual_model.strip():
                st.session_state.selected_model = manual_model.strip()
                st.success(f"✅ Switched to: {manual_model}")
                st.rerun()
            else:
                st.error("Please enter a model ID")

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
