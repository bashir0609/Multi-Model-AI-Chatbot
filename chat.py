# chat.py - Chat interface with input at bottom

import os
import streamlit as st
from models import MODEL_OPTIONS, get_cost_info
from api_utils import validate_api_key, call_model_api, call_models_parallel

def chat_interface():
    """Main chat interface function"""
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
        st.success(f"✅ {len(st.session_state.transfer_models)} models transferred from browser!")
        # Auto-select the transferred models
        default_models = st.session_state.transfer_models
        del st.session_state.transfer_models  # Clean up
    else:
        default_models = ["meta-llama/llama-3.1-8b-instruct"]

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
    current_api_key = None  # Initialize outside the sidebar context
    
    with st.sidebar:
        st.header("🔐 API Access")
        
        # Load API key from environment
        api_key = os.getenv("OPENROUTER_API_KEY")
        
        # API Key source selection
        api_source = st.radio(
            "Choose API Key Source:",
            ["Environment Variable", "Manual Input"],
            help="Select how you want to provide your OpenRouter API key"
        )
        
        final_api_key = None
        
        if api_source == "Environment Variable":
            if api_key:
                api_key = api_key.strip()
                is_valid, message = validate_api_key(api_key)
                
                if is_valid:
                    # Show partial key for verification
                    masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "sk-..."
                    st.success(f"✅ Environment API key loaded: `{masked_key}`")
                    final_api_key = api_key
                    
                    # Option to view/edit the key
                    with st.expander("🔍 View/Edit Environment Key", expanded=False):
                        edited_key = st.text_input(
                            "Current environment key:",
                            value=api_key,
                            type="password",
                            help="Edit if needed"
                        )
                        if edited_key != api_key:
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
        
        # Final validation and assignment (inside sidebar but will be available outside)
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
                if current_api_key:  # Check if we have a valid key
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
                else:
                    st.error("❌ No API key available for testing")
        
        with col2:
            if st.button("📋 Copy API Setup", help="Copy .env file format", use_container_width=True):
                if current_api_key:
                    env_format = f"OPENROUTER_API_KEY={current_api_key}"
                    st.code(env_format, language="bash")
                    st.info("📝 Copy this to your .env file")
                else:
                    st.error("❌ No API key available")

        st.divider()
        st.header("🤖 Model Selection")
        
        # Initialize selected models in session state
        if 'chat_selected_models' not in st.session_state:
            st.session_state.chat_selected_models = default_models
        
        selected_models = st.multiselect(
            "Choose one or more models to compare:",
            options=list(MODEL_OPTIONS.keys()),
            format_func=lambda x: MODEL_OPTIONS[x],
            default=st.session_state.chat_selected_models,
            help="Select multiple models to compare their responses.",
            key="model_selector"
        )
        
        # Update session state when selection changes
        if selected_models != st.session_state.chat_selected_models:
            st.session_state.chat_selected_models = selected_models

        # Quick model selection buttons
        st.subheader("⚡ Quick Select")
        quick_models = {
            "🦙 Llama Basic": "meta-llama/llama-3.1-8b-instruct",
            "🌟 Mistral": "mistralai/mistral-7b-instruct", 
            "🧠 DeepSeek": "deepseek/deepseek-chat",
            "🆓 Try Free Llama": "meta-llama/llama-3.1-8b-instruct:free",
            "🆓 Try Free Mistral": "mistralai/mistral-7b-instruct:free",
            "💡 Check OpenRouter": "meta-llama/llama-3.1-8b-instruct"
        }
        
        cols = st.columns(2)
        for i, (name, model_id) in enumerate(quick_models.items()):
            with cols[i % 2]:
                display_name = MODEL_OPTIONS.get(model_id, model_id)
                # Show cost info
                cost_info = get_cost_info(model_id, display_name)
                cost_badge = "🆓" if cost_info['type'] == 'free' else "💰" if cost_info['type'] == 'ultra_cheap' else "💳"
                
                if st.button(f"{cost_badge} {name}", key=f"quick_chat_{i}", use_container_width=True):
                    if model_id not in st.session_state.chat_selected_models:
                        st.session_state.chat_selected_models.append(model_id)
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

    # Use the selected models from session state for consistency
    selected_models = st.session_state.chat_selected_models

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
                    if st.button(f"Clear {MODEL_OPTIONS[model][:20]}...", key=f"clear_tab_{idx}"):
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
                if st.button(f"Clear", key=f"clear_col_{idx}", help=f"Clear {MODEL_OPTIONS[model]}"):
                    st.session_state.chat_history[model] = []
                    st.rerun()

    # ---- CHAT INPUT AT THE BOTTOM ----
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

    # ---- FOOTER INFO ----
    if st.session_state.chat_history:
        with st.expander("📊 Session Info"):
            total_messages = sum(len(history) for history in st.session_state.chat_history.values())
            st.write(f"**Total messages:** {total_messages}")
            st.write(f"**Active models:** {len(selected_models)}")
            
            for model in selected_models:
                model_messages = len(st.session_state.chat_history.get(model, []))
                st.write(f"- {MODEL_OPTIONS[model]}: {model_messages} messages")
