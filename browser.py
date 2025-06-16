# browser.py - Dynamic Model Browser

import streamlit as st
from api_utils import get_available_models

def render_model_browser():
    """Renders a model browser that fetches models dynamically from OpenRouter."""
    st.title("🔍 OpenRouter Model Browser")
    
    current_api_key = st.session_state.get('api_key')

    if not current_api_key:
        st.warning("⚠️ Please enter your API key in the 'Chat' tab to browse available models.")
        st.stop()

    if 'browser_models' not in st.session_state:
        st.session_state.browser_models = {}

    if st.button("🔄 Fetch/Refresh All Models", key="browser_fetch_models"):
        with st.spinner("Fetching all available models..."):
            models, message = get_available_models(current_api_key)
            if models:
                st.session_state.browser_models = models
                st.success(f"✅ Successfully loaded {len(models)} models!")
            else:
                st.error(f"❌ {message}")

    if not st.session_state.browser_models:
        st.info("Click the button above to fetch the list of models from OpenRouter.")
        st.stop()

    # --- Search and Filter ---
    st.subheader("Filter Models")
    search_query = st.text_input("Search by name or ID:", placeholder="e.g., llama, gpt-4, vision")
    
    filtered_models = {
        mid: name for mid, name in st.session_state.browser_models.items()
        if search_query.lower() in name.lower() or search_query.lower() in mid.lower()
    }
    
    st.metric("Models Found", len(filtered_models))
    
    # --- Display Models ---
    st.subheader("Available Models")
    
    if not filtered_models:
        st.info("No models match your search query.")
    else:
        # Create a more organized display
        for model_id, display_name in filtered_models.items():
            cols = st.columns([4, 1])
            with cols[0]:
                st.write(f"**{display_name}**")
                st.caption(f"`{model_id}`")
            with cols[1]:
                if st.button("Select", key=f"select_{model_id}"):
                    st.session_state.selected_model = model_id
                    st.success(f"Selected '{display_name}' for chat!")
            st.divider()

    # You can further enhance this browser with more details from the API response,
    # such as pricing, context length, etc., by modifying the get_available_models
    # function to return more data.
