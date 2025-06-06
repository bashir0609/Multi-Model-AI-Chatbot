import os
import streamlit as st
import requests
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

# ---- STREAMLIT UI ----
st.set_page_config(page_title="Multi-Model AI Chatbot", layout="wide")
st.title("🧠 Multi-Model AI Chatbot (OpenRouter)")

with st.sidebar:
    st.header("🔐 API Access")
    if api_key:
        st.success("API key loaded from environment.")
    else:
        st.error("No API key found. Please add OPENROUTER_API_KEY to your .env file.")
        st.stop()

    st.divider()
    st.header("🤖 Model Selection")
    selected_models = st.multiselect(
        "Choose one or more models to compare:",
        options=list(MODEL_OPTIONS.keys()),
        format_func=lambda x: MODEL_OPTIONS[x],
        default=["deepseek/deepseek-v3-base:free"],
        help="Select multiple models to compare their responses."
    )

    st.divider()
    with st.expander("⚙️ Advanced Settings", expanded=False):
        temperature = st.slider(
            "Temperature",
            0.0, 1.5, 0.7, 0.05,
            help="Higher = more creative, lower = more focused."
        )
        max_tokens = st.slider(
            "Max tokens",
            16, 1024, 256, 8,
            help="Maximum length of the model's response."
        )
        if st.button("Restore Defaults"):
            temperature = 0.7
            max_tokens = 256

    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit and OpenRouter")

if not selected_models:
    st.warning("Please select at least one model.")
    st.stop()

# ---- SESSION STATE FOR CHAT ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

for model in selected_models:
    if model not in st.session_state.chat_history:
        st.session_state.chat_history[model] = []

user_input = st.chat_input("Type your message and press Enter...")

# ---- API CALL FUNCTION ----
def call_model_api(model_id, messages, api_key, temperature, max_tokens):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"❌ Error: {e}"

# ---- MAIN CHAT LOGIC ----
if user_input:
    for model in selected_models:
        st.session_state.chat_history[model].append({"role": "user", "content": user_input})

    for model in selected_models:
        with st.spinner(f"Getting response from {MODEL_OPTIONS[model]}..."):
            response = call_model_api(
                model,
                st.session_state.chat_history[model],
                api_key,
                temperature,
                max_tokens
            )
            st.session_state.chat_history[model].append({"role": "assistant", "content": response})

# ---- DISPLAY CHAT ----
cols = st.columns(len(selected_models))
for idx, model in enumerate(selected_models):
    with cols[idx]:
        st.subheader(MODEL_OPTIONS[model])
        for msg in st.session_state.chat_history[model]:
            if msg["role"] == "user":
                st.chat_message("user").markdown(msg["content"])
            else:
                st.chat_message("assistant").markdown(msg["content"])
        st.markdown("---")
