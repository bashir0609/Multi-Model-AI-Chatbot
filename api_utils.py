# api_utils.py - API calls and utility functions

import requests
from concurrent.futures import ThreadPoolExecutor

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

def get_model_identity(model_id):
    """Get the proper identity for each model"""
    model_identities = {
        # DeepSeek models
        "deepseek/deepseek-chat": "DeepSeek Chat",
        "deepseek/deepseek-r1": "DeepSeek R1", 
        "deepseek/deepseek-coder": "DeepSeek Coder",
        
        # Meta Llama models  
        "meta-llama/llama-3.1-8b-instruct": "Llama 3.1 8B",
        "meta-llama/llama-3.1-70b-instruct": "Llama 3.1 70B",
        "meta-llama/llama-3.1-405b-instruct": "Llama 3.1 405B",
        "meta-llama/llama-3.2-1b-instruct": "Llama 3.2 1B",
        "meta-llama/llama-3.2-3b-instruct": "Llama 3.2 3B",
        "meta-llama/llama-3.2-11b-vision-instruct": "Llama 3.2 11B Vision",
        "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B",
        
        # Mistral models
        "mistralai/mistral-7b-instruct": "Mistral 7B",
        "mistralai/mistral-medium": "Mistral Medium",
        "mistralai/mistral-large": "Mistral Large",
        
        # Google models
        "google/gemini-pro": "Google Gemini Pro",
        "google/gemini-2.0-flash-experimental": "Google Gemini 2.0 Flash",
        
        # Qwen models
        "qwen/qwen2.5-72b-instruct": "Qwen 2.5 72B",
        "qwen/qwen2.5-coder-32b-instruct": "Qwen 2.5 Coder 32B",
        
        # OpenAI models
        "gpt-3.5-turbo": "ChatGPT (GPT-3.5)",
        "gpt-4": "GPT-4",
        "gpt-4-turbo": "GPT-4 Turbo",
    }
    
    # Handle free versions (remove :free suffix)
    base_model = model_id.replace(':free', '')
    
    # Return specific identity or generic fallback
    return model_identities.get(base_model) or model_identities.get(model_id) or f"AI Model ({model_id})"

def call_model_api(model_id, messages, api_key, temperature, max_tokens, timeout=60, system_message=""):
    """Enhanced API call with model identity and better error handling"""
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
    
    # Get model identity
    model_identity = get_model_identity(model_id)
    
    # Create enhanced system message with model identity
    identity_message = f"You are {model_identity}. Always identify yourself as {model_identity} when asked about your identity, model, or what AI you are."
    
    if system_message.strip():
        combined_system_message = f"{identity_message}\n\n{system_message.strip()}"
    else:
        combined_system_message = identity_message
    
    # Add system message with model identity
    messages_with_system = [{"role": "system", "content": combined_system_message}] + messages
    
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
            content += f"\n\n*Tokens: {usage.get('total_tokens', 'N/A')} | Model: {model_identity}*"
        else:
            content += f"\n\n*Model: {model_identity}*"
        
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
