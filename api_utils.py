# api_utils.py - API calls and utility functions

import requests
from concurrent.futures import ThreadPoolExecutor

def get_available_models(api_key):
    """Fetch available models from the OpenRouter API."""
    if not api_key:
        return None, "API key is not set."

    url = "https://openrouter.ai/api/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}"
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes
        models_data = response.json().get('data', [])
        
        # Transform the data into a dictionary of {model_id: display_name}
        models_dict = {model['id']: model.get('name', model['id']) for model in models_data}
        return models_dict, "Successfully fetched models."

    except requests.exceptions.HTTPError as http_err:
        return None, f"HTTP error occurred: {http_err}"
    except requests.exceptions.RequestException as e:
        return None, f"An error occurred while fetching models: {e}"
    except Exception as e:
        return None, f"An unexpected error occurred: {e}"

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
    # This function can now be simplified as the display name is fetched from the API
    # However, you might want to keep it for custom or fallback names.
    # For simplicity, we will just return the model_id as the identity for now.
    return model_id

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
    
    # Create the system message
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
            content += f"\n\n*Tokens: {usage.get('total_tokens', 'N/A')} | Model: {model_id}*"
        else:
            content += f"\n\n*Model: {model_id}*"
        
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
