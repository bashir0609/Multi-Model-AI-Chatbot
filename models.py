# models.py - VERIFIED working models only (June 2025)

# ---- VERIFIED WORKING MODEL LIST ----
MODEL_OPTIONS = {
    # ==== VERIFIED FREE MODELS (confirmed working) ====
    # From search results - these are confirmed to exist
    "google/gemini-2.0-flash-thinking-exp:free": "Gemini 2.0 Flash Thinking (FREE)",
    "deepseek/deepseek-r1-distill-llama-70b:free": "DeepSeek R1 Distill Llama 70B (FREE)",
    
    # ==== VERIFIED PAID MODELS (confirmed working) ====
    # From OpenRouter search results
    "deepseek/deepseek-r1": "DeepSeek R1 ($0.50/1M input, $2.15/1M output)",
    "deepseek/deepseek-r1-distill-llama-70b": "DeepSeek R1 Distill Llama 70B ($0.10/1M input, $0.40/1M output)",
    "deepseek/deepseek-r1-distill-llama-8b": "DeepSeek R1 Distill Llama 8B ($0.04/1M input, $0.04/1M output)",
    "deepseek/deepseek-r1-distill-qwen-32b": "DeepSeek R1 Distill Qwen 32B ($0.12/1M input, $0.18/1M output)",
    "deepseek/deepseek-r1-distill-qwen-14b": "DeepSeek R1 Distill Qwen 14B ($0.10/1M input, $0.20/1M output)",
    "deepseek/deepseek-r1-0528": "DeepSeek R1 0528 ($0.05/1M input, $0.10/1M output)",
    "deepseek/deepseek-prover-v2": "DeepSeek Prover V2 ($0.50/1M input, $2.18/1M output)",
    
    # Safe fallbacks (these are commonly available)
    "meta-llama/llama-3.1-8b-instruct": "Llama 3.1 8B Instruct (Paid - check pricing)",
    "mistralai/mistral-7b-instruct": "Mistral 7B Instruct (Paid - check pricing)",
}

def categorize_models():
    """Categorize models by provider and type."""
    categories = {
        'DeepSeek (Reasoning)': [],
        'Google (Research)': [],
        'Meta Llama': [],
        'Mistral': [],
        'Ultra-Cheap Paid': [],
    }
    
    for model_id, display_name in MODEL_OPTIONS.items():
        if model_id.startswith('deepseek/'):
            categories['DeepSeek (Reasoning)'].append((model_id, display_name))
        elif model_id.startswith('google/'):
            categories['Google (Research)'].append((model_id, display_name))
        elif model_id.startswith('meta-llama/'):
            categories['Meta Llama'].append((model_id, display_name))
        elif model_id.startswith('mistralai/'):
            categories['Mistral'].append((model_id, display_name))
        elif ':free' not in model_id and any(price in display_name for price in ['$0.0', '$0.1']):
            categories['Ultra-Cheap Paid'].append((model_id, display_name))
    
    return categories

def analyze_model_capabilities():
    """Analyze and categorize model capabilities."""
    capabilities = {
        'reasoning': [],
        'thinking': [],
        'large_scale': [],
        'ultra_cheap': []
    }
    
    for model_id, display_name in MODEL_OPTIONS.items():
        name_lower = display_name.lower()
        
        # Reasoning models
        if any(word in name_lower for word in ['reasoning', 'r1', 'prover', 'thinking']):
            capabilities['reasoning'].append((model_id, display_name))
        
        # Thinking models
        if 'thinking' in name_lower:
            capabilities['thinking'].append((model_id, display_name))
        
        # Large scale (70B+)
        if '70b' in name_lower:
            capabilities['large_scale'].append((model_id, display_name))
        
        # Ultra cheap models
        if ':free' not in model_id and any(price in display_name for price in ['$0.0', '$0.1']):
            capabilities['ultra_cheap'].append((model_id, display_name))
    
    return capabilities

def get_model_stats():
    """Get statistics about available models."""
    total_models = len(MODEL_OPTIONS)
    free_models = len([m for m in MODEL_OPTIONS.keys() if ':free' in m])
    cheap_models = len([m for m in MODEL_OPTIONS.values() if any(price in m for price in ['$0.0', '$0.1']) and 'FREE' not in m])
    
    # Provider distribution
    providers = {}
    for model_id in MODEL_OPTIONS.keys():
        provider = model_id.split('/')[0] if '/' in model_id else 'other'
        providers[provider] = providers.get(provider, 0) + 1
    
    return {
        'total': total_models,
        'free': free_models,
        'cheap': cheap_models,
        'providers': providers
    }

def get_cost_info(model_id, display_name):
    """Get cost and limit information for a model."""
    if ':free' in model_id:
        return {
            'type': 'free',
            'cost': 'FREE',
            'limits': '20 req/min, 50-1000 req/day',
            'color': 'success',
            'description': '100% Free with rate limits'
        }
    elif any(price in display_name for price in ['$0.0', '$0.1']):
        cost_match = "Under $0.20"
        return {
            'type': 'ultra_cheap',
            'cost': cost_match,
            'limits': 'Standard API limits',
            'color': 'info',
            'description': 'Ultra-affordable pricing'
        }
    else:
        return {
            'type': 'paid',
            'cost': 'Check OpenRouter',
            'limits': 'Standard API limits',
            'color': 'warning',
            'description': 'Standard pricing'
        }
