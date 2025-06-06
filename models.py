# models.py - MINIMAL working models (conservative list)

# ---- ULTRA-MINIMAL MODEL LIST ----
MODEL_OPTIONS = {
    # Basic paid models (these usually exist)
    "meta-llama/llama-3.1-8b-instruct": "Llama 3.1 8B Instruct",
    "mistralai/mistral-7b-instruct": "Mistral 7B Instruct", 
    "deepseek/deepseek-chat": "DeepSeek Chat",
    
    # Try these free ones (but may not work)
    "meta-llama/llama-3.1-8b-instruct:free": "Llama 3.1 8B Instruct (FREE - if available)",
    "mistralai/mistral-7b-instruct:free": "Mistral 7B Instruct (FREE - if available)",
}

def categorize_models():
    """Simple categorization."""
    categories = {
        'Basic Models': [],
        'Free Models (May Not Work)': []
    }
    
    for model_id, display_name in MODEL_OPTIONS.items():
        if ':free' in model_id:
            categories['Free Models (May Not Work)'].append((model_id, display_name))
        else:
            categories['Basic Models'].append((model_id, display_name))
    
    return categories

def analyze_model_capabilities():
    """Simple capabilities."""
    capabilities = {
        'basic': list(MODEL_OPTIONS.items())
    }
    return capabilities

def get_model_stats():
    """Get basic stats."""
    total_models = len(MODEL_OPTIONS)
    free_models = len([m for m in MODEL_OPTIONS.keys() if ':free' in m])
    
    return {
        'total': total_models,
        'free': free_models,
        'cheap': total_models - free_models,
        'providers': {'meta-llama': 2, 'mistralai': 2, 'deepseek': 1}
    }

def get_cost_info(model_id, display_name):
    """Simple cost info."""
    if ':free' in model_id:
        return {
            'type': 'free',
            'cost': 'FREE (if available)',
            'limits': 'May have limits',
            'color': 'success',
            'description': 'Free but may not work'
        }
    else:
        return {
            'type': 'paid',
            'cost': 'Check OpenRouter',
            'limits': 'Standard limits',
            'color': 'warning',
            'description': 'Paid model'
        }
