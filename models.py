# models.py - Current working models for OpenRouter (Updated June 2025)

# ---- CURRENT WORKING MODELS ----
MODEL_OPTIONS = {
    # === FREE MODELS (Most Reliable) ===
    # DeepSeek Models (Usually Very Reliable)
    "deepseek/deepseek-chat:free": "DeepSeek Chat (FREE)",
    "deepseek/deepseek-r1:free": "DeepSeek R1 Reasoning (FREE)",
    "deepseek/deepseek-coder:free": "DeepSeek Coder (FREE)",
    
    # Meta Llama Models (Most Stable Free Options)
    "meta-llama/llama-3.1-8b-instruct:free": "Llama 3.1 8B Instruct (FREE)",
    "meta-llama/llama-3.1-70b-instruct:free": "Llama 3.1 70B Instruct (FREE)",
    "meta-llama/llama-3.2-1b-instruct:free": "Llama 3.2 1B Instruct (FREE)",
    "meta-llama/llama-3.2-3b-instruct:free": "Llama 3.2 3B Instruct (FREE)",
    "meta-llama/llama-3.2-11b-vision-instruct:free": "Llama 3.2 11B Vision (FREE)",
    "meta-llama/llama-3.3-70b-instruct:free": "Llama 3.3 70B Instruct (FREE)",
    
    # Google Models (Experimental but Often Available)
    "google/gemini-2.0-flash-experimental:free": "Gemini 2.0 Flash Experimental (FREE)",
    "google/gemini-pro:free": "Gemini Pro (FREE)",
    
    # Mistral Models
    "mistralai/mistral-7b-instruct:free": "Mistral 7B Instruct (FREE)",
    
    # Qwen Models (Often Available)
    "qwen/qwen2.5-72b-instruct:free": "Qwen 2.5 72B Instruct (FREE)",
    "qwen/qwen2.5-coder-32b-instruct:free": "Qwen 2.5 Coder 32B (FREE)",
    
    # === ULTRA-CHEAP MODELS (Under $0.10/1M tokens) ===
    "meta-llama/llama-3.2-3b-instruct": "Llama 3.2 3B Instruct ($0.02/1M)",
    "meta-llama/llama-3.1-8b-instruct": "Llama 3.1 8B Instruct ($0.05/1M)",
    "mistralai/mistral-7b-instruct": "Mistral 7B Instruct ($0.07/1M)",
    "qwen/qwen2.5-7b-instruct": "Qwen 2.5 7B Instruct ($0.07/1M)",
    "deepseek/deepseek-chat": "DeepSeek Chat ($0.02/1M)",
    
    # === BACKUP/ALTERNATIVE MODELS ===
    "nvidia/llama-3.1-nemotron-8b-instruct:free": "NVIDIA Nemotron 8B (FREE)",
    "meta-llama/llama-4-maverick:free": "Llama 4 Maverick 400B MoE (FREE)",
}

def categorize_models():
    """Categorize models by type and capability."""
    categories = {
        '🆓 Free Models (Recommended)': [],
        '💰 Ultra-Cheap Models (Under $0.10/1M)': [],
        '🧠 Reasoning & Problem Solving': [],
        '💻 Coding & Programming': [],
        '👁️ Vision & Multimodal': [],
        '🚀 Speed & Efficiency': [],
        '🦣 Large Scale (70B+)': [],
    }
    
    for model_id, display_name in MODEL_OPTIONS.items():
        # Basic categorization by price
        if ':free' in model_id:
            categories['🆓 Free Models (Recommended)'].append((model_id, display_name))
        elif any(price in display_name for price in ['$0.02', '$0.05', '$0.07']):
            categories['💰 Ultra-Cheap Models (Under $0.10/1M)'].append((model_id, display_name))
        
        # Capability-based categorization
        name_lower = display_name.lower()
        
        # Reasoning models
        if any(word in name_lower for word in ['r1', 'reasoning', '70b', '72b']):
            categories['🧠 Reasoning & Problem Solving'].append((model_id, display_name))
        
        # Coding models
        if any(word in name_lower for word in ['coder', 'code']):
            categories['💻 Coding & Programming'].append((model_id, display_name))
        
        # Vision models
        if any(word in name_lower for word in ['vision', 'vl', 'multimodal']):
            categories['👁️ Vision & Multimodal'].append((model_id, display_name))
        
        # Speed models (smaller, faster)
        if any(word in name_lower for word in ['1b', '3b', '8b']):
            categories['🚀 Speed & Efficiency'].append((model_id, display_name))
        
        # Large scale models
        if any(word in name_lower for word in ['70b', '72b', '400b', 'maverick']):
            categories['🦣 Large Scale (70B+)'].append((model_id, display_name))
    
    return categories

def analyze_model_capabilities():
    """Analyze model capabilities by specific use cases."""
    capabilities = {
        'reasoning': [],
        'coding': [],
        'vision': [],
        'speed': [],
        'multilingual': [],
        'creative': [],
        'analytical': []
    }
    
    for model_id, display_name in MODEL_OPTIONS.items():
        name_lower = display_name.lower()
        
        # Reasoning capabilities
        if any(word in name_lower for word in ['r1', 'reasoning', '70b', '72b', 'maverick']):
            capabilities['reasoning'].append((model_id, display_name))
        
        # Coding capabilities
        if any(word in name_lower for word in ['coder', 'code', 'deepseek', 'qwen']):
            capabilities['coding'].append((model_id, display_name))
        
        # Vision capabilities
        if any(word in name_lower for word in ['vision', 'vl', 'gemini', 'multimodal']):
            capabilities['vision'].append((model_id, display_name))
        
        # Speed/efficiency
        if any(word in name_lower for word in ['1b', '3b', 'flash']):
            capabilities['speed'].append((model_id, display_name))
        
        # Multilingual (Qwen, Gemini)
        if any(word in name_lower for word in ['qwen', 'gemini', 'multilingual']):
            capabilities['multilingual'].append((model_id, display_name))
        
        # Creative writing (larger models)
        if any(word in name_lower for word in ['70b', 'llama', 'mistral']):
            capabilities['creative'].append((model_id, display_name))
        
        # Analytical tasks
        if any(word in name_lower for word in ['deepseek', 'qwen', 'reasoning']):
            capabilities['analytical'].append((model_id, display_name))
    
    return capabilities

def get_model_stats():
    """Get comprehensive statistics about available models."""
    total_models = len(MODEL_OPTIONS)
    free_models = len([m for m in MODEL_OPTIONS.keys() if ':free' in m])
    cheap_models = len([m for m in MODEL_OPTIONS.values() if '$0.' in m])
    
    # Provider distribution
    providers = {}
    for model_id in MODEL_OPTIONS.keys():
        provider = model_id.split('/')[0] if '/' in model_id else 'openai'
        providers[provider] = providers.get(provider, 0) + 1
    
    # Model size distribution
    sizes = {'Small (1B-8B)': 0, 'Medium (7B-32B)': 0, 'Large (70B+)': 0}
    for display_name in MODEL_OPTIONS.values():
        if any(size in display_name.lower() for size in ['1b', '3b', '7b', '8b']):
            sizes['Small (1B-8B)'] += 1
        elif any(size in display_name.lower() for size in ['32b']):
            sizes['Medium (7B-32B)'] += 1
        elif any(size in display_name.lower() for size in ['70b', '72b', '400b']):
            sizes['Large (70B+)'] += 1
    
    return {
        'total': total_models,
        'free': free_models,
        'cheap': cheap_models,
        'providers': providers,
        'sizes': sizes,
        'free_percentage': round((free_models / total_models) * 100, 1) if total_models > 0 else 0
    }

def get_cost_info(model_id, display_name):
    """Get detailed cost and capability information for a model."""
    
    # Determine model type and cost
    if ':free' in model_id:
        cost_type = 'free'
        cost_desc = 'FREE'
        limits = 'Rate limited: 50/day (basic) or 1000/day (with $10+ credits)'
        color = 'success'
        description = 'Free to use with rate limits'
    elif '$0.' in display_name:
        cost_type = 'ultra_cheap'
        # Extract cost from display name
        cost_match = display_name.split('($')[1].split('/')[0] if '($' in display_name else '$0.XX'
        cost_desc = f'{cost_match}/1M tokens'
        limits = 'No rate limits'
        color = 'info'
        description = 'Ultra-affordable with no limits'
    else:
        cost_type = 'paid'
        cost_desc = 'Check OpenRouter for pricing'
        limits = 'Standard API limits'
        color = 'warning'
        description = 'Paid model - check current pricing'
    
    # Determine model capabilities
    capabilities = []
    name_lower = display_name.lower()
    
    if any(word in name_lower for word in ['r1', 'reasoning']):
        capabilities.append('Advanced Reasoning')
    if any(word in name_lower for word in ['coder', 'code']):
        capabilities.append('Code Generation')
    if any(word in name_lower for word in ['vision', 'vl']):
        capabilities.append('Vision/Multimodal')
    if any(word in name_lower for word in ['1b', '3b']):
        capabilities.append('Fast/Lightweight')
    if any(word in name_lower for word in ['70b', '72b', '400b']):
        capabilities.append('Large Scale')
    if 'experimental' in name_lower:
        capabilities.append('Experimental/Latest')
    
    # Default capabilities if none detected
    if not capabilities:
        capabilities = ['General Purpose']
    
    return {
        'type': cost_type,
        'cost': cost_desc,
        'limits': limits,
        'color': color,
        'description': description,
        'capabilities': capabilities
    }

def get_recommended_models():
    """Get recommended models for different use cases."""
    recommendations = {
        'general': [
            'deepseek/deepseek-chat:free',
            'meta-llama/llama-3.1-8b-instruct:free',
            'meta-llama/llama-3.3-70b-instruct:free'
        ],
        'coding': [
            'deepseek/deepseek-coder:free',
            'qwen/qwen2.5-coder-32b-instruct:free',
            'meta-llama/llama-3.1-70b-instruct:free'
        ],
        'reasoning': [
            'deepseek/deepseek-r1:free',
            'meta-llama/llama-3.3-70b-instruct:free',
            'qwen/qwen2.5-72b-instruct:free'
        ],
        'speed': [
            'meta-llama/llama-3.2-1b-instruct:free',
            'meta-llama/llama-3.2-3b-instruct:free',
            'deepseek/deepseek-chat:free'
        ],
        'vision': [
            'meta-llama/llama-3.2-11b-vision-instruct:free',
            'google/gemini-2.0-flash-experimental:free'
        ],
        'cheap': [
            'meta-llama/llama-3.2-3b-instruct',
            'deepseek/deepseek-chat',
            'meta-llama/llama-3.1-8b-instruct'
        ]
    }
    
    return recommendations

def is_model_likely_working(model_id):
    """Estimate if a model is likely to be working based on provider and type."""
    
    # Most reliable providers for free models
    reliable_free_providers = ['deepseek', 'meta-llama']
    provider = model_id.split('/')[0] if '/' in model_id else ''
    
    if ':free' in model_id:
        # Free models - DeepSeek and Meta are usually reliable
        if provider in reliable_free_providers:
            return True, "Usually reliable"
        elif provider in ['google', 'qwen']:
            return True, "Often available"
        else:
            return False, "May be limited"
    else:
        # Paid models are generally more reliable
        return True, "Paid models usually work"

def get_model_provider_info():
    """Get information about different providers."""
    provider_info = {
        'meta-llama': {
            'name': 'Meta',
            'strengths': 'General purpose, reliable, good free tier',
            'free_models': True,
            'specialties': ['General chat', 'Reasoning', 'Vision (3.2 11B)']
        },
        'deepseek': {
            'name': 'DeepSeek',
            'strengths': 'Excellent reasoning, coding, very reliable free tier',
            'free_models': True,
            'specialties': ['Reasoning (R1)', 'Coding', 'Efficiency']
        },
        'google': {
            'name': 'Google',
            'strengths': 'Latest experimental models, multimodal',
            'free_models': True,
            'specialties': ['Experimental features', 'Multimodal', 'Research']
        },
        'qwen': {
            'name': 'Alibaba Qwen',
            'strengths': 'Multilingual, coding, large context',
            'free_models': True,
            'specialties': ['Multilingual', 'Coding', 'Mathematical reasoning']
        },
        'mistralai': {
            'name': 'Mistral AI',
            'strengths': 'European AI, efficiency, instruction following',
            'free_models': True,
            'specialties': ['Instruction following', 'Efficiency', 'European focus']
        },
        'nvidia': {
            'name': 'NVIDIA',
            'strengths': 'Optimized performance, enterprise focus',
            'free_models': True,
            'specialties': ['Performance optimization', 'Enterprise use']
        }
    }
    
    return provider_info
