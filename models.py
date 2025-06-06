# models.py - Updated model definitions and categorization (June 2025)

# ---- MODEL LIST (Updated with currently available models) ----
MODEL_OPTIONS = {
    # ==== COMPLETELY FREE MODELS ====
    # DeepSeek (Free) - Top performers
    "deepseek/deepseek-r1:free": "DeepSeek R1 - Advanced Reasoning (FREE)",
    "deepseek/deepseek-r1-zero:free": "DeepSeek R1 Zero (FREE)",
    "deepseek/deepseek-r1-distill-llama-70b:free": "DeepSeek R1 Distill Llama 70B (FREE)",
    "deepseek/deepseek-r1-distill-qwen-32b:free": "DeepSeek R1 Distill Qwen 32B (FREE)",
    "deepseek/deepseek-r1-distill-qwen-14b:free": "DeepSeek R1 Distill Qwen 14B (FREE)",
    "deepseek/deepseek-prover-v2:free": "DeepSeek Prover V2 - Math & Logic (FREE)",
    "deepseek/deepseek-chat-v3-0324:free": "DeepSeek Chat V3 (FREE)",
    "deepseek/deepseek-chat:free": "DeepSeek Chat (FREE)",
    
    # Meta Llama (Free) - Reliable performers
    "meta-llama/llama-3.3-70b-instruct:free": "Llama 3.3 70B Instruct (FREE)",
    "meta-llama/llama-3.3-8b-instruct:free": "Llama 3.3 8B Instruct (FREE)",
    "meta-llama/llama-3.2-1b-instruct:free": "Llama 3.2 1B Instruct - Fastest (FREE)",
    "meta-llama/llama-3.2-3b-instruct:free": "Llama 3.2 3B Instruct (FREE)",
    "meta-llama/llama-3.2-11b-vision-instruct:free": "Llama 3.2 11B Vision (FREE)",
    "meta-llama/llama-3.1-8b-instruct:free": "Llama 3.1 8B Instruct (FREE)",
    "meta-llama/llama-3.1-405b-base:free": "Llama 3.1 405B Base - Massive (FREE)",
    "meta-llama/llama-4-scout:free": "Llama 4 Scout 109B MoE (FREE)",
    "meta-llama/llama-4-maverick:free": "Llama 4 Maverick 400B MoE (FREE)",
    
    # Google (Free) - Latest research
    "google/gemini-2.5-pro-experimental:free": "Gemini 2.5 Pro Experimental (FREE)",
    "google/gemini-2.5-flash:free": "Gemini 2.5 Flash - Latest (FREE)",
    "google/gemini-2.0-flash-experimental:free": "Gemini 2.0 Flash Experimental (FREE)",
    "google/gemini-2.0-flash-thinking-exp:free": "Gemini 2.0 Flash Thinking (FREE)",
    "google/gemma-3-27b:free": "Gemma 3 27B (FREE)",
    "google/gemma-3-12b:free": "Gemma 3 12B (FREE)",
    "google/gemma-3-4b:free": "Gemma 3 4B (FREE)",
    "google/gemma-3-1b:free": "Gemma 3 1B (FREE)",
    "google/gemma-2-9b:free": "Gemma 2 9B (FREE)",
    "google/gemma-7b-it:free": "Gemma 7B IT (FREE)",
    
    # Qwen (Free) - Multilingual & Vision
    "qwen/qwen2.5-72b-instruct:free": "Qwen2.5 72B Instruct (FREE)",
    "qwen/qwen2.5-32b-instruct:free": "Qwen2.5 32B Instruct (FREE)",
    "qwen/qwen2.5-7b-instruct:free": "Qwen2.5 7B Instruct (FREE)",
    "qwen/qwen2.5-coder-32b-instruct:free": "Qwen2.5 Coder 32B - Best for Code (FREE)",
    "qwen/qwen2.5-vl-72b-instruct:free": "Qwen2.5 VL 72B Vision (FREE)",
    "qwen/qwen2.5-vl-32b-instruct:free": "Qwen2.5 VL 32B Vision (FREE)",
    "qwen/qwen2.5-vl-7b-instruct:free": "Qwen2.5 VL 7B Vision (FREE)",
    "qwen/qwen2.5-vl-3b-instruct:free": "Qwen2.5 VL 3B Vision (FREE)",
    "qwen/qwq-32b:free": "QwQ 32B Reasoning (FREE)",
    
    # Microsoft (Free) - Enterprise grade
    "microsoft/phi-4-reasoning:free": "Phi 4 Reasoning (FREE)",
    "microsoft/phi-4-reasoning-plus:free": "Phi 4 Reasoning Plus (FREE)",
    
    # Mistral (Free) - European AI
    "mistralai/mistral-small-3.1-24b:free": "Mistral Small 3.1 24B (FREE)",
    "mistralai/mistral-small-3:free": "Mistral Small 3 (FREE)",
    "mistralai/mistral-nemo:free": "Mistral Nemo (FREE)",
    "mistralai/mistral-7b-instruct:free": "Mistral 7B Instruct (FREE)",
    
    # NVIDIA (Free) - Optimized performance
    "nvidia/llama-3.1-nemotron-ultra-253b-v1:free": "Llama 3.1 Nemotron Ultra 253B (FREE)",
    "nvidia/llama-3.3-nemotron-super-49b-v1:free": "Llama 3.3 Nemotron Super 49B (FREE)",
    
    # Specialized (Free) - Various capabilities
    "nous-research/hermes-3-llama-3.1-8b:free": "Hermes 3 Llama 3.1 8B (FREE)",
    "nousresearch/deephermes-3-mistral-24b-preview:free": "DeepHermes 3 Mistral 24B (FREE)",
    "cognitivecomputations/dolphin3.0-mistral-24b:free": "Dolphin 3.0 Mistral 24B (FREE)",
    "opengvlab/internvl3-14b:free": "InternVL3 14B Vision (FREE)",
    "opengvlab/internvl3-2b:free": "InternVL3 2B Vision (FREE)",
    "thudm/glm-4-32b:free": "GLM 4 32B (FREE)",
    "rekaai/flash-3:free": "Reka Flash 3 (FREE)",
    
    # ==== ULTRA-CHEAP MODELS (Under $0.50/1M tokens) ====
    # Ministral Series (Super Cheap)
    "mistralai/ministral-3b": "Ministral 3B ($0.04/1M tokens)",
    "mistralai/ministral-8b": "Ministral 8B ($0.06/1M tokens)",
    
    # Small Efficient Models
    "qwen/qwen2.5-0.5b-instruct": "Qwen2.5 0.5B Instruct ($0.02/1M)",
    "qwen/qwen2.5-1.5b-instruct": "Qwen2.5 1.5B Instruct ($0.04/1M)",
    "qwen/qwen2.5-3b-instruct": "Qwen2.5 3B Instruct ($0.06/1M)",
    "google/gemma-2-2b-it": "Gemma 2 2B IT ($0.05/1M)",
    "google/gemma-2-9b-it": "Gemma 2 9B IT ($0.08/1M)",
    
    # DeepSeek Paid (Very Affordable)
    "deepseek/deepseek-r1": "DeepSeek R1 Reasoning ($0.50/1M input, $2.15/1M output)",
    "deepseek/deepseek-v3": "DeepSeek V3 Chat ($0.20/1M input, $1.10/1M output)",
    "deepseek/deepseek-prover-v2": "DeepSeek Prover V2 Math ($0.50/1M input, $2.18/1M output)",
    
    # QwQ Reasoning
    "qwen/qwq-32b": "QwQ 32B Reasoning ($0.15/1M input, $0.20/1M output)",
}

def categorize_models():
    """Categorize models by provider and type."""
    categories = {
        'DeepSeek (Reasoning & Performance)': [],
        'Meta Llama (General Purpose)': [],
        'Google (Research & Latest)': [],
        'Qwen (Multilingual & Vision)': [],
        'Microsoft (Enterprise & Reasoning)': [],
        'Mistral (European AI)': [],
        'NVIDIA (Optimized Performance)': [],
        'Specialized (Vision, Code, Multimodal)': [],
        'Ultra-Cheap (Under $0.50/1M tokens)': [],
    }
    
    for model_id, display_name in MODEL_OPTIONS.items():
        # Check if it's ultra-cheap (non-free but very affordable)
        if ':free' not in model_id and any(price in display_name for price in ['$0.02', '$0.04', '$0.05', '$0.06', '$0.08', '$0.15', '$0.20', '$0.50']):
            categories['Ultra-Cheap (Under $0.50/1M tokens)'].append((model_id, display_name))
        elif model_id.startswith('deepseek/'):
            categories['DeepSeek (Reasoning & Performance)'].append((model_id, display_name))
        elif model_id.startswith('meta-llama/'):
            categories['Meta Llama (General Purpose)'].append((model_id, display_name))
        elif model_id.startswith('google/'):
            categories['Google (Research & Latest)'].append((model_id, display_name))
        elif model_id.startswith('qwen/'):
            categories['Qwen (Multilingual & Vision)'].append((model_id, display_name))
        elif model_id.startswith('microsoft/'):
            categories['Microsoft (Enterprise & Reasoning)'].append((model_id, display_name))
        elif model_id.startswith('mistralai/'):
            categories['Mistral (European AI)'].append((model_id, display_name))
        elif model_id.startswith('nvidia/'):
            categories['NVIDIA (Optimized Performance)'].append((model_id, display_name))
        else:
            categories['Specialized (Vision, Code, Multimodal)'].append((model_id, display_name))
    
    return categories

def analyze_model_capabilities():
    """Analyze and categorize model capabilities."""
    capabilities = {
        'reasoning': [],
        'vision': [],
        'coding': [],
        'thinking': [],
        'large_scale': [],
        'experimental': [],
        'ultra_cheap': []
    }
    
    for model_id, display_name in MODEL_OPTIONS.items():
        name_lower = display_name.lower()
        
        # Reasoning models
        if any(word in name_lower for word in ['reasoning', 'r1', 'prover', 'qwq', 'phi-4']):
            capabilities['reasoning'].append((model_id, display_name))
        
        # Thinking models
        if any(word in name_lower for word in ['thinking', 'reasoning']):
            capabilities['thinking'].append((model_id, display_name))
        
        # Vision/Multimodal models
        if any(word in name_lower for word in ['vision', 'vl', 'multimodal', 'internvl']):
            capabilities['vision'].append((model_id, display_name))
        
        # Coding specialized
        if any(word in name_lower for word in ['coder', 'code', 'dolphin', 'hermes']):
            capabilities['coding'].append((model_id, display_name))
        
        # Large scale (70B+)
        if any(word in name_lower for word in ['70b', '72b', '253b', '405b', '400b', '109b']):
            capabilities['large_scale'].append((model_id, display_name))
        
        # Experimental/Latest
        if any(word in name_lower for word in ['experimental', 'preview', 'maverick', 'scout', 'flash']):
            capabilities['experimental'].append((model_id, display_name))
        
        # Ultra cheap models
        if ':free' not in model_id and any(price in display_name for price in ['$0.02', '$0.04', '$0.05', '$0.06', '$0.08', '$0.15', '$0.20', '$0.50']):
            capabilities['ultra_cheap'].append((model_id, display_name))
    
    return capabilities

def get_model_stats():
    """Get statistics about available models."""
    total_models = len(MODEL_OPTIONS)
    free_models = len([m for m in MODEL_OPTIONS.keys() if ':free' in m])
    cheap_models = len([m for m in MODEL_OPTIONS.values() if any(price in m for price in ['$0.0', '$0.1', '$0.2', '$0.3', '$0.4', '$0.5']) and 'FREE' not in m])
    
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
    elif any(price in display_name for price in ['$0.02', '$0.04', '$0.05', '$0.06', '$0.08']):
        cost_match = display_name.split('$')[1].split('/')[0] if '$' in display_name else '0.06'
        return {
            'type': 'ultra_cheap',
            'cost': f'${cost_match}/1M',
            'limits': 'Standard API limits',
            'color': 'info',
            'description': 'Ultra-affordable pricing'
        }
    elif any(price in display_name for price in ['$0.15', '$0.20', '$0.50']):
        cost_match = display_name.split('$')[1].split('/')[0] if '$' in display_name else '0.50'
        return {
            'type': 'cheap',
            'cost': f'${cost_match}/1M',
            'limits': 'Standard API limits',
            'color': 'info',
            'description': 'Very affordable pricing'
        }
    else:
        return {
            'type': 'paid',
            'cost': 'Variable pricing',
            'limits': 'Standard API limits',
            'color': 'warning',
            'description': 'Standard pricing'
        }
