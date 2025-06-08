# persistent_cache.py - DISABLED (prevents any caching conflicts)

# This file is disabled to prevent any caching-related key conflicts
# All API key storage is now session-only in chat.py

print("ℹ️ Persistent cache disabled - using session storage only")

def get_api_cache():
    """Dummy function - no caching"""
    return None

# Dummy functions for compatibility
def load_cached_api_key():
    return None

def save_api_key_to_cache(api_key, source="manual"):
    return False

def clear_api_key_cache():
    return False
