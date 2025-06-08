# persistent_cache.py - Ultra Simple File Cache

import os

def get_cache_file():
    """Get cache file path"""
    return os.path.expanduser("~/.openrouter_key.txt")

def save_api_key(api_key):
    """Save API key to simple text file"""
    try:
        with open(get_cache_file(), 'w') as f:
            f.write(api_key.strip())
        print("✅ API key saved to file")
        return True
    except Exception as e:
        print(f"❌ Save failed: {e}")
        return False

def load_api_key():
    """Load API key from simple text file"""
    try:
        cache_file = get_cache_file()
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                key = f.read().strip()
            if key:
                print("✅ API key loaded from file")
                return key
        print("ℹ️ No cache file found")
        return None
    except Exception as e:
        print(f"❌ Load failed: {e}")
        return None

def clear_cache():
    """Clear the cache file"""
    try:
        cache_file = get_cache_file()
        if os.path.exists(cache_file):
            os.remove(cache_file)
        print("✅ Cache file removed")
        return True
    except Exception as e:
        print(f"❌ Clear failed: {e}")
        return False

def cache_exists():
    """Check if cache file exists"""
    return os.path.exists(get_cache_file())

# For compatibility with existing code
class SimpleCache:
    def save_api_key(self, api_key, source="manual"):
        return save_api_key(api_key)
    
    def load_api_key(self):
        key = load_api_key()
        if key:
            return {"key": key, "source": "cached", "cached": True, "method": "simple_file"}
        return None
    
    def clear_cache(self):
        return clear_cache()
    
    def is_cached(self):
        return cache_exists()
    
    def get_cache_info(self):
        return {
            "cache_file": get_cache_file(),
            "cache_exists": cache_exists(),
            "method": "simple_file"
        }

def get_api_cache():
    return SimpleCache()

# Test function
if __name__ == "__main__":
    print("🧪 Testing simple cache...")
    
    # Test save
    test_key = "sk-test-12345"
    if save_api_key(test_key):
        print("✅ Save test passed")
        
        # Test load
        loaded = load_api_key()
        if loaded == test_key:
            print("✅ Load test passed")
            
            # Test clear
            if clear_cache():
                print("✅ Clear test passed")
                print("🎉 All tests passed!")
            else:
                print("❌ Clear test failed")
        else:
            print("❌ Load test failed")
    else:
        print("❌ Save test failed")
