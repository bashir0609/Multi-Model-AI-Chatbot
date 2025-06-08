# persistent_cache.py - Secure persistent API key caching

import os
import json
import hashlib
import base64
from pathlib import Path
from cryptography.fernet import Fernet

class APIKeyCache:
    def __init__(self):
        self.cache_dir = Path.home() / ".openrouter_chatbot"
        self.cache_file = self.cache_dir / "api_cache.json"
        self.key_file = self.cache_dir / ".key"
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(exist_ok=True)
        
        # Generate or load encryption key
        self.cipher_key = self._get_or_create_key()
    
    def _get_or_create_key(self):
        """Get or create encryption key for secure storage"""
        if self.key_file.exists():
            with open(self.key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            # Make key file read-only for security
            os.chmod(self.key_file, 0o600)
            return key
    
    def _encrypt(self, data):
        """Encrypt data using Fernet"""
        try:
            fernet = Fernet(self.cipher_key)
            return fernet.encrypt(data.encode()).decode()
        except Exception:
            return None
    
    def _decrypt(self, encrypted_data):
        """Decrypt data using Fernet"""
        try:
            fernet = Fernet(self.cipher_key)
            return fernet.decrypt(encrypted_data.encode()).decode()
        except Exception:
            return None
    
    def save_api_key(self, api_key, source="manual"):
        """Save API key to encrypted cache"""
        try:
            encrypted_key = self._encrypt(api_key)
            if encrypted_key:
                cache_data = {
                    "encrypted_key": encrypted_key,
                    "source": source,
                    "timestamp": str(hash(api_key))  # Simple verification
                }
                
                with open(self.cache_file, 'w') as f:
                    json.dump(cache_data, f)
                
                # Make cache file read-only for security
                os.chmod(self.cache_file, 0o600)
                return True
        except Exception as e:
            print(f"Error saving API key: {e}")
        return False
    
    def load_api_key(self):
        """Load API key from encrypted cache"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                encrypted_key = cache_data.get("encrypted_key")
                if encrypted_key:
                    decrypted_key = self._decrypt(encrypted_key)
                    if decrypted_key:
                        return {
                            "key": decrypted_key,
                            "source": cache_data.get("source", "cached"),
                            "cached": True
                        }
        except Exception as e:
            print(f"Error loading API key: {e}")
        return None
    
    def clear_cache(self):
        """Clear all cached data"""
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
            if self.key_file.exists():
                self.key_file.unlink()
            return True
        except Exception as e:
            print(f"Error clearing cache: {e}")
        return False
    
    def is_cached(self):
        """Check if API key is cached"""
        return self.cache_file.exists()

# Fallback simple cache (if cryptography is not available)
class SimpleAPIKeyCache:
    def __init__(self):
        self.cache_dir = Path.home() / ".openrouter_chatbot"
        self.cache_file = self.cache_dir / "simple_cache.json"
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(exist_ok=True)
    
    def save_api_key(self, api_key, source="manual"):
        """Save API key to simple cache (base64 encoded)"""
        try:
            # Simple encoding (not secure, but better than plain text)
            encoded_key = base64.b64encode(api_key.encode()).decode()
            
            cache_data = {
                "encoded_key": encoded_key,
                "source": source,
                "hash": hashlib.md5(api_key.encode()).hexdigest()[:8]  # Verification
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            return True
        except Exception as e:
            print(f"Error saving API key: {e}")
        return False
    
    def load_api_key(self):
        """Load API key from simple cache"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                encoded_key = cache_data.get("encoded_key")
                if encoded_key:
                    decoded_key = base64.b64decode(encoded_key.encode()).decode()
                    return {
                        "key": decoded_key,
                        "source": cache_data.get("source", "cached"),
                        "cached": True
                    }
        except Exception as e:
            print(f"Error loading API key: {e}")
        return None
    
    def clear_cache(self):
        """Clear cached data"""
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
            return True
        except Exception as e:
            print(f"Error clearing cache: {e}")
        return False
    
    def is_cached(self):
        """Check if API key is cached"""
        return self.cache_file.exists()

# Factory function to get appropriate cache
def get_api_cache():
    """Get the best available API cache implementation"""
    try:
        from cryptography.fernet import Fernet
        return APIKeyCache()
    except ImportError:
        print("Note: Using simple cache (install 'cryptography' for secure caching)")
        return SimpleAPIKeyCache()
