# persistent_cache.py - Secure persistent API key caching

import os
import json
import hashlib
import base64
import time
import logging
from pathlib import Path

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIKeyCache:
    def __init__(self):
        self.cache_dir = Path.home() / ".openrouter_chatbot"
        self.cache_file = self.cache_dir / "api_cache.json"
        self.key_file = self.cache_dir / ".key"
        
        # Setup cache directory with better error handling
        self._setup_cache_directory()
        
        # Generate or load encryption key
        self.cipher_key = self._get_or_create_key()
        self.encryption_available = self.cipher_key is not None
    
    def _setup_cache_directory(self):
        """Setup cache directory with proper error handling"""
        try:
            # Create cache directory if it doesn't exist
            self.cache_dir.mkdir(exist_ok=True, mode=0o700)
            logger.info(f"✅ Cache directory ready: {self.cache_dir}")
        except PermissionError:
            logger.error(f"❌ Permission denied creating cache directory: {self.cache_dir}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to create cache directory: {e}")
            raise
    
    def _get_or_create_key(self):
        """Get or create encryption key for secure storage with better error handling"""
        try:
            # Check if cryptography is available
            from cryptography.fernet import Fernet
            
            if self.key_file.exists():
                try:
                    with open(self.key_file, 'rb') as f:
                        key = f.read()
                    logger.info("✅ Loaded existing encryption key")
                    return key
                except Exception as e:
                    logger.error(f"❌ Failed to load encryption key: {e}")
                    # Try to create a new key
                    try:
                        self.key_file.unlink()  # Remove corrupted key file
                        logger.info("🗑️ Removed corrupted key file")
                    except:
                        pass
            
            # Create new key
            try:
                key = Fernet.generate_key()
                with open(self.key_file, 'wb') as f:
                    f.write(key)
                # Make key file read-only for security
                os.chmod(self.key_file, 0o600)
                logger.info("✅ Generated new encryption key")
                return key
            except Exception as e:
                logger.error(f"❌ Failed to create encryption key: {e}")
                return None
                
        except ImportError:
            logger.warning("⚠️ cryptography library not available - encryption disabled")
            return None
        except Exception as e:
            logger.error(f"❌ Encryption setup failed: {e}")
            return None
    
    def _encrypt(self, data):
        """Encrypt data using Fernet with error handling"""
        if not self.encryption_available:
            logger.debug("🔓 Encryption not available, using base64 encoding")
            return None
        
        try:
            from cryptography.fernet import Fernet
            fernet = Fernet(self.cipher_key)
            encrypted = fernet.encrypt(data.encode()).decode()
            logger.debug("🔒 Data encrypted successfully")
            return encrypted
        except Exception as e:
            logger.error(f"❌ Encryption failed: {e}")
            return None
    
    def _decrypt(self, encrypted_data):
        """Decrypt data using Fernet with error handling"""
        if not self.encryption_available:
            return None
        
        try:
            from cryptography.fernet import Fernet
            fernet = Fernet(self.cipher_key)
            decrypted = fernet.decrypt(encrypted_data.encode()).decode()
            logger.debug("🔓 Data decrypted successfully")
            return decrypted
        except Exception as e:
            logger.error(f"❌ Decryption failed: {e}")
            return None
    
    def save_api_key(self, api_key, source="manual"):
        """Save API key to encrypted cache with enhanced error handling"""
        if not api_key or not api_key.strip():
            logger.warning("⚠️ Empty API key provided")
            return False
        
        api_key = api_key.strip()
        
        try:
            cache_data = {
                "source": source,
                "timestamp": int(time.time()),
                "hash": hashlib.md5(api_key.encode()).hexdigest()[:8]  # For verification
            }
            
            # Try encryption first, fall back to encoding
            if self.encryption_available:
                encrypted_key = self._encrypt(api_key)
                if encrypted_key:
                    cache_data["encrypted_key"] = encrypted_key
                    cache_data["method"] = "encrypted"
                    logger.info("💾 Using encrypted storage")
                else:
                    # Fallback to base64 encoding
                    cache_data["encoded_key"] = base64.b64encode(api_key.encode()).decode()
                    cache_data["method"] = "encoded"
                    logger.warning("⚠️ Encryption failed, using base64 encoding")
            else:
                # Use base64 encoding
                cache_data["encoded_key"] = base64.b64encode(api_key.encode()).decode()
                cache_data["method"] = "encoded"
                logger.info("💾 Using base64 encoding")
            
            # Write to temporary file first, then atomically move
            temp_file = self.cache_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            # Atomic move to prevent corruption
            temp_file.replace(self.cache_file)
            
            # Make cache file read-only for security
            os.chmod(self.cache_file, 0o600)
            
            logger.info(f"✅ API key saved successfully using {cache_data['method']} method")
            return True
            
        except PermissionError:
            logger.error("❌ Permission denied saving API key")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to save API key: {e}")
            return False
    
    def load_api_key(self):
        """Load API key from encrypted cache with enhanced error handling"""
        if not self.cache_file.exists():
            logger.info("ℹ️ No cache file found")
            return None
        
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            method = cache_data.get("method", "unknown")
            api_key = None
            
            # Try to load based on storage method
            if method == "encrypted" and "encrypted_key" in cache_data:
                api_key = self._decrypt(cache_data["encrypted_key"])
                if not api_key:
                    logger.warning("⚠️ Decryption failed, trying fallback")
            
            if method == "encoded" and "encoded_key" in cache_data:
                try:
                    api_key = base64.b64decode(cache_data["encoded_key"].encode()).decode()
                    logger.debug("🔓 Decoded API key successfully")
                except Exception as e:
                    logger.error(f"❌ Base64 decoding failed: {e}")
            
            # Try both methods if one fails
            if not api_key and "encrypted_key" in cache_data:
                api_key = self._decrypt(cache_data["encrypted_key"])
            if not api_key and "encoded_key" in cache_data:
                try:
                    api_key = base64.b64decode(cache_data["encoded_key"].encode()).decode()
                except:
                    pass
            
            if api_key:
                # Verify integrity if hash is available
                if "hash" in cache_data:
                    expected_hash = hashlib.md5(api_key.encode()).hexdigest()[:8]
                    if cache_data["hash"] != expected_hash:
                        logger.warning("⚠️ Cache integrity check failed")
                        return None
                
                logger.info(f"✅ API key loaded successfully (method: {method})")
                return {
                    "key": api_key,
                    "source": cache_data.get("source", "cached"),
                    "cached": True,
                    "method": method,
                    "timestamp": cache_data.get("timestamp")
                }
            else:
                logger.error("❌ Failed to decrypt/decode API key from cache")
                return None
                
        except json.JSONDecodeError:
            logger.error("❌ Cache file is corrupted (invalid JSON)")
            return None
        except PermissionError:
            logger.error("❌ Permission denied reading cache file")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to load API key: {e}")
            return None
    
    def clear_cache(self):
        """Clear all cached data with enhanced error handling"""
        try:
            files_removed = []
            
            if self.cache_file.exists():
                self.cache_file.unlink()
                files_removed.append("cache file")
                logger.info("🗑️ Removed cache file")
            
            if self.key_file.exists():
                self.key_file.unlink()
                files_removed.append("encryption key")
                logger.info("🗑️ Removed encryption key")
            
            if files_removed:
                logger.info(f"✅ Cleared: {', '.join(files_removed)}")
            else:
                logger.info("ℹ️ No cache files to clear")
            
            return True
            
        except PermissionError:
            logger.error("❌ Permission denied clearing cache")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to clear cache: {e}")
            return False
    
    def is_cached(self):
        """Check if API key is cached"""
        return self.cache_file.exists()
    
    def get_cache_info(self):
        """Get detailed cache information for debugging"""
        info = {
            "cache_dir": str(self.cache_dir),
            "cache_dir_exists": self.cache_dir.exists(),
            "cache_file_exists": self.cache_file.exists(),
            "key_file_exists": self.key_file.exists(),
            "encryption_available": self.encryption_available,
            "cache_file_size": None,
            "cache_file_modified": None,
            "permissions": None
        }
        
        if self.cache_file.exists():
            try:
                stat = self.cache_file.stat()
                info["cache_file_size"] = stat.st_size
                info["cache_file_modified"] = stat.st_mtime
                info["permissions"] = oct(stat.st_mode)[-3:]
            except Exception as e:
                info["error"] = str(e)
        
        return info

# Fallback simple cache (if cryptography is not available or setup fails)
class SimpleAPIKeyCache:
    def __init__(self):
        self.cache_dir = Path.home() / ".openrouter_chatbot"
        self.cache_file = self.cache_dir / "simple_cache.json"
        
        # Create cache directory if it doesn't exist
        try:
            self.cache_dir.mkdir(exist_ok=True, mode=0o700)
            logger.info(f"✅ Simple cache directory ready: {self.cache_dir}")
        except Exception as e:
            logger.error(f"❌ Failed to create simple cache directory: {e}")
            raise
    
    def save_api_key(self, api_key, source="manual"):
        """Save API key to simple cache (base64 encoded) with error handling"""
        if not api_key or not api_key.strip():
            return False
        
        try:
            # Simple encoding (not secure, but better than plain text)
            encoded_key = base64.b64encode(api_key.strip().encode()).decode()
            
            cache_data = {
                "encoded_key": encoded_key,
                "source": source,
                "timestamp": int(time.time()),
                "hash": hashlib.md5(api_key.strip().encode()).hexdigest()[:8]  # Verification
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            os.chmod(self.cache_file, 0o600)
            logger.info("✅ API key saved with simple cache")
            return True
        except Exception as e:
            logger.error(f"❌ Simple cache save failed: {e}")
            return False
    
    def load_api_key(self):
        """Load API key from simple cache with error handling"""
        if not self.cache_file.exists():
            return None
        
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            encoded_key = cache_data.get("encoded_key")
            if encoded_key:
                decoded_key = base64.b64decode(encoded_key.encode()).decode()
                
                # Verify hash if available
                if "hash" in cache_data:
                    expected_hash = hashlib.md5(decoded_key.encode()).hexdigest()[:8]
                    if cache_data["hash"] != expected_hash:
                        logger.warning("⚠️ Simple cache integrity check failed")
                        return None
                
                logger.info("✅ API key loaded from simple cache")
                return {
                    "key": decoded_key,
                    "source": cache_data.get("source", "cached"),
                    "cached": True,
                    "method": "simple",
                    "timestamp": cache_data.get("timestamp")
                }
        except Exception as e:
            logger.error(f"❌ Simple cache load failed: {e}")
        return None
    
    def clear_cache(self):
        """Clear cached data"""
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
                logger.info("✅ Simple cache cleared")
            return True
        except Exception as e:
            logger.error(f"❌ Simple cache clear failed: {e}")
            return False
    
    def is_cached(self):
        """Check if API key is cached"""
        return self.cache_file.exists()
    
    def get_cache_info(self):
        """Get cache information"""
        return {
            "cache_dir": str(self.cache_dir),
            "cache_file_exists": self.cache_file.exists(),
            "encryption_available": False,
            "method": "simple"
        }

# Factory function to get appropriate cache with enhanced error handling
def get_api_cache():
    """Get the best available API cache implementation"""
    try:
        # Try to use the full-featured cache
        cache = APIKeyCache()
        logger.info(f"✅ Cache initialized: encryption={'available' if cache.encryption_available else 'not available'}")
        return cache
    except Exception as e:
        logger.warning(f"⚠️ Full cache failed, trying simple cache: {e}")
        try:
            # Fallback to simple cache
            cache = SimpleAPIKeyCache()
            logger.info("✅ Simple cache initialized")
            return cache
        except Exception as e2:
            logger.error(f"❌ All cache methods failed: {e2}")
            # Return a dummy cache that doesn't crash the app
            return DummyCache()

class DummyCache:
    """Emergency fallback cache that keeps everything in memory only"""
    
    def __init__(self):
        self._temp_key = None
        logger.warning("⚠️ Using dummy cache - no persistence!")
    
    def save_api_key(self, api_key, source="manual"):
        self._temp_key = api_key
        logger.info("📝 API key stored in memory only")
        return True
    
    def load_api_key(self):
        if self._temp_key:
            return {
                "key": self._temp_key,
                "source": "memory",
                "cached": False,
                "method": "dummy"
            }
        return None
    
    def clear_cache(self):
        self._temp_key = None
        return True
    
    def is_cached(self):
        return False
    
    def get_cache_info(self):
        return {
            "cache_dir": "N/A",
            "cache_file_exists": False,
            "encryption_available": False,
            "method": "dummy",
            "status": "emergency_fallback"
        }

# Test function for debugging
def test_cache():
    """Test the cache functionality"""
    print("🧪 Testing cache functionality...")
    
    cache = get_api_cache()
    test_key = "sk-test-1234567890abcdef"
    
    print(f"Cache type: {type(cache).__name__}")
    print(f"Cache info: {cache.get_cache_info()}")
    
    # Test save
    result = cache.save_api_key(test_key, "test")
    print(f"Save test: {'✅' if result else '❌'}")
    
    # Test load
    loaded = cache.load_api_key()
    if loaded and loaded.get('key') == test_key:
        print("Load test: ✅")
        print(f"Method used: {loaded.get('method', 'unknown')}")
    else:
        print("Load test: ❌")
        print(f"Loaded data: {loaded}")
    
    # Cleanup
    cache.clear_cache()
    print("Cleanup: ✅")

if __name__ == "__main__":
    test_cache()
