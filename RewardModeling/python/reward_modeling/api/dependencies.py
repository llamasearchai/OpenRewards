"""
FastAPI dependencies for authentication, rate limiting, and resource management.
"""

import os
import time
import jwt
import redis
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from typing import Dict, Optional
import logging
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)

# Global state for dependency injection
_redis_client: Optional[redis.Redis] = None
_db_engine = None
_db_session_factory = None

# Rate limiting storage
rate_limit_store: Dict[str, Dict[str, int]] = {}

# Authentication configuration
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        
    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed based on rate limit."""
        current_time = int(time.time())
        window_start = current_time - self.window_seconds
        
        if key not in rate_limit_store:
            rate_limit_store[key] = {}
        
        # Clean old entries
        user_requests = rate_limit_store[key]
        rate_limit_store[key] = {
            timestamp: count 
            for timestamp, count in user_requests.items() 
            if int(timestamp) > window_start
        }
        
        # Count requests in current window
        total_requests = sum(rate_limit_store[key].values())
        
        if total_requests >= self.max_requests:
            return False
        
        # Add current request
        current_minute = str(current_time // 60)  # Group by minute
        rate_limit_store[key][current_minute] = rate_limit_store[key].get(current_minute, 0) + 1
        
        return True

# Initialize rate limiter
default_rate_limiter = RateLimiter(max_requests=100, window_seconds=3600)

def get_redis() -> redis.Redis:
    """Get Redis client."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0,
            decode_responses=True
        )
    return _redis_client

async def get_db() -> AsyncSession:
    """Get database session."""
    global _db_engine, _db_session_factory
    
    if _db_engine is None:
        database_url = os.getenv(
            "DATABASE_URL", 
            "postgresql+asyncpg://rmep:password@localhost/reward_modeling"
        )
        _db_engine = create_async_engine(database_url)
        _db_session_factory = sessionmaker(
            _db_engine, class_=AsyncSession, expire_on_commit=False
        )
    
    async with _db_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()

def create_access_token(data: dict) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> dict:
    """Verify JWT token and return payload."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    """Get current authenticated user."""
    try:
        payload = verify_token(credentials.credentials)
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return {"user_id": user_id, "payload": payload}
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

def rate_limit(
    request: Request,
    max_requests: int = 100,
    window_seconds: int = 3600,
    user_id: Optional[str] = None
):
    """Rate limiting dependency."""
    # Create rate limit key based on user or IP
    if user_id:
        key = f"user:{user_id}"
    else:
        client_ip = request.client.host
        key = f"ip:{client_ip}"
    
    # Use custom rate limiter or default
    limiter = RateLimiter(max_requests, window_seconds)
    
    if not limiter.is_allowed(key):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )

def require_permissions(*required_permissions: str):
    """Decorator for requiring specific permissions."""
    def permission_dependency(current_user: dict = Depends(get_current_user)):
        user_permissions = current_user["payload"].get("permissions", [])
        
        for permission in required_permissions:
            if permission not in user_permissions:
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission '{permission}' required"
                )
        
        return current_user
    
    return permission_dependency

class APIKeyAuth:
    """API Key authentication."""
    
    def __init__(self):
        self.api_keys = self._load_api_keys()
    
    def _load_api_keys(self) -> Dict[str, dict]:
        """Load API keys from environment or database."""
        # In production, load from secure storage
        default_key = os.getenv("DEFAULT_API_KEY", "rmep-default-key-change-me")
        return {
            default_key: {
                "name": "default",
                "permissions": ["read", "write", "admin"],
                "created_at": datetime.utcnow().isoformat()
            }
        }
    
    def verify_api_key(self, api_key: str) -> dict:
        """Verify API key and return associated data."""
        if api_key not in self.api_keys:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        return self.api_keys[api_key]

# Global API key authenticator
api_key_auth = APIKeyAuth()

def get_api_key_user(request: Request):
    """Extract user info from API key."""
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    return api_key_auth.verify_api_key(api_key)

class ResourceTracker:
    """Track resource usage per user/session."""
    
    def __init__(self):
        self.usage_store = {}
    
    def track_model_load(self, user_id: str, model_size_mb: float):
        """Track model loading resource usage."""
        if user_id not in self.usage_store:
            self.usage_store[user_id] = {
                "models_loaded": 0,
                "total_model_size_mb": 0,
                "api_calls": 0,
                "last_activity": datetime.utcnow().isoformat()
            }
        
        self.usage_store[user_id]["models_loaded"] += 1
        self.usage_store[user_id]["total_model_size_mb"] += model_size_mb
        self.usage_store[user_id]["last_activity"] = datetime.utcnow().isoformat()
    
    def track_api_call(self, user_id: str):
        """Track API call."""
        if user_id not in self.usage_store:
            self.usage_store[user_id] = {
                "models_loaded": 0,
                "total_model_size_mb": 0,
                "api_calls": 0,
                "last_activity": datetime.utcnow().isoformat()
            }
        
        self.usage_store[user_id]["api_calls"] += 1
        self.usage_store[user_id]["last_activity"] = datetime.utcnow().isoformat()
    
    def get_usage(self, user_id: str) -> dict:
        """Get usage statistics for user."""
        return self.usage_store.get(user_id, {})

# Global resource tracker
resource_tracker = ResourceTracker()

def check_resource_limits(current_user: dict = Depends(get_current_user)):
    """Check if user has exceeded resource limits."""
    user_id = current_user["user_id"]
    usage = resource_tracker.get_usage(user_id)
    
    # Example limits - configure based on your needs
    max_models = current_user["payload"].get("max_models", 5)
    max_model_size_mb = current_user["payload"].get("max_model_size_mb", 10000)
    
    if usage.get("models_loaded", 0) >= max_models:
        raise HTTPException(
            status_code=429,
            detail=f"Maximum number of models ({max_models}) exceeded"
        )
    
    if usage.get("total_model_size_mb", 0) >= max_model_size_mb:
        raise HTTPException(
            status_code=429,
            detail=f"Maximum model size limit ({max_model_size_mb}MB) exceeded"
        )
    
    # Track this API call
    resource_tracker.track_api_call(user_id)
    
    return current_user

class CacheManager:
    """Manage caching for API responses."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.default_ttl = 3600  # 1 hour
    
    def get_cache_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key from parameters."""
        key_parts = [prefix]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")
        key_string = ":".join(key_parts)
        
        # Hash long keys
        if len(key_string) > 200:
            key_string = hashlib.md5(key_string.encode()).hexdigest()
        
        return key_string
    
    async def get(self, key: str):
        """Get cached value."""
        try:
            value = self.redis_client.get(key)
            if value:
                import json
                return json.loads(value)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        return None
    
    async def set(self, key: str, value, ttl: Optional[int] = None):
        """Set cached value."""
        try:
            import json
            self.redis_client.setex(
                key,
                ttl or self.default_ttl,
                json.dumps(value, default=str)
            )
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    async def delete(self, key: str):
        """Delete cached value."""
        try:
            self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")

def get_cache_manager() -> CacheManager:
    """Get cache manager dependency."""
    redis_client = get_redis()
    return CacheManager(redis_client)

# Health check dependencies
def check_system_health():
    """Check overall system health."""
    health_status = {
        "api": "healthy",
        "database": "unknown",
        "redis": "unknown",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Check Redis
    try:
        redis_client = get_redis()
        redis_client.ping()
        health_status["redis"] = "healthy"
    except Exception as e:
        health_status["redis"] = f"unhealthy: {str(e)}"
    
    # Check database would require async context
    # This is a simplified version
    
    return health_status

# Background task tracking
class BackgroundTaskTracker:
    """Track background tasks and their status."""
    
    def __init__(self):
        self.tasks = {}
    
    def add_task(self, task_id: str, task_type: str, user_id: str):
        """Add a background task to tracking."""
        self.tasks[task_id] = {
            "type": task_type,
            "user_id": user_id,
            "status": "running",
            "created_at": datetime.utcnow().isoformat(),
            "progress": 0
        }
    
    def update_task(self, task_id: str, status: str = None, progress: float = None):
        """Update task status."""
        if task_id in self.tasks:
            if status:
                self.tasks[task_id]["status"] = status
            if progress is not None:
                self.tasks[task_id]["progress"] = progress
            self.tasks[task_id]["updated_at"] = datetime.utcnow().isoformat()
    
    def get_task(self, task_id: str) -> Optional[dict]:
        """Get task information."""
        return self.tasks.get(task_id)
    
    def get_user_tasks(self, user_id: str) -> list:
        """Get all tasks for a user."""
        return [
            {**task, "id": task_id}
            for task_id, task in self.tasks.items()
            if task["user_id"] == user_id
        ]

# Global task tracker
task_tracker = BackgroundTaskTracker()

def get_task_tracker() -> BackgroundTaskTracker:
    """Get task tracker dependency."""
    return task_tracker 