#!/usr/bin/env python3
"""
Configuration validation and testing script.
"""
import sys
import os
sys.path.append('/root/projects/test_task')

from backend.config import config
import json

def validate_config():
    """Validate the current configuration."""
    print("🔧 RAG System Configuration Validation")
    print("=" * 50)
    
    # Validate configuration
    issues = config.validate()
    
    if issues["errors"]:
        print("❌ Configuration Errors:")
        for error in issues["errors"]:
            print(f"  - {error}")
    
    if issues["warnings"]:
        print("⚠️  Configuration Warnings:")
        for warning in issues["warnings"]:
            print(f"  - {warning}")
    
    if not issues["errors"] and not issues["warnings"]:
        print("✅ Configuration is valid!")
    
    print("\n📊 Current Configuration:")
    print("-" * 30)
    
    # Display configuration info
    env_info = config.get_env_vars_info()
    
    print("Database:")
    for key, value in env_info["database"].items():
        print(f"  {key}: {value}")
    
    print("\nLLM:")
    for key, value in env_info["llm"].items():
        print(f"  {key}: {value}")
    
    print("\nAPI:")
    for key, value in env_info["api"].items():
        print(f"  {key}: {value}")
    
    print("\nUI:")
    for key, value in env_info["ui"].items():
        print(f"  {key}: {value}")
    
    # Test directory creation
    print("\n📁 Directory Status:")
    directories = [
        ("Data folder", config.ingestion.data_folder),
        ("ChromaDB persist", config.database.chroma_persist_dir),
        ("Embedding cache", config.embedding.cache_dir),
    ]
    
    for name, path in directories:
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"  {status} {name}: {path}")
    
    return len(issues["errors"]) == 0

def test_config_access():
    """Test accessing various configuration values."""
    print("\n🧪 Testing Configuration Access:")
    print("-" * 30)
    
    try:
        # Test basic access
        print(f"✅ LLM Provider: {config.llm.default_provider}")
        print(f"✅ LLM Model: {config.llm.default_model}")
        print(f"✅ API Port: {config.api.port}")
        print(f"✅ Chunk Size: {config.ingestion.chunk_size}")
        print(f"✅ Neo4j URI: {config.database.neo4j_uri}")
        
        # Test boolean conversions
        print(f"✅ Debug Mode: {config.api.debug}")
        print(f"✅ Graph Expansion: {config.retrieval.enable_graph_expansion}")
        
        # Test list conversions
        print(f"✅ CORS Origins: {config.api.cors_origins}")
        print(f"✅ Allowed Hosts: {config.security.allowed_hosts}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration access failed: {e}")
        return False

def show_environment_variables():
    """Show which environment variables are being used."""
    print("\n🌍 Environment Variables:")
    print("-" * 30)
    
    env_vars = [
        "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD",
        "DEFAULT_LLM_PROVIDER", "DEFAULT_LLM_MODEL", 
        "API_PORT", "STREAMLIT_PORT",
        "OPENAI_API_KEY", "OLLAMA_BASE_URL",
        "DATA_FOLDER", "CHUNK_SIZE"
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            # Hide sensitive values
            if "PASSWORD" in var or "KEY" in var:
                display_value = "***"
            else:
                display_value = value
            print(f"  ✅ {var}: {display_value}")
        else:
            print(f"  ❌ {var}: (not set, using default)")

def main():
    """Main validation function."""
    try:
        # Test configuration loading
        print("Loading configuration...")
        
        # Run validation
        is_valid = validate_config()
        
        # Test configuration access
        access_works = test_config_access()
        
        # Show environment variables
        show_environment_variables()
        
        # Overall status
        print("\n" + "=" * 50)
        if is_valid and access_works:
            print("🎉 Configuration is working correctly!")
            return 0
        else:
            print("💥 Configuration has issues that need to be resolved!")
            return 1
            
    except Exception as e:
        print(f"💥 Configuration validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
