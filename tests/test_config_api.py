#!/usr/bin/env python3
"""
Test if the API can load configuration from .env file correctly.
"""
import sys
import os
sys.path.append('/root/projects/test_task')

def test_config_loading():
    """Test configuration loading."""
    print("🧪 Testing Configuration Loading")
    print("=" * 40)
    
    try:
        from backend.config import config
        
        print("✅ Configuration loaded successfully!")
        print(f"📊 LLM Provider: {config.llm.default_provider}")
        print(f"🤖 LLM Model: {config.llm.default_model}")
        print(f"🌡️ Temperature: {config.llm.default_temperature}")
        print(f"🔑 OpenAI API Key: {'***' if config.llm.openai_api_key else 'Not set'}")
        print(f"🌐 API Port: {config.api.port}")
        
        # Test if configuration is valid for OpenAI
        if config.llm.default_provider.lower() == "openai":
            if config.llm.openai_api_key:
                print("✅ OpenAI configuration is complete!")
            else:
                print("❌ OpenAI provider selected but no API key found!")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_initialization():
    """Test if the API can initialize with the configuration."""
    print("\n🚀 Testing API Initialization")
    print("=" * 40)
    
    try:
        from backend.rag.pipeline import RAGPipeline, ModelProvider
        from backend.config import config
        
        # Test creating RAG pipeline with config
        provider = ModelProvider.OLLAMA if config.llm.default_provider.lower() == "ollama" else ModelProvider.OPENAI
        
        print(f"🔧 Creating RAG pipeline with {provider.value}...")
        
        pipeline = RAGPipeline(
            provider=provider,
            model_name=config.llm.default_model,
            temperature=config.llm.default_temperature,
            openai_api_key=config.llm.openai_api_key
        )
        
        print("✅ RAG pipeline created successfully!")
        print(f"📋 Provider: {pipeline.provider.value}")
        print(f"📋 Model: {pipeline.model_name}")
        print(f"📋 Temperature: {pipeline.temperature}")
        
        return True
        
    except Exception as e:
        print(f"❌ API initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    config_ok = test_config_loading()
    
    if config_ok:
        api_ok = test_api_initialization()
        
        if api_ok:
            print("\n🎉 Everything looks good!")
            print("💡 Your .env configuration will be used automatically.")
            print("💡 You don't need to configure via UI unless you want to change settings.")
            return 0
        else:
            print("\n💥 API initialization failed!")
            return 1
    else:
        print("\n💥 Configuration loading failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
