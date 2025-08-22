#!/usr/bin/env python3
"""Test Ollama connection and model availability."""

import ollama
from ollama import Client

def test_ollama():
    """Test Ollama connection and model availability."""
    print("Testing Ollama connection...")
    
    try:
        # Create client
        client = Client(host="http://localhost:11434")
        print("✅ Connected to Ollama")
        
        # List available models
        print("\nAvailable models:")
        models_response = client.list()
        
        # The response is a ListResponse object with a 'models' attribute
        if hasattr(models_response, 'models'):
            models = models_response.models
            for model in models:
                # Each model is an object with attributes like 'name', 'modified_at', etc.
                model_name = model.name if hasattr(model, 'name') else str(model)
                print(f"  - {model_name}")
                
            # Check if llama3 is available
            model_names = [m.name for m in models if hasattr(m, 'name')]
            if any('llama3' in name for name in model_names):
                print("\n✅ llama3 model is available")
                
                # Test the model
                print("\nTesting llama3 model...")
                response = client.chat(
                    model='llama3',
                    messages=[
                        {'role': 'user', 'content': 'Say hello in 5 words or less'}
                    ]
                )
                
                # The response is a ChatResponse object
                if hasattr(response, 'message'):
                    print(f"Response: {response.message.content}")
                else:
                    print(f"Response: {response}")
                    
                print("\n✅ Ollama is working correctly!")
            else:
                print("\n❌ llama3 model not found")
                print("Run: ollama pull llama3")
        else:
            print("  ❌ Unexpected response structure")
            print(f"  Response type: {type(models_response)}")
            print(f"  Response: {models_response}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        
        # Try a simpler approach - just test if we can use the model
        print("\nTrying direct model test...")
        try:
            client = Client(host="http://localhost:11434")
            response = client.chat(
                model='llama3',
                messages=[
                    {'role': 'user', 'content': 'Say hello in 5 words or less'}
                ]
            )
            
            if hasattr(response, 'message'):
                print(f"Response: {response.message.content}")
            else:
                print(f"Response: {response}")
                
            print("\n✅ Ollama and llama3 are working!")
        except Exception as inner_e:
            print(f"❌ Direct test failed: {inner_e}")
            print("\nMake sure Ollama is running:")
            print("  ollama serve")

if __name__ == "__main__":
    test_ollama()
