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
        models = client.list()
        if models and 'models' in models:
            for model in models['models']:
                print(f"  - {model['name']}")
        else:
            print("  ❌ No models found")
            print("  Run: ollama pull llama3")
            return
        
        # Check if llama3 is available
        model_names = [m['name'] for m in models['models']]
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
            print(f"Response: {response['message']['content']}")
            print("\n✅ Ollama is working correctly!")
        else:
            print("\n❌ llama3 model not found")
            print("Run: ollama pull llama3")
            
    except Exception as e:
        print(f"\n❌ Error connecting to Ollama: {e}")
        print("\nMake sure Ollama is running:")
        print("  ollama serve")

if __name__ == "__main__":
    test_ollama()
