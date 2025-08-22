#!/usr/bin/env python3
"""Alternative Ollama test using requests to debug the API."""

import requests
import json

def test_ollama_raw():
    """Test Ollama using raw HTTP requests."""
    base_url = "http://localhost:11434"
    
    print("Testing Ollama API directly...")
    
    # Test 1: Check if server is running
    try:
        response = requests.get(f"{base_url}/api/tags")
        print(f"✅ Server is running (status: {response.status_code})")
        
        if response.status_code == 200:
            data = response.json()
            print("\nAvailable models:")
            if 'models' in data:
                for model in data['models']:
                    print(f"  - {model.get('name', 'Unknown')}")
                    # Print full model info for debugging
                    # print(f"    Full info: {json.dumps(model, indent=4)}")
            else:
                print("  No models found")
        else:
            print(f"  Server returned: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama at http://localhost:11434")
        print("Make sure Ollama is running: ollama serve")
        return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Test 2: Try to use llama3
    print("\nTesting llama3 model...")
    try:
        response = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": "llama3",
                "messages": [
                    {"role": "user", "content": "Say hello in 5 words or less"}
                ],
                "stream": False
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model response: {data['message']['content']}")
            print("\n✅ Ollama and llama3 are working correctly!")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing model: {e}")

if __name__ == "__main__":
    test_ollama_raw()
