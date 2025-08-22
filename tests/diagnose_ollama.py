#!/usr/bin/env python3
"""Diagnose Ollama setup and API structure."""

import sys
import json

def diagnose_ollama():
    """Diagnose Ollama installation and API."""
    
    print("=== Ollama Diagnostic Tool ===\n")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check ollama package
    try:
        import ollama
        print(f"✅ ollama package installed")
        print(f"   Version: {ollama.__version__ if hasattr(ollama, '__version__') else 'Unknown'}")
    except ImportError:
        print("❌ ollama package not installed")
        print("   Run: pip install ollama")
        return
    
    # Test connection methods
    print("\n--- Testing Connection Methods ---")
    
    # Method 1: Direct client
    try:
        from ollama import Client
        client = Client(host="http://localhost:11434")
        print("✅ Client created successfully")
        
        # Try to list models
        try:
            result = client.list()
            print("✅ client.list() executed")
            print(f"   Response type: {type(result)}")
            print(f"   Response keys: {result.keys() if hasattr(result, 'keys') else 'Not a dict'}")
            
            # Print full structure for debugging
            print("\n   Full response structure (first 500 chars):")
            result_str = json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
            print(f"   {result_str[:500]}...")
            
        except Exception as e:
            print(f"❌ client.list() failed: {e}")
            
    except Exception as e:
        print(f"❌ Client creation failed: {e}")
    
    # Method 2: Using module directly
    print("\n--- Testing Module Methods ---")
    try:
        models = ollama.list()
        print("✅ ollama.list() executed")
        print(f"   Response type: {type(models)}")
        
    except Exception as e:
        print(f"❌ ollama.list() failed: {e}")
    
    # Method 3: Raw HTTP test
    print("\n--- Testing Raw HTTP ---")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags")
        print(f"✅ HTTP request successful (status: {response.status_code})")
        if response.status_code == 200:
            data = response.json()
            if 'models' in data and data['models']:
                print(f"   Found {len(data['models'])} model(s)")
                for model in data['models']:
                    print(f"   - {model.get('name', 'Unknown')}")
            else:
                print("   No models found")
    except Exception as e:
        print(f"❌ HTTP request failed: {e}")
    
    print("\n=== End of Diagnostics ===")

if __name__ == "__main__":
    diagnose_ollama()
