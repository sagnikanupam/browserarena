#!/usr/bin/env python3
"""
Test if the OpenAI API key is working
"""

from openai import OpenAI

# Your API key
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

try:
    # Simple test call
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "user", "content": "Say 'API is working' if you can read this."}
        ],
        max_tokens=10
    )
    
    print("✅ API Key is working!")
    print(f"Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"❌ API Key error: {e}")