import google.generativeai as genai

# Check available attributes
print("Available attributes in genai:")
print([attr for attr in dir(genai) if not attr.startswith('_')])

# Try different ways to initialize
api_key = 'AIzaSyDmO1Ynd1rd3IN7kEolIMRlwNTj-iLcqCw'
genai.configure(api_key=api_key)

# Check for model creation methods
if hasattr(genai, 'GenerativeModel'):
    print("\nGenerativeModel found directly")
elif hasattr(genai, 'Model'):
    print("\nModel found")
elif hasattr(genai, 'Client'):
    print("\nClient found")