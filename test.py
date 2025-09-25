# To run this code you need to install the following dependencies:
# pip install google-genai
from google import genai
from google.genai import types

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key="AIzaSyCU9c8t1J6chDuQzAa1teYj1SXpD8hF8mc")

response = client.models.generate_content(
    model="gemini-2.5-flash-lite", contents="什么是量子力学",
        config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
    ),

)
print(response.text)