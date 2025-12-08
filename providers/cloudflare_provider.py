import requests
import os
from providers.base_provider import BaseProvider

class CloudflareProvider(BaseProvider):
    def __init__(self):
        self.api_base_url = os.getenv("CLOUDFLARE_API_BASE_URL", "https://api.cloudflare.com/client/v4/accounts/03157a3894e23338c180e62608b43b2d/ai/run/")
        self.api_token = os.getenv("CLOUDFLARE_API_TOKEN")
        if not self.api_token:
            raise ValueError("CLOUDFLARE_API_TOKEN environment variable not set.")
        self.headers = {"Authorization": f"Bearer {self.api_token}"}

    def run(self, model_name, messages):
        input_data = {"messages": messages}
        response = requests.post(f"{self.api_base_url}{model_name}", headers=self.headers, json=input_data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()

    def get_supported_models(self):
        return ["@cf/meta/llama-3-8b-instruct"] # Example model, add more as needed