import openai
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
# Do not set api_version explicitly
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

def transaction_to_text(txn: dict) -> str:
    return (
        f"User {txn['user_id']} spent ${txn['amount']} at {txn['merchant']} "
        f"in {txn['location']} on {txn['timestamp']} for {txn['category']}"
    )

def get_embedding(text: str) -> list:
    # For openai>=1.0.0, use the new AzureOpenAI client
    client = openai.AzureOpenAI(
        api_key=openai.api_key,
        azure_endpoint=openai.api_base,
        # api_version is omitted
    )
    response = client.embeddings.create(
        input=[text],
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    )
    return response.data[0].embedding 