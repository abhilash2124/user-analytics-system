from google import genai
from google.genai import types
from app.config import settings

# Force the SDK client to target the v1 endpoint so gemini-embedding-001 is found
client = genai.Client(
    api_key=settings.GEMINI_API_KEY, http_options=types.HttpOptions(api_version="v1")
)


def generate_embedding(text: str) -> list[float]:
    """
    Transforms plain text event logs into vector footprints using gemini-embedding-001 over v1.
    Outputs a standard 768-dimension list array.
    """
    try:
        response = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"Error calling Gemini Embedding API: {e}")
        raise e

