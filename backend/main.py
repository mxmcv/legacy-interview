from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from openai import OpenAI
import os
from pathlib import Path
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv

# Load environment variables from .env file in parent directory
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mental Health Context Matcher API")

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.warning("OPENAI_API_KEY not found in environment variables")
client = OpenAI(api_key=openai_api_key) if openai_api_key else None

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "train.csv"
EMBEDDINGS_PATH = BASE_DIR / "backend" / "embeddings.npy"
CONTEXTS_PATH = BASE_DIR / "backend" / "contexts.npy"

# Global variables to store loaded data
embeddings_array = None
contexts_df = None


class SituationInput(BaseModel):
    situation: str
    top_k: int = 5


class SimilarResponse(BaseModel):
    context: str
    response: str
    similarity: float


class ChatInput(BaseModel):
    situation: str
    top_k: int = 5


class ChatResponse(BaseModel):
    generated_response: str
    similar_responses: List[SimilarResponse]


def get_embedding(text: str) -> List[float]:
    """
    Generate embedding for a given text using OpenAI API.
    """
    if not client:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file."
        )

    try:
        # Convert to string and handle NaN/None values
        if pd.isna(text):
            text = ""
        text = str(text).replace("\n", " ").strip()

        # Skip empty texts
        if not text:
            text = "No response provided"

        # Truncate to prevent token limit errors
        # OpenAI embedding model limit: 8192 tokens ≈ ~6000 characters (to be safe)
        MAX_CHARS = 6000
        if len(text) > MAX_CHARS:
            original_len = len(text)
            text = text[:MAX_CHARS] + "..."
            logger.warning(
                f"Truncated text from {original_len} to {MAX_CHARS} characters")

        response = client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error generating embedding: {str(e)}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def load_embeddings_and_contexts():
    """Load pre-generated embeddings and contexts from disk."""
    global embeddings_array, contexts_df

    if embeddings_array is None or contexts_df is None:
        if not EMBEDDINGS_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail="Embeddings not found. Please run /generate-embeddings first."
            )

        embeddings_array = np.load(EMBEDDINGS_PATH)
        contexts_df = pd.read_csv(CSV_PATH)
        logger.info(
            f"Loaded {len(embeddings_array)} embeddings and {len(contexts_df)} contexts")


@app.on_event("startup")
async def startup_event():
    """Load embeddings on startup if they exist."""
    try:
        if EMBEDDINGS_PATH.exists():
            load_embeddings_and_contexts()
            logger.info("Embeddings loaded successfully on startup")
    except Exception as e:
        logger.warning(f"Could not load embeddings on startup: {e}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Mental Health Context Matcher API",
        "endpoints": {
            "/generate-embeddings": "POST - Generate embeddings for responses from train.csv",
            "/find-similar": "POST - Find most relevant responses for a given situation",
            "/chat": "POST - Get AI-generated response using RAG with OpenAI ChatGPT",
            "/stats": "GET - Get statistics about loaded data"
        },
        "info": {
            "embedding_model": "OpenAI text-embedding-3-small",
            "llm_model": "OpenAI GPT-3.5-turbo",
            "note": "Requires OpenAI API key with credits"
        }
    }


@app.post("/generate-embeddings")
async def generate_embeddings():
    """
    Generate embeddings for all responses in train.csv and save to numpy array.
    This should be run once to prepare the data.
    Handles long texts by truncating to prevent token limit errors.
    """
    try:
        logger.info("Starting embedding generation for responses...")

        # Read CSV
        if not CSV_PATH.exists():
            raise HTTPException(
                status_code=404, detail=f"train.csv not found at {CSV_PATH}")

        df = pd.read_csv(CSV_PATH)
        logger.info(f"Loaded {len(df)} rows from train.csv")

        # Generate embeddings for responses
        embeddings = []
        batch_size = 100

        for i in range(0, len(df), batch_size):
            batch = df['Response'].iloc[i:i+batch_size]
            logger.info(
                f"Processing batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")

            for response in batch:
                embedding = get_embedding(response)
                embeddings.append(embedding)

        # Convert to numpy array and save
        embeddings_array_new = np.array(embeddings)
        np.save(EMBEDDINGS_PATH, embeddings_array_new)

        logger.info(
            f"Saved {len(embeddings_array_new)} embeddings to {EMBEDDINGS_PATH}")
        logger.info(f"Embedding shape: {embeddings_array_new.shape}")

        # Reload global variables
        global embeddings_array, contexts_df
        embeddings_array = embeddings_array_new
        contexts_df = df

        return {
            "message": "Embeddings generated successfully",
            "total_embeddings": len(embeddings_array_new),
            "embedding_dimension": embeddings_array_new.shape[1],
            "file_path": str(EMBEDDINGS_PATH)
        }

    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/find-similar", response_model=List[SimilarResponse])
async def find_similar(situation_input: SituationInput):
    """
    Find the most relevant responses from the training data for a given situation.
    Embeds the user's situation and finds responses with highest cosine similarity.
    """
    try:
        # Ensure embeddings are loaded
        load_embeddings_and_contexts()

        # Generate embedding for input situation
        query_embedding = np.array(get_embedding(situation_input.situation))

        # Calculate cosine similarity with all stored embeddings
        similarities = []
        for i, stored_embedding in enumerate(embeddings_array):
            similarity = cosine_similarity(query_embedding, stored_embedding)
            similarities.append((i, similarity))

        # Sort by similarity (highest first) and get top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_matches = similarities[:situation_input.top_k]

        # Build response
        results = []
        for idx, similarity in top_matches:
            results.append(SimilarResponse(
                context=contexts_df.iloc[idx]['Context'],
                response=contexts_df.iloc[idx]['Response'],
                similarity=float(similarity)
            ))

        return results

    except Exception as e:
        logger.error(f"Error finding similar contexts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(chat_input: ChatInput):
    """
    Process user situation with RAG approach:
    1. Find similar responses from training data (using OpenAI embeddings)
    2. Use OpenAI ChatGPT to synthesize an ideal response based on retrieved responses
    """
    try:
        # Ensure embeddings are loaded
        load_embeddings_and_contexts()

        # Generate embedding for input situation
        query_embedding = np.array(get_embedding(chat_input.situation))

        # Calculate cosine similarity with all stored embeddings
        similarities = []
        for i, stored_embedding in enumerate(embeddings_array):
            similarity = cosine_similarity(query_embedding, stored_embedding)
            similarities.append((i, similarity))

        # Sort by similarity (highest first) and get top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_matches = similarities[:chat_input.top_k]

        # Build similar responses list
        similar_responses = []
        retrieved_responses_text = []

        for idx, similarity in top_matches:
            context = contexts_df.iloc[idx]['Context']
            response = contexts_df.iloc[idx]['Response']

            similar_responses.append(SimilarResponse(
                context=context,
                response=response,
                similarity=float(similarity)
            ))

            # Collect responses for OpenAI prompt
            retrieved_responses_text.append(
                f"Response {len(retrieved_responses_text) + 1}:\n{response}")

        # Create prompt for OpenAI
        retrieved_context = "\n\n".join(retrieved_responses_text)

        system_prompt = """You are a compassionate and professional mental health counselor. 
Using the provided reference responses as inspiration, provide thoughtful, empathetic, and professional advice.
Your response should be compassionate, validating, offer practical guidance, and maintain a supportive tone."""

        user_prompt = f"""A person has shared the following situation:

{chat_input.situation}

Based on similar cases, here are some relevant professional responses for reference:

{retrieved_context}

Please provide a personalized response to the user's situation, drawing from the above references but adapting them specifically to their needs."""

        # Generate response using OpenAI ChatGPT
        if not client:
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file."
            )

        logger.info("Generating response with OpenAI ChatGPT...")
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        generated_text = completion.choices[0].message.content

        return ChatResponse(
            generated_response=generated_text,
            similar_responses=similar_responses
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get statistics about the loaded embeddings and contexts."""
    try:
        load_embeddings_and_contexts()

        return {
            "total_contexts": len(contexts_df),
            "total_embeddings": len(embeddings_array),
            "embedding_dimension": embeddings_array.shape[1],
            "embeddings_file": str(EMBEDDINGS_PATH),
            "csv_file": str(CSV_PATH)
        }
    except Exception as e:
        return {
            "error": str(e),
            "embeddings_generated": EMBEDDINGS_PATH.exists()
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
