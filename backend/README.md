# Mental Health Context Matcher API

A FastAPI backend that uses RAG (Retrieval Augmented Generation) to provide mental health support responses. It combines vector embeddings for similarity search with OpenAI GPT-3.5-turbo for response generation.

## Setup

1. Install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

2. Set up your API key:

**Required:**

- OpenAI API key (for embeddings and LLM)

```bash
export OPENAI_API_KEY='your-openai-key-here'
```

Or create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-openai-key-here
```

**Note:** Make sure you have credits in your OpenAI account:
https://platform.openai.com/account/billing

## Running the Server

```bash
cd backend
python main.py
```

Or with uvicorn:

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Generate Embeddings (First-time setup)

**POST** `/generate-embeddings`

Generates embeddings for all **responses** in `train.csv` and saves them to `embeddings.npy`.

**Features:**

- Processes all 3,500+ responses from the training dataset
- Automatically truncates very long responses (>6,000 chars) to prevent token limit errors
- Handles NaN/empty values gracefully
- Batch processing for efficiency

**Note:** This only needs to be run once, or whenever you update the training data.

```bash
curl -X POST http://localhost:8000/generate-embeddings
```

**Response:**

```json
{
  "message": "Embeddings generated successfully",
  "total_embeddings": 3512,
  "embedding_dimension": 1536,
  "file_path": "/path/to/embeddings.npy"
}
```

**Time:** ~5-10 minutes  
**Cost Estimate:** ~$0.15-$0.25 using `text-embedding-3-small`

### 2. Find Similar Contexts

**POST** `/find-similar`

Finds the most similar contexts from the training data for a given situation.

**Request Body:**

```json
{
  "situation": "I feel anxious and overwhelmed at work",
  "top_k": 5
}
```

**Response:**

```json
[
  {
    "context": "Patient's original context...",
    "response": "Therapist's response...",
    "similarity": 0.89
  },
  ...
]
```

**Example:**

```bash
curl -X POST http://localhost:8000/find-similar \
  -H "Content-Type: application/json" \
  -d '{"situation": "I feel anxious and overwhelmed at work", "top_k": 3}'
```

### 3. Chat with AI (RAG-powered Response)

**POST** `/chat`

**⭐ Recommended endpoint for chat UI** - Combines retrieval with AI generation for personalized responses.

Finds similar responses and uses OpenAI GPT-3.5-turbo to synthesize an ideal, personalized response.

**Request Body:**

```json
{
  "situation": "I feel anxious and overwhelmed at work",
  "top_k": 5
}
```

**Response:**

```json
{
  "generated_response": "Thank you for sharing what you're experiencing. Work-related anxiety and feeling overwhelmed are common challenges...",
  "similar_responses": [
    {
      "context": "Original patient context...",
      "response": "Original therapist response...",
      "similarity": 0.89
    },
    ...
  ]
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"situation": "I feel anxious and overwhelmed at work", "top_k": 5}'
```

**How it works:**

1. Retrieves top K most relevant therapist responses based on user situation
2. Passes retrieved responses to OpenAI GPT-3.5-turbo as context
3. GPT synthesizes a personalized, empathetic response
4. Returns both the AI-generated response and the source responses

### 4. Get Statistics

**GET** `/stats`

Returns statistics about the loaded embeddings and contexts.

```bash
curl http://localhost:8000/stats
```

**Response:**

```json
{
  "total_contexts": 6304,
  "total_embeddings": 6304,
  "embedding_dimension": 1536,
  "embeddings_file": "/path/to/embeddings.npy",
  "csv_file": "/path/to/train.csv"
}
```

## Workflow

1. **Initial Setup:** Run `/generate-embeddings` to create embeddings from `train.csv` (one-time)
2. **For Chat UI:** Use `/chat` endpoint to get AI-generated responses (recommended)
3. **For Raw Data:** Use `/find-similar` endpoint to get similar responses without AI generation

## How It Works (RAG Architecture)

### Embedding Generation (One-time Setup)

1. Reads all therapist **responses** from `train.csv`
2. Uses OpenAI's `text-embedding-3-small` to convert each response into a 1536-dimensional vector
3. Stores embeddings in NumPy array (`embeddings.npy`) for fast loading

### Chat Endpoint (RAG Pipeline)

1. **Retrieval**: User's situation is embedded and compared against all stored response embeddings using cosine similarity
2. **Context Building**: Top K most similar responses are retrieved as context
3. **Augmentation**: Retrieved responses are formatted into a prompt for the LLM
4. **Generation**: OpenAI GPT-3.5-turbo synthesizes a personalized response based on retrieved context
5. **Response**: Returns both the AI-generated response and the source responses used

### Text Truncation

- Automatically truncates responses longer than 6,000 characters during embedding generation
- Prevents OpenAI token limit errors (8,192 token max)
- Logs warnings when truncation occurs

## API Documentation

Interactive API documentation is available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
