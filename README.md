# Mental Health Support Chat Application

A full-stack mental health support chat application using RAG (Retrieval Augmented Generation) to provide empathetic, context-aware responses.

## 🏗️ Architecture

```
┌─────────────────┐
│   Frontend      │  HTML/CSS/JS Chat Interface
│   (Port 3000)   │
└────────┬────────┘
         │
         │ HTTP/JSON
         ▼
┌─────────────────┐
│   Backend       │  FastAPI + RAG Pipeline
│   (Port 8000)   │
└────────┬────────┘
         │
         ├──► OpenAI API (Embeddings)
         ├──► OpenAI API (GPT-3.5-turbo LLM)
         └──► NumPy (Vector Storage)
```

## 🎯 How It Works

1. **Embedding Generation** (One-time setup)

   - Reads 3,500+ therapist responses from `train.csv`
   - Generates vector embeddings using OpenAI
   - Truncates very long responses (>6,000 chars) to prevent token limits
   - Stores in NumPy array for fast retrieval

2. **User Interaction** (Real-time)
   - User types situation in chat interface
   - Backend embeds the situation
   - Finds top 5 most similar responses via cosine similarity
   - Passes them to OpenAI GPT-3.5-turbo as context
   - GPT generates personalized response
   - Response displayed in chat UI

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key with credits ([Get one here](https://platform.openai.com/api-keys))
  - Add billing/credits at: https://platform.openai.com/account/billing

### 1. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Set Up API Key

```bash
export OPENAI_API_KEY='your-openai-key-here'
```

Or create a `.env` file in the project root:

```
OPENAI_API_KEY=your-openai-key-here
```

### 3. Generate Embeddings (One-time, ~$0.15-0.25)

```bash
# Start the backend
cd backend
python main.py

# In another terminal, generate embeddings
curl -X POST http://localhost:8000/generate-embeddings
```

This will process all 3,500+ responses and save embeddings to `backend/embeddings.npy`.
Takes ~5-10 minutes. Automatically truncates very long responses to prevent token limits.

### 4. Start Frontend

```bash
cd frontend
python3 -m http.server 3000
```

Or simply open `frontend/index.html` in your browser.

### 5. Use the App!

1. Visit `http://localhost:3000`
2. Type a situation or question
3. Get an empathetic, AI-powered response
4. Continue the conversation

## 📁 Project Structure

```
Legacy-Interview/
├── train.csv                    # Training data (6,304 context-response pairs)
├── backend/
│   ├── main.py                  # FastAPI application
│   ├── requirements.txt         # Python dependencies
│   ├── README.md               # Backend documentation
│   ├── embeddings.npy          # Generated embeddings (created after setup)
│   └── contexts.npy            # Cached contexts (optional)
└── frontend/
    ├── index.html              # Chat interface (single-file app)
    └── README.md               # Frontend documentation
```

## 🔌 API Endpoints

### `POST /generate-embeddings`

Generate embeddings from train.csv (one-time setup)

### `POST /chat`

Get AI-generated response using RAG pipeline

```json
{
  "situation": "I feel anxious and overwhelmed",
  "top_k": 5
}
```

### `POST /find-similar`

Get similar responses without AI generation (raw data)

### `GET /stats`

Get statistics about loaded data

## 💰 Cost Estimate

- **Initial Setup**: ~$0.15-0.25 (one-time embedding generation for 3,500+ responses)
- **Per Query**: ~$0.001 (embedding) + ~$0.001-0.002 (GPT-3.5-turbo) = **~$0.002-0.003 per conversation**

Very affordable for an MVP! Less than $1 for complete setup and testing.

## 🎨 Features

### Frontend

- Clean, modern chat interface
- Smooth animations and transitions
- Mobile-responsive design
- Typing indicators
- Error handling
- Auto-scrolling

### Backend

- RAG pipeline for context-aware responses
- Vector similarity search
- FastAPI with automatic docs
- CORS-enabled for frontend
- Error handling and logging
- Efficient NumPy storage

## 📚 Technologies Used

**Frontend:**

- HTML5, CSS3, Vanilla JavaScript
- Gradient design with animations
- Fetch API for HTTP requests

**Backend:**

- FastAPI (Python web framework)
- OpenAI API (text-embedding-3-small for embeddings)
- OpenAI API (gpt-3.5-turbo for LLM)
- NumPy (vector storage)
- Pandas (data processing)

## 🔍 Example Usage

**User Input:**

> "I'm feeling really anxious about an upcoming presentation at work. I can't sleep and keep overthinking everything."

**System Process:**

1. Embeds the user's situation using OpenAI
2. Finds 5 most similar therapist responses via cosine similarity
3. Sends them to GPT-3.5-turbo with the user's situation
4. GPT synthesizes a personalized, empathetic response

**AI Response:**

> "Thank you for sharing what you're going through. Anxiety about presentations is very common, and your feelings are completely valid. It sounds like you're experiencing anticipatory anxiety, which can indeed affect sleep and lead to overthinking..."

## 🛠️ Development

### View API Documentation

Once the backend is running, visit:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Testing the API

```bash
# Test chat endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"situation": "I feel anxious", "top_k": 5}'

# Check stats
curl http://localhost:8000/stats
```

## ⚠️ Disclaimer

This application is for informational and educational purposes only. It is not a substitute for professional mental health care. If you're experiencing a mental health crisis, please contact a qualified healthcare provider or emergency services.

