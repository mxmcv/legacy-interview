# Mental Health Support Chat - Frontend

A clean, simple chat interface for the mental health support application.

## Features

- 💬 Clean, modern chat interface
- 🎨 Beautiful gradient design with smooth animations
- 📱 Fully responsive (mobile & desktop)
- ⌨️ Enter to send, Shift+Enter for new line
- 🔄 Loading indicator while processing
- 🎯 Auto-scroll to latest messages
- ⚡ Pure HTML/CSS/JavaScript (no frameworks needed)

## Quick Start

### Option 1: Open Directly in Browser

Simply open `index.html` in your web browser:

```bash
# On Mac
open index.html

# On Linux
xdg-open index.html

# On Windows
start index.html
```

### Option 2: Use Python HTTP Server (Recommended)

```bash
cd frontend
python3 -m http.server 3000
```

Then visit: `http://localhost:3000`

### Option 3: Use Node.js HTTP Server

```bash
cd frontend
npx http-server -p 3000
```

Then visit: `http://localhost:3000`

## Prerequisites

**Before using the frontend, make sure the backend is running!**

1. Start the backend server:

```bash
cd backend
python main.py
```

2. Ensure embeddings are generated:

```bash
curl -X POST http://localhost:8000/generate-embeddings
```

The backend should be running on `http://localhost:8000`

## How to Use

1. Open the chat interface in your browser
2. Type your situation, question, or feelings in the input box
3. Press Enter or click "Send"
4. Wait for the AI-powered response (uses RAG with OpenAI GPT-3.5-turbo)
5. Continue the conversation as needed

## Configuration

### Changing API URL

If your backend is running on a different port, edit the `API_BASE_URL` in `index.html`:

```javascript
const API_BASE_URL = 'http://localhost:8000'; // Change this if needed
```

## Troubleshooting

### "Could not connect to the server"

**Solution:** Make sure the backend is running:

```bash
cd backend
python main.py
```

### CORS Errors

**Solution:** The backend already has CORS enabled for all origins. If you still see CORS errors, try:

1. Running the frontend through a local server (Option 2 or 3 above) instead of opening the file directly
2. Check that the backend URL is correct

### "Embeddings not found" Error

**Solution:** Generate embeddings first:

```bash
curl -X POST http://localhost:8000/generate-embeddings
```

## Tech Stack

- **HTML5** - Structure
- **CSS3** - Styling with gradients, animations, and flexbox
- **Vanilla JavaScript** - All functionality (no frameworks)
- **Fetch API** - HTTP requests to backend

## Features Breakdown

### UI/UX

- Smooth message animations
- Typing indicator with bouncing dots
- Auto-resizing textarea
- Gradient theme (purple/blue)
- Clean, minimalist design
- Mobile-optimized layout

### Functionality

- Real-time chat messaging
- Error handling and user feedback
- Keyboard shortcuts (Enter to send)
- Automatic scrolling to new messages
- Loading states during API calls

## Future Enhancements

Potential features to add:

- Message history persistence (localStorage)
- Copy response to clipboard
- Show similar responses used (transparency)
- Dark mode toggle
- Export chat history
- Markdown support in responses
- Voice input
