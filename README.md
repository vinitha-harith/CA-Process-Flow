# CA-Process-Flow

Client Advisory horizontal process flow built with React + TypeScript (Vite) and a FastAPI backend.

## Steps

1. **Teams Call with Client** — pre-completed starting point
2. **Call Transcription** — paste a Teams meeting URL
3. **AI Summary & Tasks** — simulated AI generation
4. **Task Verification** — real API call to the FastAPI backend (`POST /verify-tasks`)
5. **HITL Verification** — simulated human-in-the-loop check
6. **CA Submission** — edit and submit final content

## Getting Started

### Frontend

```bash
npm install
npm run dev
```

### Backend

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --port 8000 --reload
```

Open http://localhost:5173 to use the app.
