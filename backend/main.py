import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Task Verification Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TaskVerificationRequest(BaseModel):
    tasks: list[str]
    meeting_id: str | None = None


class TaskVerificationResponse(BaseModel):
    verified: bool
    checked: int
    flagged: int
    details: list[dict]


@app.post("/verify-tasks", response_model=TaskVerificationResponse)
async def verify_tasks(req: TaskVerificationRequest):
    """Simulate an external task-verification service.

    In production this would call a downstream compliance / validation
    system.  Here we add a short delay and return a deterministic result.
    """
    await asyncio.sleep(2)

    details = [
        {"task": t, "status": "ok", "confidence": 0.95}
        for t in req.tasks
    ]

    return TaskVerificationResponse(
        verified=True,
        checked=len(req.tasks),
        flagged=0,
        details=details,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
