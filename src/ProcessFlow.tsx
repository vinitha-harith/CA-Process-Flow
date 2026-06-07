import { useState, useCallback } from "react";
import {
  MdVideoCall,
  MdTranscribe,
  MdAutoAwesome,
  MdVerifiedUser,
  MdCheckCircle,
  MdAccessTime,
  MdRadioButtonUnchecked,
  MdHourglassTop,
  MdEdit,
  MdFactCheck,
} from "react-icons/md";
import "./ProcessFlow.scss";

type StepStatus = "pending" | "active" | "completed";

interface Step {
  id: number;
  label: string;
  icon: React.ReactNode;
  status: StepStatus;
  timestamp: string | null;
}

interface VerificationResult {
  verified: boolean;
  checked: number;
  flagged: number;
  details: { task: string; status: string; confidence: number }[];
}

const now = () =>
  new Date().toLocaleString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

const STATUS_ICON: Record<StepStatus, React.ReactNode> = {
  completed: <MdCheckCircle className="status-icon completed" />,
  active: <MdHourglassTop className="status-icon active" />,
  pending: <MdRadioButtonUnchecked className="status-icon pending" />,
};

const INITIAL_STEPS: Step[] = [
  {
    id: 1,
    label: "Teams Call with Client",
    icon: <MdVideoCall />,
    status: "active",
    timestamp: null,
  },
  {
    id: 2,
    label: "Call Transcription",
    icon: <MdTranscribe />,
    status: "pending",
    timestamp: null,
  },
  {
    id: 3,
    label: "AI Summary & Tasks",
    icon: <MdAutoAwesome />,
    status: "pending",
    timestamp: null,
  },
  {
    id: 4,
    label: "Task Verification",
    icon: <MdFactCheck />,
    status: "pending",
    timestamp: null,
  },
  {
    id: 5,
    label: "HITL Verification",
    icon: <MdVerifiedUser />,
    status: "pending",
    timestamp: null,
  },
  {
    id: 6,
    label: "CA Submission",
    icon: <MdEdit />,
    status: "pending",
    timestamp: null,
  },
];

export default function ProcessFlow() {
  const [steps, setSteps] = useState<Step[]>(INITIAL_STEPS);
  const [teamsUrl, setTeamsUrl] = useState("");
  const [aiContent, setAiContent] = useState("");
  const [aiTasks, setAiTasks] = useState<string[]>([]);
  const [apiLoading, setApiLoading] = useState(false);
  const [taskVerifying, setTaskVerifying] = useState(false);
  const [taskVerifyResult, setTaskVerifyResult] =
    useState<VerificationResult | null>(null);
  const [taskVerifyError, setTaskVerifyError] = useState<string | null>(null);
  const [hitlVerifying, setHitlVerifying] = useState(false);

  const advance = useCallback((completedId: number) => {
    setSteps((prev) =>
      prev.map((s) => {
        if (s.id === completedId)
          return { ...s, status: "completed", timestamp: now() };
        if (s.id === completedId + 1) return { ...s, status: "active" };
        return s;
      })
    );
  }, []);

  /* Step 1 → 2 → 3: providing the Teams URL completes Step 1,
     then transcription (Step 2) runs automatically */
  const handleTeamsUrl = () => {
    if (!teamsUrl.trim()) return;
    advance(1);
    setTimeout(() => {
      advance(2);
      simulateAiGeneration();
    }, 1500);
  };

  /* Step 3: simulated AI summary */
  const simulateAiGeneration = () => {
    setApiLoading(true);
    setTimeout(() => {
      const tasks = [
        "Follow up with client on portfolio rebalancing.",
        "Schedule review meeting for Q3 targets.",
        "Send updated risk-assessment report.",
      ];
      setAiTasks(tasks);
      setAiContent(tasks.map((t) => `• ${t}`).join("\n"));
      setApiLoading(false);
      advance(3);
    }, 2500);
  };

  /* Step 4: real API call to FastAPI backend */
  const runTaskVerification = async () => {
    setTaskVerifying(true);
    setTaskVerifyError(null);
    setTaskVerifyResult(null);

    try {
      const res = await fetch("/api/verify-tasks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tasks: aiTasks,
          meeting_id: teamsUrl,
        }),
      });

      if (!res.ok) throw new Error(`Backend returned ${res.status}`);

      const data: VerificationResult = await res.json();
      setTaskVerifyResult(data);
      advance(4);
      simulateHitlVerification();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setTaskVerifyError(msg);
    } finally {
      setTaskVerifying(false);
    }
  };

  /* Step 5: simulated HITL */
  const simulateHitlVerification = () => {
    setHitlVerifying(true);
    setTimeout(() => {
      setHitlVerifying(false);
      advance(5);
    }, 2000);
  };

  /* Step 6: CA submit */
  const handleSubmit = () => {
    if (!aiContent.trim()) return;
    advance(6);
  };

  const handleReset = () => {
    setSteps(
      INITIAL_STEPS.map((s, i) =>
        i === 0
          ? { ...s, status: "active", timestamp: null }
          : { ...s, status: "pending", timestamp: null }
      )
    );
    setTeamsUrl("");
    setAiContent("");
    setAiTasks([]);
    setApiLoading(false);
    setTaskVerifying(false);
    setTaskVerifyResult(null);
    setTaskVerifyError(null);
    setHitlVerifying(false);
  };

  const allDone = steps.every((s) => s.status === "completed");

  return (
    <div className="pf-container">
      <header className="pf-header">
        <h1>Client Advisory — Process Flow</h1>
        <p className="pf-subtitle">
          End-to-end workflow from Teams call to final submission
        </p>
      </header>

      {/* ── Horizontal stepper ── */}
      <div className="pf-stepper">
        {steps.map((step, idx) => (
          <div key={step.id} className={`pf-step ${step.status}`}>
            <div className="pf-node">
              {idx > 0 && <div className={`pf-connector ${step.status}`} />}
              <div className="pf-circle">{step.icon}</div>
            </div>
            <span className="pf-label">{step.label}</span>
            <span className="pf-status-badge">
              {STATUS_ICON[step.status]}
              <span>{step.status}</span>
            </span>
            {step.timestamp && (
              <span className="pf-timestamp">
                <MdAccessTime size={12} /> {step.timestamp}
              </span>
            )}
          </div>
        ))}
      </div>

      {/* ── Action panel ── */}
      <div className="pf-panel">
        {/* Step 1 — Teams Call: provide URL to mark complete */}
        {steps[0].status === "active" && (
          <div className="pf-card">
            <h2>Step 1 — Teams Call with Client</h2>
            <p>
              Paste the Microsoft Teams meeting URL to confirm the call
              took place.
            </p>
            <div className="pf-input-row">
              <input
                type="text"
                placeholder="https://teams.microsoft.com/l/meetup-join/..."
                value={teamsUrl}
                onChange={(e) => setTeamsUrl(e.target.value)}
              />
              <button onClick={handleTeamsUrl} disabled={!teamsUrl.trim()}>
                Confirm Call
              </button>
            </div>
          </div>
        )}

        {/* Step 2 — Transcription (runs automatically) */}
        {steps[1].status === "active" && (
          <div className="pf-card">
            <h2>Step 2 — Call Transcription</h2>
            <div className="pf-loader">
              <div className="pf-spinner" />
              <span>Transcribing the Teams call…</span>
            </div>
          </div>
        )}

        {/* Step 3 — AI Generation */}
        {steps[2].status === "active" && (
          <div className="pf-card">
            <h2>Step 3 — AI Summary &amp; Follow-Up Tasks</h2>
            {apiLoading ? (
              <div className="pf-loader">
                <div className="pf-spinner" />
                <span>Generating AI summary from transcript…</span>
              </div>
            ) : (
              <p>Waiting for API response…</p>
            )}
          </div>
        )}

        {/* Step 4 — Task Verification (real API call) */}
        {steps[3].status === "active" && (
          <div className="pf-card">
            <h2>Step 4 — Task Verification</h2>
            <p>
              Verify the AI-generated tasks against the external validation
              service.
            </p>

            {!taskVerifying && !taskVerifyResult && !taskVerifyError && (
              <>
                <ul className="pf-task-list">
                  {aiTasks.map((t, i) => (
                    <li key={i}>{t}</li>
                  ))}
                </ul>
                <button className="pf-verify-btn" onClick={runTaskVerification}>
                  Verify Tasks via API
                </button>
              </>
            )}

            {taskVerifying && (
              <div className="pf-loader">
                <div className="pf-spinner" />
                <span>
                  Calling external verification API&nbsp;
                  <code>POST /verify-tasks</code>…
                </span>
              </div>
            )}

            {taskVerifyError && (
              <div className="pf-error">
                <strong>Error:</strong> {taskVerifyError}
                <button className="pf-retry" onClick={runTaskVerification}>
                  Retry
                </button>
              </div>
            )}
          </div>
        )}

        {/* Step 5 — HITL Verification */}
        {steps[4].status === "active" && (
          <div className="pf-card">
            <h2>Step 5 — HITL / System Verification</h2>
            {hitlVerifying ? (
              <div className="pf-loader">
                <div className="pf-spinner" />
                <span>Verifying AI-generated content…</span>
              </div>
            ) : (
              <p>Verification in progress…</p>
            )}
          </div>
        )}

        {/* Step 6 — CA Submission */}
        {steps[5].status === "active" && (
          <div className="pf-card">
            <h2>Step 6 — Review &amp; Submit</h2>
            <p>
              Review and edit the AI-generated content, then submit to complete
              the process.
            </p>
            <textarea
              rows={5}
              value={aiContent}
              onChange={(e) => setAiContent(e.target.value)}
            />
            <button
              className="pf-submit"
              onClick={handleSubmit}
              disabled={!aiContent.trim()}
            >
              Submit Final Content
            </button>
          </div>
        )}

        {allDone && (
          <div className="pf-card pf-done">
            <MdCheckCircle size={40} />
            <h2>Process Complete</h2>
            <p>All steps have been successfully completed.</p>
            <button className="pf-reset" onClick={handleReset}>
              Reset Flow
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
