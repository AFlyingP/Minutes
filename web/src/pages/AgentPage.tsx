import { useState, type FormEvent } from "react";
import { Link } from "react-router-dom";

import { ApiError, runAgent } from "../api";
import type { AgentResponse } from "../types";

function errorCode(err: unknown): string {
  return err instanceof ApiError ? err.code : "network";
}

export default function AgentPage() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<AgentResponse>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>();

  async function submit(event: FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text) return;
    setLoading(true);
    setError(undefined);
    setResult(undefined);
    try {
      setResult(await runAgent(text));
    } catch (err) {
      setError(errorCode(err));
    } finally {
      setLoading(false);
    }
  }

  const status =
    result &&
    `steps ${result.steps.length} of ${result.max_steps}` +
      (result.status === "limit_exceeded"
        ? ` · stopped: ${result.limit} limit`
        : result.status === "error"
          ? " · error"
          : "");

  return (
    <>
      <form onSubmit={submit}>
        <textarea
          data-testid="agent-input"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
        />
        <button data-testid="agent-submit" type="submit" disabled={loading}>
          Run
        </button>
      </form>
      {loading && <p data-testid="loading">Loading...</p>}
      {error && <p data-testid="error">Request failed: {error}</p>}
      {result && (
        <>
          <p data-testid="agent-status">{status}</p>
          <ol data-testid="steps">
            {result.steps.map((step) => (
              <li key={step.index} data-testid={`step-${step.index}`}>
                {step.index}. {step.tool} - {step.result_summary}
                {step.error && ` (${step.error})`}
                <details>
                  <summary>Arguments</summary>
                  <pre>{JSON.stringify(step.arguments, null, 2)}</pre>
                </details>
              </li>
            ))}
          </ol>
          <section data-testid="agent-answer">
            <p>{result.answer}</p>
            {result.citations.map((citation, index) => (
              <Link
                key={`${citation.document_id}-${citation.unit_index}-${index}`}
                to={`/doc/${citation.document_id}?unit=${citation.unit_index}`}
              >
                [{citation.document_id} {citation.label}]
              </Link>
            ))}
          </section>
        </>
      )}
    </>
  );
}
