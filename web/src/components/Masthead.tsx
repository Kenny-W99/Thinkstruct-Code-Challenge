import type { Health } from "../api/types";
import { formatCount, shortModel } from "../lib/text";

export function Masthead({ health, error }: { health: Health | null; error: string | null }) {
  return (
    <header className="masthead">
      <div className="wordmark">
        <span className="wordmark-mark" aria-hidden>
          ¶
        </span>
        <div>
          <h1>Patent Search</h1>
          <p>Keyword + meaning search, metadata filters, two-phase reranking</p>
        </div>
      </div>
      <div className="index-status" aria-live="polite">
        {error ? (
          <span className="status-down">API unreachable: {error}</span>
        ) : health ? (
          <>
            <span>
              <b>{formatCount(health.n_patents)}</b> patents
            </span>
            <span>
              <b>{formatCount(health.n_passages)}</b> passages
            </span>
            <span title="Embedding model (phase 1)">{shortModel(health.embedder)}</span>
            <span title="Cross-encoder (phase 2)">{shortModel(health.reranker)}</span>
            <a href="/docs" target="_blank" rel="noreferrer">
              API docs ↗
            </a>
          </>
        ) : (
          <span>connecting…</span>
        )}
      </div>
    </header>
  );
}
