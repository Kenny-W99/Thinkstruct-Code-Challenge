import { useCallback, useEffect, useRef } from "react";
import type { Patent } from "../api/types";
import { useResource } from "../hooks/useResource";
import { displayTitle } from "../lib/text";

interface Props {
  doc: string | null;
  anchor?: string;
  load: (doc: string, signal: AbortSignal) => Promise<Patent>;
  onClose: () => void;
}

function Claims({ claims }: { claims: Patent["claims"] }) {
  return (
    <ol className="claims">
      {claims.map((c) => (
        <li key={c.number} id={`claim-${c.number}`} className={c.independent ? "independent" : "dependent"}>
          <span className="claim-number">{c.number}.</span>
          <span>
            {!c.independent && <span className="claim-dep">↳ claim {c.depends_on.join(", ")} · </span>}
            {c.text}
          </span>
        </li>
      ))}
    </ol>
  );
}

/** Side panel with one full patent. Opens scrolled to the passage that was clicked. */
export function PatentView({ doc, anchor, load, onClose }: Props) {
  const loader = useCallback((signal: AbortSignal) => load(doc!, signal), [doc, load]);
  const { data: patent, error, loading } = useResource(doc, loader);
  const panel = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!doc) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    panel.current?.focus();
    return () => window.removeEventListener("keydown", onKey);
  }, [doc, onClose]);

  useEffect(() => {
    if (!patent || !anchor) return;
    const target = panel.current?.querySelector<HTMLElement>(`#${CSS.escape(anchor)}`);
    if (!target) return;
    target.classList.add("target");
    target.scrollIntoView?.({ block: "center" });
  }, [patent, anchor]);

  if (!doc) return null;
  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <div
        className="drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="patent-title"
        tabIndex={-1}
        ref={panel}
        onClick={(e) => e.stopPropagation()}
      >
        <header className="drawer-head">
          <span className="doc">US {doc}</span>
          <button type="button" className="close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>
        {loading && <p className="drawer-status">Loading…</p>}
        {error && <p className="error-banner">{error}</p>}
        {patent && (
          <article className="patent">
            <h2 id="patent-title">{displayTitle(patent.title)}</h2>
            <dl className="patent-meta">
              <dt>Classification</dt>
              <dd>{patent.cpc ?? "none"}</dd>
              <dt>Published</dt>
              <dd>{patent.pub_date ?? patent.year ?? "unknown"}</dd>
              <dt>Source</dt>
              <dd>{patent.filename || "—"}</dd>
            </dl>
            <section id="abstract">
              <h3>Abstract</h3>
              <p>{patent.abstract ?? "No abstract."}</p>
            </section>
            <section>
              <h3>
                Claims <span className="muted">{patent.claims.length}</span>
              </h3>
              <Claims claims={patent.claims} />
            </section>
            <section>
              <h3>
                Description <span className="muted">{patent.description.length} ¶</span>
              </h3>
              {patent.description.length === 0 ? (
                <p className="muted">{patent.has_description ? "Not kept in this index." : "This record has no description."}</p>
              ) : (
                patent.description.map((para, i) => (
                  <p key={i} id={`para-${i + 1}`} className="para">
                    <span className="para-number">¶{i + 1}</span>
                    {para}
                  </p>
                ))
              )}
            </section>
          </article>
        )}
      </div>
    </div>
  );
}
