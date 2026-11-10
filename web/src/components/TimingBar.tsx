import { formatMs } from "../lib/text";

// Pipeline order; anything else the API reports is shown after these.
const STAGES: { key: string; label: string; phase: 1 | 2 }[] = [
  { key: "embed", label: "embed query", phase: 1 },
  { key: "filter", label: "filter", phase: 1 },
  { key: "bm25", label: "BM25", phase: 1 },
  { key: "dense", label: "vectors", phase: 1 },
  { key: "fuse", label: "fuse", phase: 1 },
  { key: "post_filter", label: "post-filter", phase: 1 },
  { key: "group", label: "group", phase: 1 },
  { key: "rerank", label: "rerank", phase: 2 },
];

export interface Stage {
  key: string;
  label: string;
  ms: number;
  phase: 1 | 2;
}

/** The measured stages in pipeline order, plus the phase 1 / phase 2 totals. */
export function stagesOf(timings: Record<string, number>): { stages: Stage[]; phase1: number; phase2: number; total: number } {
  const known = STAGES.filter((s) => s.key in timings).map((s) => ({ ...s, ms: timings[s.key] }));
  const extra = Object.keys(timings)
    .filter((k) => k !== "total" && !STAGES.some((s) => s.key === k))
    .map((k) => ({ key: k, label: k, ms: timings[k], phase: 1 as const }));
  const stages = [...known, ...extra];
  const phase2 = stages.filter((s) => s.phase === 2).reduce((a, s) => a + s.ms, 0);
  const total = timings.total ?? stages.reduce((a, s) => a + s.ms, 0);
  return { stages, phase1: total - phase2, phase2, total };
}

/** A stacked bar of where the time went. */
export function TimingBar({ timings }: { timings: Record<string, number> }) {
  const { stages, phase1, phase2, total } = stagesOf(timings);
  const sum = stages.reduce((a, s) => a + s.ms, 0) || 1;
  return (
    <figure className="timing" aria-label="Time spent per stage">
      <div className="timing-head">
        <span className="timing-total">{formatMs(total)}</span>
        <span className="timing-phases">
          phase 1 <b>{formatMs(phase1)}</b>
          {phase2 > 0 && (
            <>
              {" "}
              · phase 2 <b>{formatMs(phase2)}</b>
            </>
          )}
        </span>
      </div>
      <div className="timing-bar">
        {stages.map((s) => (
          <span
            key={s.key}
            className={`timing-seg seg-${s.key}`}
            style={{ flexGrow: Math.max(s.ms / sum, 0.008) }}
            title={`${s.label}: ${formatMs(s.ms)}`}
          />
        ))}
      </div>
      <figcaption className="timing-legend">
        {stages.map((s) => (
          <span key={s.key}>
            <i className={`seg-${s.key}`} />
            {s.label} {formatMs(s.ms)}
          </span>
        ))}
      </figcaption>
    </figure>
  );
}
