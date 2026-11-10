import { useState } from "react";
import type { Passage, PatentHit } from "../api/types";
import { anchorFor, displayTitle, snippetWindow } from "../lib/text";
import { Highlighted } from "./Highlighted";
import { RankChange } from "./RankChange";

const KIND_LABEL: Record<Passage["kind"], string> = { abstract: "Abstract", claim: "Claim", description: "Description" };

function Provenance({ p }: { p: Passage }) {
  const parts = [p.bm25_rank && `keyword #${p.bm25_rank}`, p.dense_rank && `meaning #${p.dense_rank}`].filter(Boolean);
  return (
    <span className="passage-provenance">
      {parts.join(" · ") || "—"}
      {p.rerank_score !== null && <span className="passage-rerank">rerank {p.rerank_score.toFixed(2)}</span>}
    </span>
  );
}

function PassageRow({ p, onOpen }: { p: Passage; onOpen: () => void }) {
  const [expanded, setExpanded] = useState(false);
  const w = snippetWindow(p.text, p.highlights);
  const clipped = w.clippedStart || w.clippedEnd;
  return (
    <li className={`passage kind-${p.kind}`}>
      <div className="passage-head">
        <button type="button" className="passage-label" onClick={onOpen} title="Show in the full patent">
          <span className="passage-kind">{KIND_LABEL[p.kind]}</span>
          {p.label.replace(/^Claim |^Description |^Title \+ Abstract/, "")}
        </button>
        <Provenance p={p} />
      </div>
      <p className="passage-text">
        {expanded || !clipped ? (
          <Highlighted text={p.text} spans={p.highlights} />
        ) : (
          <>
            {w.clippedStart && "… "}
            <Highlighted text={p.text} spans={p.highlights} from={w.from} to={w.to} />
            {w.clippedEnd && " …"}
          </>
        )}
        {clipped && (
          <button type="button" className="link-button" onClick={() => setExpanded(!expanded)}>
            {expanded ? "less" : "more"}
          </button>
        )}
      </p>
    </li>
  );
}

interface Props {
  hit: PatentHit;
  onOpen: (doc: string, anchor?: string) => void;
}

export function ResultCard({ hit, onOpen }: Props) {
  const date = hit.pub_date ?? (hit.year ? String(hit.year) : null);
  return (
    <article className="result" style={{ ["--i" as string]: hit.rank }}>
      <div className="result-rank">
        <span className="result-rank-number">{String(hit.rank).padStart(2, "0")}</span>
        <RankChange rank={hit.rank} firstStageRank={hit.first_stage_rank} />
      </div>
      <div className="result-body">
        <header className="result-head">
          <h3>
            <button type="button" className="result-title" onClick={() => onOpen(hit.doc_number)}>
              {displayTitle(hit.title)}
            </button>
          </h3>
          <p className="result-meta">
            <span className="doc">US {hit.doc_number}</span>
            <span>{hit.cpc ?? "no classification"}</span>
            {date && <span>{date}</span>}
            {!hit.has_description && <span className="muted">no description</span>}
          </p>
        </header>
        <ul className="passages">
          {hit.passages.map((p) => (
            <PassageRow key={p.row} p={p} onOpen={() => onOpen(hit.doc_number, anchorFor(p.label))} />
          ))}
        </ul>
      </div>
    </article>
  );
}
