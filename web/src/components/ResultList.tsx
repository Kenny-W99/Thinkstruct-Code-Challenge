import type { SearchResponse } from "../api/types";
import { formatCount, shortModel } from "../lib/text";
import { ResultCard } from "./ResultCard";
import { TimingBar } from "./TimingBar";

interface Props {
  response: SearchResponse;
  totalPatents: number | null;
  onOpen: (doc: string, anchor?: string) => void;
}

const MODE_LABEL = { hybrid: "keyword + meaning", bm25: "keyword only", dense: "meaning only" } as const;

export function Summary({ response: r, totalPatents }: Omit<Props, "onOpen">) {
  return (
    <section className="summary" aria-label="About these results">
      <div className="summary-lines">
        {r.note && <p className="summary-note">{r.note}.</p>}
        <p>
          <b>{r.results.length}</b> patent{r.results.length === 1 ? "" : "s"} · {MODE_LABEL[r.mode]}
          {r.candidate_patents !== null && (
            <>
              {" "}
              · <b>{formatCount(r.candidate_patents)}</b>
              {totalPatents !== null && <> of {formatCount(totalPatents)}</>} patents pass the filters (
              {r.strategy === "pre" ? "filtered first" : "filtered after scoring"})
            </>
          )}
          {r.excluded_no_cpc > 0 && <> · {formatCount(r.excluded_no_cpc)} have no classification</>}
        </p>
        <p className={`summary-rerank ${r.reranker ? "on" : "off"}`}>
          {r.reranker ? (
            <>
              Phase 2: top results reranked by <code>{shortModel(r.reranker)}</code>
            </>
          ) : (
            (r.rerank_note ?? "Phase 2 reranking is off.")
          )}
        </p>
      </div>
      <TimingBar timings={r.timings_ms} />
    </section>
  );
}

export function ResultList({ response, totalPatents, onOpen }: Props) {
  return (
    <>
      <Summary response={response} totalPatents={totalPatents} />
      {response.results.length === 0 ? (
        <div className="empty">
          <h3>No patents found</h3>
          <p>
            {response.strategy === "post"
              ? "The post-filter threw away every top result. Try the pre-filter, which searches only matching patents."
              : "Try fewer filters or different words."}
          </p>
        </div>
      ) : (
        <ol className="results-list">
          {response.results.map((hit) => (
            <li key={hit.doc_number}>
              <ResultCard hit={hit} onOpen={onOpen} />
            </li>
          ))}
        </ol>
      )}
    </>
  );
}
