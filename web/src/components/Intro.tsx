/** Shown before the first search: what happens to a query. */
export function Intro() {
  return (
    <section className="intro" aria-label="How search works">
      <ol className="pipeline">
        <li>
          <span className="step">1</span>
          <h3>Filter</h3>
          <p>Lookup tables find the patents that match the classification, words, title and dates, before any scoring.</p>
        </li>
        <li>
          <span className="step">2</span>
          <h3>Retrieve</h3>
          <p>
            Every passage (abstract, each claim, each description paragraph) is scored by keywords (BM25) and by meaning
            (embeddings). The two rankings are merged.
          </p>
        </li>
        <li>
          <span className="step">3</span>
          <h3>Rerank</h3>
          <p>
            For short queries, a cross-encoder reads the query and each of the top 30 patents together and reorders them.
            ▲ marks show what it moved.
          </p>
        </li>
      </ol>
    </section>
  );
}
