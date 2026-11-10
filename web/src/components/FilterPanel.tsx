import type { Facets, Kind, Strategy } from "../api/types";
import { KINDS, activeFilterCount, withoutFilters, type SearchState } from "../lib/query";
import { formatCount } from "../lib/text";
import { Segmented } from "./Segmented";
import { TagInput } from "./TagInput";

interface Props {
  state: SearchState;
  facets: Facets | null;
  onChange: (next: SearchState) => void;
}

const KIND_LABEL: Record<Kind, string> = { abstract: "Title + abstract", claim: "Claims", description: "Description" };
const TOP_SUBCLASSES = 12;

function toggle<T>(list: T[], item: T): T[] {
  return list.includes(item) ? list.filter((x) => x !== item) : [...list, item];
}

export function FilterPanel({ state, facets, onChange }: Props) {
  const set = (patch: Partial<SearchState>) => onChange({ ...state, ...patch });
  const count = activeFilterCount(state);
  const subclasses = facets?.subclasses.slice(0, TOP_SUBCLASSES) ?? [];
  const typedPrefixes = state.cpc.filter((c) => !subclasses.some((s) => s.code === c));

  return (
    <aside className="filters" aria-label="Filters">
      <div className="filters-head">
        <h2>
          Filters {count > 0 && <span className="count-badge">{count}</span>}
        </h2>
        {count > 0 && (
          <button type="button" className="link-button" onClick={() => onChange(withoutFilters(state))}>
            Clear all
          </button>
        )}
      </div>

      <div className="field">
        <span className="field-label">Classification (any of)</span>
        <div className="cpc-grid">
          {subclasses.map((s) => (
            <label key={s.code} className="cpc-option">
              <input type="checkbox" checked={state.cpc.includes(s.code)} onChange={() => set({ cpc: toggle(state.cpc, s.code) })} />
              <span className="cpc-code">{s.code}</span>
              <span className="cpc-count">{formatCount(s.count)}</span>
            </label>
          ))}
        </div>
      </div>
      <TagInput
        label="More classification prefixes"
        values={typedPrefixes}
        onChange={(typed) => set({ cpc: [...state.cpc.filter((c) => subclasses.some((s) => s.code === c)), ...typed] })}
        normalize={(v) => v.toUpperCase()}
        placeholder="B60C 11/03, B60B1…"
        hint="A prefix matches deeper codes: B60C11 matches B60C 11/03."
      />

      <TagInput
        label="Must contain (title or abstract)"
        values={state.keyword}
        onChange={(keyword) => set({ keyword })}
        placeholder="e.g. groove"
        hint="Every word must appear. Plurals match: groove = grooves."
      />

      <div className="field">
        <label className="field-label">
          Title
          <input type="text" value={state.title} placeholder="contains…" onChange={(e) => set({ title: e.target.value })} />
        </label>
        <label className="check">
          <input type="checkbox" checked={state.titleExact} onChange={(e) => set({ titleExact: e.target.checked })} />
          exact title
        </label>
      </div>

      <div className="field date-range">
        <span className="field-label">Published</span>
        <div className="date-row">
          <input
            type="date"
            aria-label="Published from"
            value={state.dateFrom}
            min={facets?.date_min ?? undefined}
            max={state.dateTo || undefined}
            onChange={(e) => set({ dateFrom: e.target.value })}
          />
          <span aria-hidden>–</span>
          <input
            type="date"
            aria-label="Published to"
            value={state.dateTo}
            min={state.dateFrom || undefined}
            max={facets?.date_max ?? undefined}
            onChange={(e) => set({ dateTo: e.target.value })}
          />
        </div>
        {facets?.date_min && (
          <p className="field-hint">
            Data covers {facets.date_min} to {facets.date_max}.
          </p>
        )}
      </div>

      <div className="field">
        <span className="field-label">Search in</span>
        {KINDS.map((k) => (
          <label key={k} className="check">
            <input type="checkbox" checked={state.kinds.includes(k)} onChange={() => set({ kinds: toggle(state.kinds, k) })} />
            {KIND_LABEL[k]}
          </label>
        ))}
        <p className="field-hint">None ticked = search everything.</p>
      </div>

      <Segmented<Strategy>
        label="How filters run"
        value={state.strategy}
        onChange={(strategy) => set({ strategy })}
        options={[
          { value: "pre", label: "Filter first", title: "Find matching patents with lookup tables, then score only those (fast, never loses results)" },
          { value: "post", label: "Filter after", title: "Score everything, keep the top 1000, then drop non-matching ones (naive; can return fewer results)" },
        ]}
      />
    </aside>
  );
}
