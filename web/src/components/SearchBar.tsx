import type { FormEvent, KeyboardEvent } from "react";

export const EXAMPLES = [
  "tire tread pattern that reduces road noise",
  "bicycle wheel spoke made of carbon fiber",
  "trailer hitch with a locking ball mount",
  "20240051338 claim 6",
];

interface Props {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onExample: (query: string) => void;
  busy: boolean;
}

export function SearchBar({ value, onChange, onSubmit, onExample, busy }: Props) {
  const submit = (e: FormEvent) => {
    e.preventDefault();
    onSubmit();
  };
  // Enter searches; Shift+Enter adds a new line (for pasting long claims).
  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSubmit();
    }
  };
  return (
    <form className="searchbar" role="search" onSubmit={submit}>
      <label htmlFor="q" className="visually-hidden">
        Search query
      </label>
      <textarea
        id="q"
        name="q"
        rows={2}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder="Describe an invention, paste a claim, or enter a patent number (e.g. 20240051338 claim 6)"
        spellCheck={false}
        autoFocus
      />
      <button type="submit" className="search-button" disabled={!value.trim()} aria-busy={busy}>
        {busy ? "Searching…" : "Search"}
      </button>
      <p className="examples">
        <span>Try</span>
        {EXAMPLES.map((ex) => (
          <button key={ex} type="button" className="example" onClick={() => onExample(ex)}>
            {ex}
          </button>
        ))}
      </p>
    </form>
  );
}
