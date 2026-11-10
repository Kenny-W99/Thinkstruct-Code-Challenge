import { useState, type KeyboardEvent } from "react";

interface Props {
  label: string;
  values: string[];
  onChange: (values: string[]) => void;
  placeholder?: string;
  hint?: string;
  /** Normalise a typed value before adding it (e.g. upper-case a CPC code). */
  normalize?: (value: string) => string;
}

/** A list of removable chips plus a text box; Enter or comma adds a chip. */
export function TagInput({ label, values, onChange, placeholder, hint, normalize = (v) => v }: Props) {
  const [draft, setDraft] = useState("");

  const add = () => {
    const value = normalize(draft.trim());
    if (value && !values.includes(value)) onChange([...values, value]);
    setDraft("");
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      add();
    } else if (e.key === "Backspace" && !draft && values.length) {
      onChange(values.slice(0, -1));
    }
  };

  return (
    <div className="field">
      <label className="field-label">
        {label}
        <div className="tag-input">
          {values.map((v) => (
            <span key={v} className="chip">
              {v}
              <button type="button" aria-label={`Remove ${v}`} onClick={() => onChange(values.filter((x) => x !== v))}>
                ×
              </button>
            </span>
          ))}
          <input
            value={draft}
            placeholder={values.length ? "" : placeholder}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={onKeyDown}
            onBlur={add}
          />
        </div>
      </label>
      {hint && <p className="field-hint">{hint}</p>}
    </div>
  );
}
