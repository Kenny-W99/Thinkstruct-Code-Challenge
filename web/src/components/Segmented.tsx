import { useId } from "react";

export interface Option<T extends string> {
  value: T;
  label: string;
  title?: string;
  disabled?: boolean;
}

interface Props<T extends string> {
  label: string;
  value: T;
  options: Option<T>[];
  onChange: (value: T) => void;
}

/** A row of mutually exclusive options (a styled radio group). */
export function Segmented<T extends string>({ label, value, options, onChange }: Props<T>) {
  const name = useId();
  return (
    <fieldset className="segmented">
      <legend>{label}</legend>
      <div className="segmented-track">
        {options.map((o) => (
          <label key={o.value} className="segmented-option" title={o.title} data-disabled={o.disabled || undefined}>
            <input
              type="radio"
              name={name}
              value={o.value}
              checked={value === o.value}
              disabled={o.disabled}
              onChange={() => onChange(o.value)}
            />
            <span>{o.label}</span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}
