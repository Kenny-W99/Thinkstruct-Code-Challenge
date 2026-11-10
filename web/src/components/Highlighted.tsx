import { segments } from "../lib/text";

interface Props {
  text: string;
  spans: [number, number][];
  from?: number;
  to?: number;
}

/** Text with the matched query words wrapped in <mark>. */
export function Highlighted({ text, spans, from = 0, to = text.length }: Props) {
  return (
    <>
      {segments(text, spans, from, to).map((s, i) => (s.hit ? <mark key={i}>{s.text}</mark> : s.text))}
    </>
  );
}
