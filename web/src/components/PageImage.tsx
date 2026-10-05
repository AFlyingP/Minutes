import type { Box } from "../types";

interface Props {
  src: string;
  boxes: Box[] | null;
  start?: number;
  end?: number;
}

export default function PageImage({ src, boxes, start, end }: Props) {
  const selected =
    start === undefined || end === undefined || start >= end
      ? []
      : (boxes ?? []).filter(([from, to]) => from < end && to > start);
  return (
    <div className="page-image" data-testid="page-image" style={{ position: "relative" }}>
      <img src={src} alt="Source page" style={{ display: "block", width: "100%" }} />
      {selected.map(([from, to, x0, y0, x1, y1]) => (
        <span
          key={`${from}-${to}`}
          data-testid="highlight"
          style={{
            position: "absolute",
            left: `${x0 * 100}%`,
            top: `${y0 * 100}%`,
            width: `${(x1 - x0) * 100}%`,
            height: `${(y1 - y0) * 100}%`,
            backgroundColor: "rgba(255, 215, 0, 0.35)",
            pointerEvents: "none",
          }}
        />
      ))}
    </div>
  );
}
