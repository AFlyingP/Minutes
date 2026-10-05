import { useEffect, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";

import { ApiError, fetchCities, fetchDocument, fetchUnit, unitImageUrl } from "../api";
import PageImage from "../components/PageImage";
import type { City, DocumentDetail, Unit } from "../types";

export function timeLabel(milliseconds: number): string {
  const seconds = Math.floor(milliseconds / 1000);
  return [Math.floor(seconds / 3600), Math.floor(seconds / 60) % 60, seconds % 60]
    .map((part) => String(part).padStart(2, "0"))
    .join(":");
}

export function UnitViewer({
  doc,
  unit,
  start,
  end,
}: {
  doc: DocumentDetail;
  unit: Unit;
  start?: number;
  end?: number;
}) {
  return (
    <div className={unit.unit_kind === "page" ? "split unit-viewer" : "unit-viewer"}>
      {unit.unit_kind === "page" ? (
        <PageImage
          src={unitImageUrl(unit.document_id, unit.unit_index)}
          boxes={unit.boxes}
          start={start}
          end={end}
        />
      ) : (
        <div>
          <p data-testid="segment-time">
            {unit.label} to {timeLabel(unit.end_ms)}
          </p>
          {unit.recording_link === null ? (
            <p>No public recording link.</p>
          ) : (
            <a data-testid="recording-link" href={unit.recording_link}>
              {unit.recording_link !== doc.meeting.recording_url
                ? `Open recording at ${unit.label}`
                : `Open recording (seek to ${unit.label})`}
            </a>
          )}
        </div>
      )}
      <pre data-testid="unit-text">
        {start === undefined || end === undefined ? (
          unit.text
        ) : (
          <>
            {unit.text.slice(0, start)}
            <mark data-testid="mark">{unit.text.slice(start, end)}</mark>
            {unit.text.slice(end)}
          </>
        )}
      </pre>
    </div>
  );
}

export function UnitControls({
  unit,
  count,
  onChange,
}: {
  unit: Unit;
  count: number | null;
  onChange: (index: number) => void;
}) {
  return (
    <div className="unit-controls">
      <button
        data-testid="unit-prev"
        disabled={unit.unit_index === 1}
        onClick={() => onChange(unit.unit_index - 1)}
      >
        Previous
      </button>
      <span data-testid="unit-label">
        {unit.unit_kind === "page"
          ? `${unit.label} of ${count}`
          : `${unit.label} (segment ${unit.unit_index} of ${count})`}
      </span>
      <button
        data-testid="unit-next"
        disabled={count === null || unit.unit_index >= count}
        onClick={() => onChange(unit.unit_index + 1)}
      >
        Next
      </button>
    </div>
  );
}

export default function DocPage() {
  const id = useParams().id!;
  const [params, setParams] = useSearchParams();
  const index = Number(params.get("unit") ?? "1");
  const start = params.has("start") ? Number(params.get("start")) : undefined;
  const end = params.has("end") ? Number(params.get("end")) : undefined;
  const key = `${id}/${index}`;
  const [outcome, setOutcome] = useState<{
    key: string;
    result?: { doc: DocumentDetail; unit: Unit; cities: City[] };
    error?: string;
  }>();
  const current = outcome?.key === key ? outcome : undefined;
  const result = current?.result;
  const error = current?.error;
  const loading = !current;

  useEffect(() => {
    let stale = false;
    Promise.all([fetchDocument(id), fetchUnit(id, index), fetchCities()]).then(
      ([doc, unit, cities]) => {
        if (!stale) {
          setOutcome({ key, result: { doc, unit, cities } });
        }
      },
      (err: unknown) => {
        if (!stale) {
          setOutcome({ key, error: err instanceof ApiError ? err.code : "network" });
        }
      },
    );
    return () => {
      stale = true;
    };
  }, [id, index, key]);

  function changeUnit(next: number) {
    setParams({ unit: String(next) });
  }

  return (
    <>
      {loading && <p data-testid="loading">Loading...</p>}
      {error && <p data-testid="error">Request failed: {error}</p>}
      {!loading && !error && result && (
        <>
          <h1 data-testid="doc-title">
            {result.cities.find((city) => city.id === result.doc.city_id)?.name}{" "}
            {result.doc.meeting.body}, {result.doc.meeting.meeting_date}, {result.doc.kind}
          </h1>
          <UnitControls unit={result.unit} count={result.doc.unit_count} onChange={changeUnit} />
          <UnitViewer doc={result.doc} unit={result.unit} start={start} end={end} />
        </>
      )}
    </>
  );
}
