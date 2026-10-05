import { useEffect, useState, type FormEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { ApiError, fetchCities, search } from "../api";
import type { City, Hit, SpanRef } from "../types";

const FILTER_KEYS = ["city", "body", "date_from", "date_to", "kind"] as const;
const FORM_KEYS = ["q", ...FILTER_KEYS] as const;

type Form = Record<(typeof FORM_KEYS)[number], string>;

// the query string a result belongs to, so a slow response for an older search is not shown
interface Outcome {
  key: string;
  hits?: Hit[];
  error?: string;
}

function errorCode(err: unknown): string {
  return err instanceof ApiError ? err.code : "network";
}

function spanPath(span: SpanRef): string {
  return `/doc/${span.document_id}?unit=${span.unit_index}&start=${span.start_offset}&end=${span.end_offset}`;
}

export default function SearchPage() {
  const [params, setParams] = useSearchParams();
  const [form, setForm] = useState<Form>(
    () => Object.fromEntries(FORM_KEYS.map((name) => [name, params.get(name) ?? ""])) as Form,
  );
  const [cities, setCities] = useState<City[]>([]);
  const [citiesError, setCitiesError] = useState<string>();
  const [outcome, setOutcome] = useState<Outcome>();

  const key = params.toString();
  const searched = Boolean(params.get("q"));

  useEffect(() => {
    fetchCities().then(setCities, (err) => setCitiesError(errorCode(err)));
  }, []);

  useEffect(() => {
    const sent = new URLSearchParams(key);
    const q = sent.get("q");
    if (!q) return;
    let stale = false;
    const filters = Object.fromEntries(FILTER_KEYS.map((name) => [name, sent.get(name) ?? ""]));
    search(q, filters).then(
      (response) => {
        if (!stale) setOutcome({ key, hits: response.hits });
      },
      (err) => {
        if (!stale) setOutcome({ key, error: errorCode(err) });
      },
    );
    return () => {
      stale = true;
    };
  }, [key]);

  const current = outcome?.key === key ? outcome : undefined;
  const error = current?.error ?? citiesError;
  const selected = cities.filter((city) => !form.city || city.id === form.city);
  const bodies = [...new Set(selected.flatMap((city) => city.bodies))].sort();

  function submit(event: FormEvent) {
    event.preventDefault();
    const q = form.q.trim();
    if (!q) return;
    const next = new URLSearchParams({ q });
    for (const name of FILTER_KEYS) {
      if (form[name]) next.set(name, form[name]);
    }
    setParams(next);
  }

  function field(name: keyof Form) {
    return {
      value: form[name],
      onChange: (event: { target: { value: string } }) =>
        setForm({ ...form, [name]: event.target.value }),
    };
  }

  return (
    <>
      <form onSubmit={submit}>
        <div className="search-row">
          <input
            type="text"
            data-testid="search-input"
            placeholder="Search agendas, minutes, and transcripts"
            {...field("q")}
          />
          <button type="submit" data-testid="search-submit">
            Search
          </button>
        </div>
        <div className="filters" data-testid="filters">
          <select
            data-testid="filter-city"
            aria-label="City"
            value={form.city}
            onChange={(event) => setForm({ ...form, city: event.target.value, body: "" })}
          >
            <option value="">All cities</option>
            {cities.map((city) => (
              <option key={city.id} value={city.id}>
                {city.name}
              </option>
            ))}
          </select>
          <select data-testid="filter-body" aria-label="Body" {...field("body")}>
            <option value="">All bodies</option>
            {bodies.map((body) => (
              <option key={body}>{body}</option>
            ))}
          </select>
          <input
            type="date"
            data-testid="filter-date-from"
            aria-label="From date"
            {...field("date_from")}
          />
          <input
            type="date"
            data-testid="filter-date-to"
            aria-label="To date"
            {...field("date_to")}
          />
          <select data-testid="filter-kind" aria-label="Kind" {...field("kind")}>
            <option value="">All kinds</option>
            <option value="agenda">Agenda</option>
            <option value="minutes">Minutes</option>
            <option value="transcript">Transcript</option>
          </select>
        </div>
      </form>

      {!searched && <p>Enter a search to begin.</p>}
      {searched && !current && <p data-testid="loading">Loading...</p>}
      {error && <p data-testid="error">Request failed: {error}</p>}
      {current?.hits?.length === 0 && <p data-testid="empty">No results.</p>}
      {current?.hits && current.hits.length > 0 && (
        <ol className="results" data-testid="results">
          {current.hits.map((hit) => (
            <li key={hit.rank} data-testid={`result-${hit.rank}`}>
              <h2 data-testid="result-heading">{hit.heading}</h2>
              <p>{hit.snippet}</p>
              {hit.spans.map((span) => (
                <Link key={spanPath(span)} to={spanPath(span)} data-testid="result-span">
                  {span.label}
                </Link>
              ))}
            </li>
          ))}
        </ol>
      )}
    </>
  );
}
