import { useEffect, useState, type FormEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { ApiError, fetchCities, search } from "../api";
import Filters from "../components/Filters";
import type { City, FilterValues, Hit, SpanRef } from "../types";

const FILTER_KEYS = ["city", "body", "date_from", "date_to", "kind"] as const;

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

function filtersFromParams(params: URLSearchParams): FilterValues {
  return {
    city: params.get("city") || undefined,
    body: params.get("body") || undefined,
    dateFrom: params.get("date_from") || undefined,
    dateTo: params.get("date_to") || undefined,
    kind: params.get("kind") || undefined,
  };
}

export default function SearchPage() {
  const [params, setParams] = useSearchParams();
  const key = params.toString();
  const [draft, setDraft] = useState(() => ({
    key,
    query: params.get("q") ?? "",
    filters: filtersFromParams(params),
  }));
  const query = draft.key === key ? draft.query : (params.get("q") ?? "");
  const filters = draft.key === key ? draft.filters : filtersFromParams(params);
  const [cities, setCities] = useState<City[]>([]);
  const [citiesError, setCitiesError] = useState<string>();
  const [outcome, setOutcome] = useState<Outcome>();

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

  function submit(event: FormEvent) {
    event.preventDefault();
    const q = query.trim();
    if (!q) return;
    const next = new URLSearchParams({ q });
    const values: Record<(typeof FILTER_KEYS)[number], string | undefined> = {
      city: filters.city,
      body: filters.body,
      date_from: filters.dateFrom,
      date_to: filters.dateTo,
      kind: filters.kind,
    };
    for (const name of FILTER_KEYS) {
      const value = values[name];
      if (value) next.set(name, value);
    }
    setParams(next);
  }

  return (
    <>
      <form onSubmit={submit}>
        <div className="search-row">
          <input
            type="text"
            data-testid="search-input"
            placeholder="Search agendas, minutes, and transcripts"
            value={query}
            onChange={(event) => setDraft({ key, query: event.target.value, filters })}
          />
          <button type="submit" data-testid="search-submit">
            Search
          </button>
        </div>
        <Filters
          value={filters}
          onChange={(next) => setDraft({ key, query, filters: next })}
          cities={cities}
        />
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
