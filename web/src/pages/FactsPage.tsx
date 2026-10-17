import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { ApiError, fetchCities, fetchFacts } from "../api";
import Filters from "../components/Filters";
import type { City, FactCollection, FactsResponse, FilterValues } from "../types";

const KINDS: { value: FactCollection; label: string }[] = [
  { value: "votes", label: "Votes" },
  { value: "motions", label: "Motions" },
  { value: "ordinances", label: "Ordinances" },
  { value: "amounts", label: "Dollar amounts" },
  { value: "statements", label: "Statements" },
];

const COLUMNS: Record<FactCollection, { label: string; field?: string }[]> = {
  votes: [
    { label: "Date", field: "meeting_date" },
    { label: "City", field: "city_id" },
    { label: "Body", field: "body" },
    { label: "Subject", field: "subject_identifier" },
    { label: "Ayes", field: "ayes" },
    { label: "Noes", field: "noes" },
    { label: "Abstain", field: "abstain" },
    { label: "Absent", field: "absent" },
    { label: "Outcome", field: "outcome" },
    { label: "Source" },
  ],
  motions: [
    { label: "Date", field: "meeting_date" },
    { label: "City", field: "city_id" },
    { label: "Body", field: "body" },
    { label: "Motion", field: "text" },
    { label: "Mover", field: "mover" },
    { label: "Seconder", field: "seconder" },
    { label: "Outcome", field: "outcome" },
    { label: "Source" },
  ],
  ordinances: [
    { label: "Date", field: "meeting_date" },
    { label: "City", field: "city_id" },
    { label: "Body", field: "body" },
    { label: "Identifier", field: "identifier" },
    { label: "Type", field: "legislation_type" },
    { label: "Title", field: "title" },
    { label: "Status", field: "status" },
    { label: "Source" },
  ],
  amounts: [
    { label: "Date", field: "meeting_date" },
    { label: "City", field: "city_id" },
    { label: "Body", field: "body" },
    { label: "Amount", field: "amount_usd" },
    { label: "Purpose", field: "purpose" },
    { label: "Payee", field: "payee" },
    { label: "Source" },
  ],
  statements: [
    { label: "Date", field: "meeting_date" },
    { label: "City", field: "city_id" },
    { label: "Body", field: "body" },
    { label: "Speaker", field: "speaker" },
    { label: "Role", field: "role" },
    { label: "Summary", field: "summary" },
    { label: "Source" },
  ],
};

function errorCode(err: unknown): string {
  return err instanceof ApiError ? err.code : "network";
}

function cellValue(row: FactsResponse["rows"][number], field: string): string | number {
  const value =
    field === "meeting_date" || field === "body" || field === "city_id"
      ? row[field]
      : row.data[field];
  if (field === "city_id") return row.city_id;
  if (field === "amount_usd" && typeof value === "number") {
    return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(value);
  }
  return value == null ? "—" : typeof value === "object" ? JSON.stringify(value) : value;
}

export default function FactsPage() {
  const [kind, setKind] = useState<FactCollection>("votes");
  const [filters, setFilters] = useState<FilterValues>({});
  const [query, setQuery] = useState("");
  const [cities, setCities] = useState<City[]>([]);
  const [resultState, setResultState] = useState<{ key: string; value: FactsResponse }>();
  const [errorState, setErrorState] = useState<{ key: string; value: string }>();
  const [citiesError, setCitiesError] = useState<string>();
  const [offset, setOffset] = useState(0);
  const requestKey = JSON.stringify([kind, filters, query, offset]);

  useEffect(() => {
    fetchCities().then(setCities, (err) => setCitiesError(errorCode(err)));
  }, []);

  useEffect(() => {
    let stale = false;
    fetchFacts(kind, {
      city: filters.city,
      body: filters.body,
      date_from: filters.dateFrom,
      date_to: filters.dateTo,
      q: query.trim() || undefined,
      limit: "50",
      offset: String(offset),
    }).then(
      (response) => {
        if (!stale) setResultState({ key: requestKey, value: response });
      },
      (err) => {
        if (!stale) setErrorState({ key: requestKey, value: errorCode(err) });
      },
    );
    return () => {
      stale = true;
    };
  }, [kind, filters, query, offset, requestKey]);

  const result = resultState?.key === requestKey ? resultState.value : undefined;
  const error = errorState?.key === requestKey ? errorState.value : citiesError;
  const loading = !result && !error;

  const cityNames = Object.fromEntries(cities.map((city) => [city.id, city.name]));
  const columns = COLUMNS[kind];

  return (
    <>
      <div className="facts-controls">
        <select
          data-testid="facts-kind"
          aria-label="Fact kind"
          value={kind}
          onChange={(event) => {
            setKind(event.target.value as FactCollection);
            setOffset(0);
          }}
        >
          {KINDS.map((item) => (
            <option key={item.value} value={item.value}>
              {item.label}
            </option>
          ))}
        </select>
        <Filters
          value={filters}
          onChange={(next) => {
            setFilters(next);
            setOffset(0);
          }}
          cities={cities}
          showKind={false}
        />
      </div>
      <input
        data-testid="facts-q"
        placeholder="Filter by text in the quote"
        value={query}
        onChange={(event) => {
          setQuery(event.target.value);
          setOffset(0);
        }}
      />
      {loading && <p data-testid="loading">Loading...</p>}
      {error && <p data-testid="error">Request failed: {error}</p>}
      {result && (
        <>
          {result.rows.length === 0 ? (
            <p data-testid="empty">No rows.</p>
          ) : (
            <div className="facts-table-wrap">
              <table data-testid="facts-table">
                <thead>
                  <tr>
                    {columns.map((column) => (
                      <th key={column.label}>{column.label}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.rows.map((row) => (
                    <tr key={row.id}>
                      {columns.map((column) => (
                        <td key={column.label}>
                          {column.field ? (
                            column.field === "city_id" ? (
                              (cityNames[row.city_id] ?? row.city_id)
                            ) : (
                              cellValue(row, column.field)
                            )
                          ) : (
                            <Link
                              data-testid="fact-source"
                              to={`/doc/${row.document_id}?unit=${row.unit_index}${row.quote_start != null && row.quote_end != null ? `&start=${row.quote_start}&end=${row.quote_end}` : ""}`}
                            >
                              {row.label}
                            </Link>
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <div className="facts-pager">
            <button
              data-testid="facts-prev"
              disabled={offset === 0}
              onClick={() => setOffset(Math.max(0, offset - 50))}
            >
              Previous
            </button>
            <span data-testid="facts-total">{result.total} rows</span>
            <button
              data-testid="facts-next"
              disabled={offset + 50 >= result.total}
              onClick={() => setOffset(offset + 50)}
            >
              Next
            </button>
          </div>
        </>
      )}
    </>
  );
}
