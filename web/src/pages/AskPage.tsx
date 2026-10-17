import { useEffect, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";

import { ApiError, ask, fetchCities } from "../api";
import CitationList from "../components/CitationList";
import Filters from "../components/Filters";
import type { AskResponse, City, FilterValues } from "../types";

function errorCode(err: unknown): string {
  return err instanceof ApiError ? err.code : "network";
}

export default function AskPage() {
  const [question, setQuestion] = useState("");
  const [filters, setFilters] = useState<FilterValues>({});
  const [cities, setCities] = useState<City[]>([]);
  const [result, setResult] = useState<AskResponse>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>();

  useEffect(() => {
    fetchCities().then(setCities, (err) => setError(errorCode(err)));
  }, []);

  async function submit(event: FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text) return;
    setLoading(true);
    setError(undefined);
    setResult(undefined);
    try {
      setResult(
        await ask(text, {
          city: filters.city,
          body: filters.body,
          date_from: filters.dateFrom,
          date_to: filters.dateTo,
          kind: filters.kind,
        }),
      );
    } catch (err) {
      setError(errorCode(err));
    } finally {
      setLoading(false);
    }
  }

  const cityNames = Object.fromEntries(cities.map((city) => [city.id, city.name]));

  return (
    <>
      <form onSubmit={submit}>
        <textarea
          data-testid="ask-input"
          placeholder="Ask a question about council records"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
        />
        <Filters value={filters} onChange={setFilters} cities={cities} />
        <button data-testid="ask-submit" type="submit" disabled={loading}>
          Ask
        </button>
      </form>
      {loading && <p data-testid="loading">Loading...</p>}
      {error && <p data-testid="error">Request failed: {error}</p>}
      {result && (
        <section data-testid="answer">
          {result.declined ? (
            <>
              <p data-testid="declined">{result.decline_message}</p>
              <p data-testid="decline-reason">{result.decline_reason}</p>
            </>
          ) : (
            <>
              {result.sentences.map((sentence, index) => (
                <p key={`${index}-${sentence.text}`}>
                  <span data-testid="sentence">{sentence.text}</span>{" "}
                  {sentence.citations.map((number) => {
                    const citation = result.citations.find((item) => item.number === number);
                    return citation ? (
                      <Link
                        key={number}
                        data-testid={`cite-${number}`}
                        to={`/doc/${citation.document_id}?unit=${citation.unit_index}&start=${citation.start_offset}&end=${citation.end_offset}`}
                      >
                        [{number}]
                      </Link>
                    ) : null;
                  })}
                </p>
              ))}
              <CitationList citations={result.citations} cityNames={cityNames} />
              <p data-testid="answer-meta">
                model {result.model ?? "unknown"} · cost ${result.cost_usd.toFixed(4)} ·{" "}
                {result.latency_ms} ms
                {result.cached ? " · cached" : ""}
              </p>
            </>
          )}
        </section>
      )}
    </>
  );
}
