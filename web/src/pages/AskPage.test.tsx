import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

import AskPage from "./AskPage";

const ANSWER = {
  declined: false,
  decline_reason: null,
  decline_message: null,
  sentences: [{ text: "The roll call vote was 4 ayes and 1 noes.", citations: [1] }],
  citations: [
    {
      number: 1,
      document_id: "birch-minutes-201",
      unit_index: 2,
      unit_kind: "page",
      start_offset: 0,
      end_offset: 214,
      quote: "Page 2 of 3 ...",
      city_id: "birch",
      body: "City Council",
      meeting_date: "2024-03-12",
      doc_kind: "minutes",
      start_ms: null,
      label: "p. 2",
    },
  ],
  pipeline: "final",
  model: "gpt-6-luna",
  cost_usd: 0.0012,
  latency_ms: 2140,
  cached: true,
  dropped_sentences: 0,
};

const CITIES = {
  cities: [
    {
      id: "birch",
      name: "Birch",
      state: "CA",
      bodies: ["City Council", "Planning Commission"],
      date_min: "2024-03-12",
      date_max: "2024-04-09",
      documents: 6,
    },
  ],
};

function renderPage() {
  render(
    <MemoryRouter>
      <AskPage />
    </MemoryRouter>,
  );
}

function mockFetch(answer: unknown = ANSWER, status = 200) {
  const fetchMock = vi.fn((url: string, init?: RequestInit) => {
    if (url === "/api/cities") return Promise.resolve(Response.json(CITIES));
    if (url === "/api/ask")
      return Promise.resolve(
        Response.json(status === 200 ? answer : { error: "llm_quota" }, { status }),
      );
    throw new Error(`unexpected request ${url} ${init?.method ?? "GET"}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => vi.unstubAllGlobals());

test("renders sentences with numbered citation links", async () => {
  mockFetch();
  renderPage();
  fireEvent.change(screen.getByTestId("ask-input"), {
    target: { value: "How did the council vote?" },
  });
  fireEvent.click(screen.getByTestId("ask-submit"));
  expect(await screen.findByTestId("sentence")).toHaveTextContent(
    "The roll call vote was 4 ayes and 1 noes.",
  );
  expect(screen.getByTestId("cite-1")).toHaveAttribute(
    "href",
    "/doc/birch-minutes-201?unit=2&start=0&end=214",
  );
});

test("renders the citation list with city, body, date, kind, and label", async () => {
  mockFetch();
  renderPage();
  fireEvent.change(screen.getByTestId("ask-input"), { target: { value: "How did they vote?" } });
  fireEvent.click(screen.getByTestId("ask-submit"));
  expect(await screen.findByTestId("citation-1")).toHaveTextContent(
    "[1] Birch City Council, 2024-03-12, minutes, p. 2",
  );
});

test("renders model, cost, and latency, and marks cached answers", async () => {
  mockFetch();
  renderPage();
  fireEvent.change(screen.getByTestId("ask-input"), { target: { value: "How did they vote?" } });
  fireEvent.click(screen.getByTestId("ask-submit"));
  expect(await screen.findByTestId("answer-meta")).toHaveTextContent(
    "model gpt-6-luna · cost $0.0012 · 2140 ms · cached",
  );
});

test("renders the decline message and reason", async () => {
  mockFetch({
    ...ANSWER,
    declined: true,
    decline_reason: "no_retrieval",
    decline_message: "I can't answer that from the indexed records.",
    sentences: [],
    citations: [],
  });
  renderPage();
  fireEvent.change(screen.getByTestId("ask-input"), {
    target: { value: "What about the airport runway?" },
  });
  fireEvent.click(screen.getByTestId("ask-submit"));
  expect(await screen.findByTestId("declined")).toHaveTextContent(
    "I can't answer that from the indexed records.",
  );
  expect(screen.getByTestId("decline-reason")).toHaveTextContent("no_retrieval");
});

test("sends filters with the question", async () => {
  const fetchMock = mockFetch();
  renderPage();
  await screen.findByRole("option", { name: "Birch" });
  fireEvent.change(screen.getByTestId("ask-input"), { target: { value: "How did they vote?" } });
  fireEvent.change(screen.getByTestId("filter-city"), { target: { value: "birch" } });
  fireEvent.click(screen.getByTestId("ask-submit"));
  await screen.findByTestId("answer");
  const request = fetchMock.mock.calls.find(([url]) => url === "/api/ask");
  expect(request).toBeDefined();
  const body = JSON.parse(String(request?.[1]?.body));
  expect(body).toEqual({ question: "How did they vote?", pipeline: "final", city: "birch" });
  expect(body).not.toHaveProperty("body");
  expect(body).not.toHaveProperty("date_from");
  expect(body).not.toHaveProperty("date_to");
  expect(body).not.toHaveProperty("kind");
});

test("shows the error code on failure", async () => {
  mockFetch(undefined, 503);
  renderPage();
  fireEvent.change(screen.getByTestId("ask-input"), { target: { value: "How did they vote?" } });
  fireEvent.click(screen.getByTestId("ask-submit"));
  expect(await screen.findByTestId("error")).toHaveTextContent("Request failed: llm_quota");
});

test("does not submit an empty question", async () => {
  const fetchMock = mockFetch();
  renderPage();
  fireEvent.click(screen.getByTestId("ask-submit"));
  expect(fetchMock).toHaveBeenCalledTimes(1);
  expect(fetchMock.mock.calls[0][0]).toBe("/api/cities");
});
