import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

import SearchPage from "./SearchPage";

const CITIES = {
  cities: [
    {
      id: "alder",
      name: "Alder",
      state: "WA",
      bodies: ["City Council"],
      date_min: "2024-02-06",
      date_max: "2024-03-05",
      documents: 6,
    },
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

const HIT = {
  rank: 1,
  score: 1.0,
  document_id: "birch-minutes-201",
  meeting_id: "birch-201",
  city_id: "birch",
  city_name: "Birch",
  body: "City Council",
  meeting_date: "2024-03-12",
  doc_kind: "minutes",
  item_identifier: "9.B",
  item_title: "Adopt RESOLUTION 2024-07 Awarding a Construction Contract",
  heading: "Birch City Council, 2024-03-12, minutes, item 9.B",
  snippet: "9.B Adopt RESOLUTION 2024-07 Awarding a Construction Contract to Granite Works Inc.",
  spans: [
    {
      document_id: "birch-minutes-201",
      unit_index: 1,
      unit_kind: "page",
      start_offset: 301,
      end_offset: 468,
      start_ms: null,
      label: "p. 1",
    },
  ],
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status });
}

function mockFetch(searchResponse: () => Promise<Response>) {
  const fetchMock = vi.fn((url: string) =>
    url.startsWith("/api/cities") ? Promise.resolve(json(CITIES)) : searchResponse(),
  );
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function renderPage(url = "/") {
  render(
    <MemoryRouter initialEntries={[url]}>
      <SearchPage />
    </MemoryRouter>,
  );
}

async function submitSearch() {
  await screen.findByRole("option", { name: "Birch" });
  fireEvent.change(screen.getByTestId("search-input"), { target: { value: "sidewalk repair" } });
  fireEvent.change(screen.getByTestId("filter-city"), { target: { value: "birch" } });
  fireEvent.click(screen.getByTestId("search-submit"));
}

afterEach(() => {
  vi.unstubAllGlobals();
});

test("renders the empty prompt before any search", async () => {
  const fetchMock = mockFetch(() => Promise.resolve(json({ hits: [] })));
  renderPage();
  expect(screen.getByText("Enter a search to begin.")).toBeInTheDocument();
  await screen.findByRole("option", { name: "Alder" });
  expect(fetchMock).toHaveBeenCalledTimes(1);
});

test("submits the query with filters and renders hits", async () => {
  const fetchMock = mockFetch(() => Promise.resolve(json({ hits: [HIT] })));
  renderPage();
  await submitSearch();

  const result = await screen.findByTestId("result-1");
  const url = fetchMock.mock.calls[1][0];
  expect(url).toMatch(/q=sidewalk(\+|%20)repair/);
  expect(url).toContain("city=birch");
  expect(result).toHaveTextContent("Birch City Council, 2024-03-12, minutes, item 9.B");
  const span = screen.getByTestId("result-span");
  expect(span).toHaveTextContent("p. 1");
  expect(span).toHaveAttribute("href", "/doc/birch-minutes-201?unit=1&start=301&end=468");
});

test("shows No results for an empty hit list", async () => {
  mockFetch(() => Promise.resolve(json({ hits: [] })));
  renderPage();
  await submitSearch();
  expect(await screen.findByTestId("empty")).toHaveTextContent("No results.");
});

test("shows the error code when the request fails", async () => {
  mockFetch(() => Promise.resolve(json({ error: "validation_error" }, 422)));
  renderPage();
  await submitSearch();
  expect(await screen.findByTestId("error")).toHaveTextContent("Request failed: validation_error");
});

test("shows the loading state while the request is pending", async () => {
  mockFetch(() => new Promise(() => {}));
  renderPage("/?q=sidewalk+repair");
  expect(screen.getByTestId("loading")).toHaveTextContent("Loading...");
  await screen.findByRole("option", { name: "Alder" });
  expect(screen.getByTestId("loading")).toBeInTheDocument();
});

test("treats an empty q in the address as no search", async () => {
  const fetchMock = mockFetch(() => Promise.resolve(json({ hits: [] })));
  renderPage("/?q=");
  expect(screen.getByText("Enter a search to begin.")).toBeInTheDocument();
  expect(screen.queryByTestId("loading")).not.toBeInTheDocument();
  await screen.findByRole("option", { name: "Alder" });
  expect(fetchMock).toHaveBeenCalledTimes(1);
});
