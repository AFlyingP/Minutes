import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

import FactsPage from "./FactsPage";

const CITIES = {
  cities: [
    {
      id: "alder",
      name: "Alder",
      state: "WA",
      bodies: ["City Council"],
      date_min: null,
      date_max: null,
      documents: 2,
    },
    {
      id: "birch",
      name: "Birch",
      state: "CA",
      bodies: ["City Council", "Planning Commission"],
      date_min: null,
      date_max: null,
      documents: 4,
    },
  ],
};

function row(data: Record<string, unknown> = {}) {
  return {
    id: 17,
    document_id: "birch-minutes-201",
    meeting_id: "birch-201",
    city_id: "birch",
    body: "City Council",
    meeting_date: "2024-03-12",
    doc_kind: "minutes",
    unit_index: 2,
    unit_kind: "page",
    quote_start: 12,
    quote_end: 250,
    quote: "RESOLUTION 2024-07 passed",
    label: "p. 2",
    data: {
      subject_identifier: "RESOLUTION 2024-07",
      ayes: 4,
      noes: 1,
      abstain: 0,
      absent: 0,
      outcome: "passed",
      ...data,
    },
  };
}

function renderPage() {
  render(
    <MemoryRouter>
      <FactsPage />
    </MemoryRouter>,
  );
}

function mockFetch(
  responder: (url: string) => unknown = () => ({ kind: "votes", total: 1, rows: [row()] }),
) {
  const fetchMock = vi.fn((input: string | URL | Request) => {
    const url = String(input);
    if (url === "/api/cities") return Promise.resolve(Response.json(CITIES));
    return Promise.resolve(Response.json(responder(url)));
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => vi.unstubAllGlobals());

test("renders vote columns and rows from the response", async () => {
  mockFetch();
  renderPage();
  const table = await screen.findByTestId("facts-table");
  expect(within(table).getByText("Date")).toBeInTheDocument();
  expect(within(table).getByText("Subject")).toBeInTheDocument();
  expect(within(table).getByText("Ayes")).toBeInTheDocument();
  const tr = within(table).getByText("RESOLUTION 2024-07").closest("tr");
  expect(tr).toHaveTextContent("4");
  expect(tr).toHaveTextContent("1");
  expect(tr).toHaveTextContent("passed");
  expect(within(tr as HTMLElement).getByTestId("fact-source")).toHaveTextContent("p. 2");
  expect(within(tr as HTMLElement).getByTestId("fact-source")).toHaveAttribute(
    "href",
    "/doc/birch-minutes-201?unit=2&start=12&end=250",
  );
});

test("switching kind requests the matching path and shows its columns", async () => {
  const fetchMock = mockFetch((url) =>
    url.startsWith("/api/facts/amounts")
      ? {
          kind: "amounts",
          total: 1,
          rows: [
            row({ amount_usd: 184500, purpose: "Sidewalk repair", payee: "Granite Works Inc." }),
          ],
        }
      : { kind: "votes", total: 1, rows: [row()] },
  );
  renderPage();
  await screen.findByTestId("facts-table");
  await screen.findByRole("option", { name: "Dollar amounts" });
  fireEvent.change(screen.getByTestId("facts-kind"), { target: { value: "amounts" } });
  expect(await screen.findByText("$184,500.00")).toBeInTheDocument();
  expect(screen.getByText("Purpose")).toBeInTheDocument();
  expect(fetchMock.mock.calls.some(([url]) => String(url).startsWith("/api/facts/amounts?"))).toBe(
    true,
  );
});

test("filters and text query are sent as parameters and reset the offset", async () => {
  const fetchMock = mockFetch(() => ({ kind: "votes", total: 100, rows: [row()] }));
  renderPage();
  await screen.findByTestId("facts-table");
  fireEvent.click(screen.getByTestId("facts-next"));
  await waitFor(() =>
    expect(fetchMock.mock.calls.some(([url]) => String(url).includes("offset=50"))).toBe(true),
  );
  await screen.findByRole("option", { name: "Birch" });
  fireEvent.change(screen.getByTestId("filter-city"), { target: { value: "birch" } });
  fireEvent.change(screen.getByTestId("filter-body"), { target: { value: "City Council" } });
  fireEvent.change(screen.getByTestId("filter-date-from"), { target: { value: "2024-03-01" } });
  fireEvent.change(screen.getByTestId("facts-q"), { target: { value: "resolution" } });
  await waitFor(() => {
    const url = String(fetchMock.mock.calls.at(-1)?.[0]);
    expect(url).toContain("city=birch");
    expect(url).toContain("body=City+Council");
    expect(url).toContain("date_from=2024-03-01");
    expect(url).toContain("q=resolution");
    expect(url).toContain("offset=0");
  });
});

test("pager disables at the ends and requests the next offset", async () => {
  const fetchMock = mockFetch(() => ({ kind: "votes", total: 51, rows: [row()] }));
  renderPage();
  await screen.findByTestId("facts-table");
  expect(screen.getByTestId("facts-prev")).toBeDisabled();
  expect(screen.getByTestId("facts-next")).toBeEnabled();
  fireEvent.click(screen.getByTestId("facts-next"));
  await waitFor(() => expect(String(fetchMock.mock.calls.at(-1)?.[0])).toContain("offset=50"));
  await waitFor(() => expect(screen.getByTestId("facts-next")).toBeDisabled());
  expect(screen.getByTestId("facts-prev")).toBeEnabled();
});

test("shows No rows for an empty result", async () => {
  mockFetch(() => ({ kind: "votes", total: 0, rows: [] }));
  renderPage();
  expect(await screen.findByTestId("empty")).toHaveTextContent("No rows.");
});

test("shows total rows", async () => {
  mockFetch(() => ({ kind: "votes", total: 23, rows: [row()] }));
  renderPage();
  expect(await screen.findByTestId("facts-total")).toHaveTextContent("23 rows");
});
