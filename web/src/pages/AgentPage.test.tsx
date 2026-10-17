import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

import AgentPage from "./AgentPage";

const RESPONSE = {
  status: "completed",
  answer: "RESOLUTION 2024-07 passed 4-1",
  citations: [{ document_id: "birch-minutes-201", unit_index: 2, label: "p. 2" }],
  steps: [
    {
      index: 1,
      tool: "search",
      arguments: { query: "vote" },
      result_summary: "5 results",
      error: null,
    },
    {
      index: 2,
      tool: "sql",
      arguments: { query: "votes" },
      result_summary: "10 rows",
      error: null,
    },
  ],
  cost_usd: 0,
  limit: null,
  max_steps: 8,
};

function renderPage() {
  render(
    <MemoryRouter>
      <AgentPage />
    </MemoryRouter>,
  );
}

function mockFetch(response: unknown = RESPONSE, status = 200) {
  const fetchMock = vi.fn(() =>
    Promise.resolve(
      Response.json(status === 200 ? response : { error: "internal_error" }, { status }),
    ),
  );
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function submit(question = "Which contract passed?") {
  fireEvent.change(screen.getByTestId("agent-input"), { target: { value: question } });
  fireEvent.click(screen.getByTestId("agent-submit"));
}

afterEach(() => vi.unstubAllGlobals());

test("renders steps with tool and summary and the answer", async () => {
  mockFetch();
  renderPage();
  submit();
  expect(await screen.findByTestId("step-1")).toHaveTextContent("1. search - 5 results");
  expect(screen.getByTestId("step-2")).toHaveTextContent("2. sql - 10 rows");
  expect(screen.getByTestId("agent-answer")).toHaveTextContent("RESOLUTION 2024-07 passed 4-1");
  expect(within(screen.getByTestId("step-1")).getByText("Arguments")).toBeInTheDocument();
});

test("shows steps n of 8", async () => {
  mockFetch();
  renderPage();
  submit();
  expect(await screen.findByTestId("agent-status")).toHaveTextContent("steps 2 of 8");
});

test("shows the stopped limit for limit_exceeded", async () => {
  mockFetch({
    ...RESPONSE,
    status: "limit_exceeded",
    limit: "steps",
    steps: Array.from({ length: 8 }, (_, index) => ({
      index: index + 1,
      tool: "search",
      arguments: {},
      result_summary: "5 results",
      error: null,
    })),
  });
  renderPage();
  submit();
  expect(await screen.findByTestId("agent-status")).toHaveTextContent(
    "steps 8 of 8 · stopped: steps limit",
  );
});

test("shows error status", async () => {
  mockFetch({ ...RESPONSE, status: "error" });
  renderPage();
  submit();
  expect(await screen.findByTestId("agent-status")).toHaveTextContent("steps 2 of 8 · error");
});

test("renders citation links to the viewer", async () => {
  mockFetch();
  renderPage();
  submit();
  const citation = await screen.findByRole("link", { name: "[birch-minutes-201 p. 2]" });
  expect(citation).toHaveAttribute("href", "/doc/birch-minutes-201?unit=2");
});

test("does not submit an empty question", () => {
  const fetchMock = mockFetch();
  renderPage();
  fireEvent.click(screen.getByTestId("agent-submit"));
  expect(fetchMock).not.toHaveBeenCalled();
});
