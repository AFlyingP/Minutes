import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";

import type { DocumentDetail, Unit } from "../types";
import DocPage from "./DocPage";

const DOCUMENT: DocumentDetail = {
  id: "birch-minutes-201",
  city_id: "birch",
  kind: "minutes",
  unit_kind: "page",
  unit_count: 3,
  source_url: "https://source.invalid/minutes.pdf",
  meeting: {
    id: "birch-201",
    body: "City Council",
    meeting_date: "2024-03-12",
    title: "Council meeting",
    recording_url: "https://video.invalid/birch/201",
  },
};
const PAGE: Unit = {
  document_id: DOCUMENT.id,
  unit_index: 1,
  unit_kind: "page",
  text: "Approve Granite Works Inc. contract.",
  boxes: [
    [0, 7, 0.1, 0.2, 0.2, 0.25],
    [8, 15, 0.2, 0.2, 0.3, 0.25],
    [16, 21, 0.3, 0.2, 0.4, 0.25],
    [22, 26, 0.4, 0.2, 0.45, 0.25],
    [27, 36, 0.45, 0.2, 0.6, 0.25],
  ],
  text_source: "pdf",
  start_ms: null,
  end_ms: null,
  speaker: null,
  width_pt: 612,
  height_pt: 792,
  label: "p. 1",
  recording_link: null,
};

function mockFetch(doc = DOCUMENT, unit = PAGE) {
  const fetchMock = vi.fn(async (url: string) => {
    if (url === "/api/cities") return Response.json({ cities: [{ id: "birch", name: "Birch" }] });
    if (url === `/api/documents/${doc.id}`) return Response.json(doc);
    const match = url.match(/\/units\/(\d+)$/);
    if (match) {
      const index = Number(match[1]);
      return Response.json({
        ...unit,
        unit_index: index,
        label: unit.unit_kind === "page" ? `p. ${index}` : unit.label,
      });
    }
    throw new Error(`unexpected request ${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function renderPage(url = `/doc/${DOCUMENT.id}?unit=1&start=8&end=26`) {
  return render(
    <MemoryRouter initialEntries={[url]}>
      <Routes>
        <Route path="/doc/:id" element={<DocPage />} />
      </Routes>
    </MemoryRouter>,
  );
}

afterEach(() => vi.unstubAllGlobals());

test("renders page image, text, and mark for the given range", async () => {
  const fetchMock = mockFetch();
  renderPage();
  expect(screen.getByTestId("loading")).toHaveTextContent("Loading...");
  expect(await screen.findByTestId("doc-title")).toHaveTextContent(
    "Birch City Council, 2024-03-12, minutes",
  );
  expect(screen.getByTestId("page-image").querySelector("img")).toHaveAttribute(
    "src",
    "/api/documents/birch-minutes-201/units/1/image?dpi=110",
  );
  expect(screen.getByTestId("unit-text").textContent).toBe(PAGE.text);
  expect(screen.getByTestId("mark")).toHaveTextContent(PAGE.text.slice(8, 26));
  expect(fetchMock).toHaveBeenCalledWith("/api/documents/birch-minutes-201/units/1");
});

test("draws one highlight per overlapping box", async () => {
  mockFetch();
  renderPage(`/doc/${DOCUMENT.id}?start=15&end=22`);
  await screen.findByTestId("page-image");
  const highlights = screen.getAllByTestId("highlight");
  expect(highlights).toHaveLength(1);
  expect(highlights[0].style.left).toBe("30%");
  expect(highlights[0].style.top).toBe("20%");
  expect(parseFloat(highlights[0].style.width)).toBeCloseTo(10);
  expect(parseFloat(highlights[0].style.height)).toBeCloseTo(5);
  fireEvent.click(screen.getByTestId("unit-next"));
  await screen.findByText("p. 2 of 3");
  expect(screen.queryByTestId("mark")).not.toBeInTheDocument();
  expect(screen.queryByTestId("highlight")).not.toBeInTheDocument();
});

test("renders a segment with time range and recording link", async () => {
  const doc: DocumentDetail = {
    ...DOCUMENT,
    id: "birch-transcript-201",
    kind: "transcript",
    unit_kind: "segment",
    unit_count: 2,
  };
  const segment: Unit = {
    ...PAGE,
    document_id: doc.id,
    unit_kind: "segment",
    text_source: "caption",
    boxes: null,
    width_pt: null,
    height_pt: null,
    start_ms: 0,
    end_ms: 29000,
    label: "00:00:00",
    recording_link: `${doc.meeting.recording_url}&entrytime=0`,
  };
  mockFetch(doc, segment);
  const view = renderPage(`/doc/${doc.id}?start=8&end=26`);
  expect(await screen.findByTestId("segment-time")).toHaveTextContent("00:00:00 to 00:00:29");
  expect(screen.getByTestId("recording-link")).toHaveTextContent("Open recording at 00:00:00");
  expect(screen.getByTestId("recording-link")).toHaveAttribute("href", segment.recording_link);
  expect(screen.getByTestId("unit-label")).toHaveTextContent("00:00:00 (segment 1 of 2)");
  expect(screen.queryByTestId("page-image")).not.toBeInTheDocument();
  expect(screen.getByTestId("mark")).toHaveTextContent(PAGE.text.slice(8, 26));
  view.unmount();
  mockFetch(doc, {
    ...segment,
    start_ms: 70000,
    end_ms: 90000,
    label: "00:01:10",
    recording_link: doc.meeting.recording_url,
  });
  renderPage(`/doc/${doc.id}?unit=2`);
  expect(await screen.findByTestId("recording-link")).toHaveTextContent(
    "Open recording (seek to 00:01:10)",
  );
});

test("shows No public recording link when the link is null", async () => {
  const doc: DocumentDetail = {
    ...DOCUMENT,
    id: "alder-transcript-102",
    kind: "transcript",
    unit_kind: "segment",
    unit_count: 1,
    meeting: { ...DOCUMENT.meeting, recording_url: null },
  };
  mockFetch(doc, {
    ...PAGE,
    document_id: doc.id,
    unit_kind: "segment",
    boxes: null,
    width_pt: null,
    height_pt: null,
    start_ms: 0,
    end_ms: 29000,
    label: "00:00:00",
    recording_link: null,
  });
  renderPage(`/doc/${doc.id}`);
  expect(await screen.findByText("No public recording link.")).toBeInTheDocument();
  expect(screen.queryByTestId("recording-link")).not.toBeInTheDocument();
});

test("disables prev on the first unit and next on the last", async () => {
  const fetchMock = mockFetch();
  renderPage(`/doc/${DOCUMENT.id}`);
  await screen.findByTestId("unit-label");
  expect(screen.getByTestId("unit-prev")).toBeDisabled();
  expect(screen.getByTestId("unit-next")).toBeEnabled();
  fireEvent.click(screen.getByTestId("unit-next"));
  await screen.findByText("p. 2 of 3");
  fireEvent.click(screen.getByTestId("unit-next"));
  await screen.findByText("p. 3 of 3");
  expect(screen.getByTestId("unit-next")).toBeDisabled();
  fireEvent.click(screen.getByTestId("unit-prev"));
  await screen.findByText("p. 2 of 3");
  fetchMock.mockResolvedValueOnce(Response.json({ error: "not_found" }, { status: 404 }));
  fireEvent.click(screen.getByTestId("unit-prev"));
  expect(await screen.findByTestId("error")).toHaveTextContent("Request failed: not_found");
});
