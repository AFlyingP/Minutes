import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";

import type {
  DocumentDetail,
  DocumentSummary,
  FactKind,
  Label,
  LabelIn,
  LabelType,
  Unit,
} from "../types";
import LabelPage from "./LabelPage";

const UNIT_TEXT = "Award the contract to Granite Works Inc. for sidewalk repair.";
const PASSAGE = {
  document_id: "birch-minutes-201",
  unit_index: 1,
  start: 22,
  end: 40,
  text: UNIT_TEXT.slice(22, 40),
};
const QUESTION: Label = {
  id: "q-birch-001",
  type: "question",
  city: "birch",
  author: "human",
  created_utc: "2024-05-01T00:00:00Z",
  human_reviewed: false,
  note: "",
  payload: {
    question: "Who received the contract?",
    answerable: true,
    question_type: "person",
    answer: "Granite Works Inc.",
    answer_type: "name",
    passages: [PASSAGE],
  },
};
const DOCUMENTS: DocumentSummary[] = [
  {
    id: "birch-agenda-201",
    meeting_id: "birch-201",
    city_id: "birch",
    kind: "agenda",
    body: "City Council",
    meeting_date: "2024-03-12",
    unit_kind: "page",
    unit_count: 3,
  },
  {
    id: "birch-minutes-201",
    meeting_id: "birch-201",
    city_id: "birch",
    kind: "minutes",
    body: "City Council",
    meeting_date: "2024-03-12",
    unit_kind: "page",
    unit_count: 3,
  },
  {
    id: "birch-transcript-201",
    meeting_id: "birch-201",
    city_id: "birch",
    kind: "transcript",
    body: "City Council",
    meeting_date: "2024-03-12",
    unit_kind: "segment",
    unit_count: 2,
  },
];

interface MockOptions {
  labels?: Label[];
  frozen?: boolean;
  writeError?: Response | Error;
  exportError?: Response;
  searchError?: Response;
}

function mockFetch(options: MockOptions = {}) {
  let labels = [...(options.labels ?? [])];
  const writes: { method: string; input: LabelIn }[] = [];
  const counters: Record<LabelType, number> = {
    question: labels.length,
    agenda_count: 0,
    extraction: 0,
    agent_task: 0,
  };
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    const parsed = new URL(url, "http://localhost");
    const path = parsed.pathname;
    if (path === "/api/cities")
      return Response.json({
        cities: [
          { id: "alder", name: "Alder" },
          { id: "birch", name: "Birch" },
        ],
      });
    if (path === "/api/labels/progress") {
      return Response.json({
        frozen: options.frozen ?? false,
        types: Object.fromEntries(
          Object.keys(counters).map((type) => [
            type,
            {
              count: labels.filter((entry) => entry.type === type).length,
              violations: ["More source labels needed"],
            },
          ]),
        ),
        human_reviewed: Object.fromEntries(
          Object.keys(counters).map((type) => [
            type,
            labels.filter((entry) => entry.type === type && entry.human_reviewed).length,
          ]),
        ),
      });
    }
    if (path === "/api/documents") {
      const documents = parsed.searchParams.get("city") === "birch" ? DOCUMENTS : [];
      const offset = Number(parsed.searchParams.get("offset"));
      return Response.json({
        total: documents.length,
        documents: documents.slice(offset, offset + 2),
      });
    }
    const document = DOCUMENTS.find(
      (entry) =>
        path === `/api/documents/${entry.id}` ||
        path.startsWith(`/api/documents/${entry.id}/units/`),
    );
    if (document) {
      const doc: DocumentDetail = {
        ...document,
        source_url: "https://source.invalid/document",
        meeting: {
          id: document.meeting_id,
          body: document.body,
          meeting_date: document.meeting_date,
          title: "Council meeting",
          recording_url: "https://video.invalid/birch/201",
        },
      };
      if (path.endsWith(document.id)) return Response.json(doc);
      const index = Number(path.split("/").at(-1));
      const common = {
        document_id: document.id,
        unit_index: index,
        text: index === 3 ? "Adopt ORDINANCE 1042" : UNIT_TEXT,
        speaker: null,
      };
      const unit: Unit =
        document.unit_kind === "page"
          ? {
              ...common,
              unit_kind: "page",
              text_source: "pdf",
              start_ms: null,
              end_ms: null,
              width_pt: 612,
              height_pt: 792,
              boxes: [[22, 40, 0.1, 0.2, 0.3, 0.25]],
              label: `p. ${index}`,
              recording_link: null,
            }
          : {
              ...common,
              unit_kind: "segment",
              text_source: "caption",
              start_ms: 0,
              end_ms: 29000,
              width_pt: null,
              height_pt: null,
              boxes: null,
              label: "00:00:00",
              recording_link: `${doc.meeting.recording_url}&entrytime=0`,
            };
      return Response.json(unit);
    }
    if (path === "/api/labels/text-search")
      return options.searchError?.clone() ?? Response.json({ unit_count: 0, matches: [] });
    if (path === "/api/labels/sample-meetings")
      return Response.json({
        meetings: [
          {
            meeting_id: "birch-201",
            body: "City Council",
            meeting_date: "2024-03-12",
            agenda_document_id: "birch-agenda-201",
            labelled: labels.some((entry) => entry.type === "agenda_count"),
          },
        ],
      });
    if (path === "/api/labels/export")
      return (
        options.exportError?.clone() ??
        Response.json({ path: "eval/labels/questions.jsonl", count: labels.length })
      );
    if (path === "/api/labels" && !init)
      return Response.json({
        labels: labels.filter(
          (entry) =>
            entry.type === parsed.searchParams.get("type") &&
            entry.city === parsed.searchParams.get("city"),
        ),
      });
    if (path.startsWith("/api/labels") && init) {
      if (options.writeError instanceof Error) throw options.writeError;
      if (options.writeError) return options.writeError.clone();
      const id = path.split("/")[3];
      if (path.endsWith("/review")) {
        const reviewed = JSON.parse(String(init.body)) as { human_reviewed: boolean };
        labels = labels.map((entry) => (entry.id === id ? { ...entry, ...reviewed } : entry));
        return Response.json(labels.find((entry) => entry.id === id));
      }
      if (init.method === "DELETE") {
        labels = labels.filter((entry) => entry.id !== id);
        return new Response(null, { status: 204 });
      }
      const input = JSON.parse(String(init.body)) as LabelIn;
      writes.push({ method: init.method!, input });
      const prefix = { question: "q", agenda_count: "c", extraction: "x", agent_task: "a" }[
        input.type
      ];
      const label: Label = {
        ...input,
        id: id ?? `${prefix}-${input.city}-${String(++counters[input.type]).padStart(3, "0")}`,
        created_utc: "2024-05-01T00:00:00Z",
        human_reviewed: false,
      };
      labels = [...labels.filter((entry) => entry.id !== label.id), label];
      return Response.json(label, { status: init.method === "POST" ? 201 : 200 });
    }
    throw new Error(`unexpected request ${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  return { fetchMock, writes };
}

function change(id: string, value: string) {
  fireEvent.change(screen.getByTestId(id), { target: { value } });
}

async function chooseCity() {
  await screen.findByRole("option", { name: "Birch" });
  change("label-city", "birch");
  await screen.findByRole("option", { name: "birch-transcript-201" });
}

async function openPage() {
  await chooseCity();
  change("label-document", "birch-minutes-201");
  await screen.findByTestId("unit-text");
}

async function chooseType(type: LabelType) {
  change("label-type", type);
  await waitFor(() => expect(screen.queryByTestId("loading")).not.toBeInTheDocument());
}

function selectPassage() {
  const root = screen.getByTestId("unit-text");
  const node = document.createTreeWalker(root, NodeFilter.SHOW_TEXT).nextNode()!;
  const start = node.textContent!.indexOf("Granite Works Inc.");
  const range = document.createRange();
  range.setStart(node, start);
  range.setEnd(node, start + "Granite Works Inc.".length);
  const selection = window.getSelection()!;
  selection.removeAllRanges();
  selection.addRange(range);
  fireEvent.click(screen.getByTestId("mark-passage"));
}

async function save() {
  fireEvent.click(screen.getByTestId("label-save"));
  await waitFor(() => expect(screen.queryByTestId("loading")).not.toBeInTheDocument());
}

afterEach(() => {
  window.getSelection()?.removeAllRanges();
  vi.unstubAllGlobals();
});

test("shows progress counts and violations", async () => {
  const options: MockOptions = { labels: [QUESTION] };
  const { fetchMock } = mockFetch(options);
  render(<LabelPage />);
  expect(screen.getByTestId("loading")).toHaveTextContent("Loading...");
  const progress = await screen.findByTestId("label-progress");
  expect(progress).toHaveTextContent("not frozen");
  expect(progress).toHaveTextContent("question: 1 labels, 0 reviewed");
  expect(progress).toHaveTextContent("agenda_count: 0 labels");
  expect(progress).toHaveTextContent("More source labels needed");
  await chooseCity();
  expect(fetchMock).toHaveBeenCalledWith("/api/documents?city=birch&limit=500&offset=2");
  fireEvent.click(screen.getByTestId("review-q-birch-001"));
  await waitFor(() => expect(progress).toHaveTextContent("question: 1 labels, 1 reviewed"));
  expect(screen.getByTestId("review-q-birch-001")).toBeChecked();
  fireEvent.click(screen.getByTestId("label-export"));
  await screen.findByText("Exported 1 labels to eval/labels/questions.jsonl");
  options.exportError = Response.json(
    { error: "label_rules", detail: ["More labels needed before export"] },
    { status: 409 },
  );
  fireEvent.click(screen.getByTestId("label-export"));
  expect(await screen.findByTestId("error-list")).toHaveTextContent(
    "More labels needed before export",
  );
  change("label-city", "alder");
  await waitFor(() => expect(screen.queryByTestId("loading")).not.toBeInTheDocument());
  expect(screen.getByTestId("label-list")).toBeEmptyDOMElement();
  expect(within(screen.getByTestId("label-document")).getAllByRole("option")).toHaveLength(1);
});

test("marks a passage from a text selection with exact offsets", async () => {
  const { writes } = mockFetch();
  render(<LabelPage />);
  await openPage();
  selectPassage();
  const passages = screen.getByTestId("passages");
  expect(passages).toHaveTextContent("Granite Works Inc.");
  const text = screen.getByTestId("unit-text");
  const range = document.createRange();
  range.setStart(text, 1);
  range.setEnd(text, 2);
  window.getSelection()!.removeAllRanges();
  window.getSelection()!.addRange(range);
  fireEvent.click(screen.getByTestId("mark-passage"));
  expect(within(passages).getAllByRole("listitem")).toHaveLength(2);
  fireEvent.click(within(passages).getAllByRole("button", { name: "remove" })[0]);
  change("question", "Who received the contract?");
  change("label-answer", "Granite Works Inc.");
  await save();
  const input = writes[0].input;
  expect(input.type).toBe("question");
  if (input.type !== "question" || !input.payload.answerable)
    throw new Error("expected answerable question");
  const passage = input.payload.passages[0];
  expect(passage.start).toBe(UNIT_TEXT.indexOf("Granite Works Inc."));
  expect(passage.end).toBe(passage.start + "Granite Works Inc.".length);
  expect(passage.text).toBe(UNIT_TEXT.slice(passage.start, passage.end));
  expect(passage.text).toBe("Granite Works Inc.");

  await chooseType("extraction");
  selectPassage();
  const records: Record<FactKind, Record<string, string | number | null>> = {
    motion: { text: "Award the contract", mover: "Lopez", seconder: null, outcome: "passed" },
    vote: {
      subject_identifier: "1042",
      ayes: 4,
      noes: 1,
      abstain: 0,
      absent: 0,
      outcome: "passed",
    },
    ordinance: {
      identifier: "1042",
      legislation_type: "ordinance",
      title: "Sidewalk repair",
      status: "adopted",
    },
    amount: { amount_usd: 184500, purpose: "Sidewalk repair", payee: null },
    statement: { speaker: "Lopez", role: null, summary: "Supported sidewalk repair" },
  };
  for (const kind of Object.keys(records) as FactKind[]) {
    change("kind", kind);
    for (const [name, value] of Object.entries(records[kind])) change(name, String(value ?? ""));
    if (kind === "vote") {
      fireEvent.click(screen.getByTestId("add-member"));
      change("member-name", "Lopez");
      change("member-value", "aye");
      fireEvent.click(screen.getByTestId("add-member"));
      const members = screen.getAllByTestId("member-name");
      fireEvent.click(
        within(members[1].closest("fieldset")!).getByRole("button", { name: "remove" }),
      );
    }
    await save();
    const last = writes.at(-1)!.input;
    expect(last.type).toBe("extraction");
    if (last.type !== "extraction") throw new Error("expected extraction");
    expect(last.payload.kind).toBe(kind);
    expect(last.payload.passage.text).toBe("Granite Works Inc.");
    expect(last.payload.record).toEqual(
      kind === "vote"
        ? { ...records[kind], members: [{ name: "Lopez", value: "aye" }] }
        : records[kind],
    );
    if (kind === "motion") {
      fireEvent.click(screen.getByTestId("edit-x-birch-001"));
      expect(screen.getByTestId("mover")).toHaveValue("Lopez");
      change("note", "Checked passage");
      await save();
      expect(writes.at(-1)!.method).toBe("PUT");
      fireEvent.click(screen.getByRole("button", { name: "New label" }));
      await waitFor(() => expect(screen.getByTestId("unit-text").textContent).toBe(UNIT_TEXT));
      selectPassage();
    }
  }
  fireEvent.click(within(screen.getByTestId("passages")).getByRole("button", { name: "remove" }));
  expect(screen.getByTestId("label-save")).toBeDisabled();
  selectPassage();
  expect(screen.getByTestId("label-save")).toBeEnabled();
  range.selectNodeContents(screen.getByTestId("unit-text").lastChild!);
  window.getSelection()!.removeAllRanges();
  window.getSelection()!.addRange(range);
  fireEvent.click(screen.getByTestId("mark-passage"));
  const extractionPassages = screen.getByTestId("passages");
  expect(within(extractionPassages).getAllByRole("listitem")).toHaveLength(1);
  expect(extractionPassages).toHaveTextContent("for sidewalk repair.");
  expect(extractionPassages).not.toHaveTextContent("Granite Works Inc.");
});

test("rejects an empty selection", async () => {
  mockFetch();
  render(<LabelPage />);
  await openPage();
  window.getSelection()!.removeAllRanges();
  fireEvent.click(screen.getByTestId("mark-passage"));
  expect(screen.getByTestId("error")).toHaveTextContent("Select text inside one page or segment.");
  const text = screen.getByTestId("unit-text").firstChild!;
  const range = document.createRange();
  range.setStart(text, 1);
  range.collapse(true);
  window.getSelection()!.addRange(range);
  fireEvent.click(screen.getByTestId("mark-passage"));
  expect(screen.getByTestId("passages")).toBeEmptyDOMElement();
  range.setStart(screen.getByTestId("unit-text"), 0);
  range.setEnd(text, 0);
  window.getSelection()!.removeAllRanges();
  window.getSelection()!.addRange(range);
  fireEvent.click(screen.getByTestId("mark-passage"));
  expect(screen.getByTestId("passages")).toBeEmptyDOMElement();
  range.setStart(screen.getByTestId("label-city").parentElement!.firstChild!, 0);
  range.setEnd(text, 10);
  window.getSelection()!.removeAllRanges();
  window.getSelection()!.addRange(range);
  fireEvent.click(screen.getByTestId("mark-passage"));
  expect(screen.getByTestId("error")).toHaveTextContent("Select text inside one page or segment.");
  expect(screen.getByTestId("passages")).toBeEmptyDOMElement();
});

test("saves a question and shows the new id", async () => {
  const { fetchMock, writes } = mockFetch();
  render(<LabelPage />);
  await openPage();
  for (const id of ["results", "answer", "facts-table"]) {
    expect(screen.queryByTestId(id)).not.toBeInTheDocument();
  }
  selectPassage();
  change("question", "Who received the contract?");
  change("question_type", "person");
  change("label-answer", "Granite Works Inc.");
  change("answer_type", "name");
  change("note", "Read source page");
  await save();
  expect(screen.getByTestId("saved")).toHaveTextContent("Saved q-birch-001");
  expect(screen.getByTestId("label-list")).toHaveTextContent("q-birch-001");
  expect(writes[0].input).toMatchObject({
    type: "question",
    city: "birch",
    author: "human",
    note: "Read source page",
    payload: { question_type: "person", answer_type: "name" },
  });
  const request = fetchMock.mock.calls.find(([, init]) => init?.method === "POST")![1]!;
  expect(request.headers).toEqual({ "Content-Type": "application/json" });
  fireEvent.click(screen.getByTestId("edit-q-birch-001"));
  expect(screen.getByTestId("question")).toHaveValue("Who received the contract?");
  expect(screen.getByTestId("label-city")).toBeDisabled();
  change("label-answer", "Granite Works");
  await save();
  expect(writes.at(-1)!.method).toBe("PUT");
  expect(writes.at(-1)!.input.author).toBe("human");
  fireEvent.click(screen.getByTestId("delete-q-birch-001"));
  await waitFor(() => expect(screen.queryByTestId("edit-q-birch-001")).not.toBeInTheDocument());

  await chooseType("agent_task");
  change("question", "Which documents support the contract?");
  change("label-answer", "The minutes and transcript");
  change("answer_type", "free_text");
  fireEvent.click(screen.getByTestId("add-document"));
  fireEvent.click(screen.getByTestId("unit-next"));
  await screen.findByText("p. 2 of 3");
  fireEvent.click(screen.getByTestId("unit-prev"));
  await screen.findByText("p. 1 of 3");
  change("label-document", "birch-transcript-201");
  await screen.findByTestId("segment-time");
  expect(screen.queryByTestId("page-image")).not.toBeInTheDocument();
  fireEvent.click(screen.getByTestId("add-document"));
  fireEvent.click(
    within(screen.getByTestId("document-ids")).getAllByRole("button", { name: "remove" })[1],
  );
  fireEvent.click(screen.getByTestId("add-document"));
  await save();
  expect(writes.at(-1)!.input).toEqual({
    type: "agent_task",
    city: "birch",
    author: "human",
    note: "",
    payload: {
      question: "Which documents support the contract?",
      answer: "The minutes and transcript",
      answer_type: "free_text",
      document_ids: ["birch-minutes-201", "birch-transcript-201"],
    },
  });
  fireEvent.click(screen.getByTestId("edit-a-birch-001"));
  expect(screen.getByTestId("label-answer")).toHaveValue("The minutes and transcript");
  await save();
  expect(writes.at(-1)!.method).toBe("PUT");
});

test("lists validation messages from a 422 response", async () => {
  const options: MockOptions = {
    labels: [QUESTION],
    writeError: Response.json(
      {
        error: "label_invalid",
        detail: ["Question is required", "Passage text must match source"],
      },
      { status: 422 },
    ),
  };
  mockFetch(options);
  render(<LabelPage />);
  await chooseCity();
  await save();
  expect(
    within(screen.getByTestId("error-list"))
      .getAllByRole("listitem")
      .map((item) => item.textContent),
  ).toEqual(["Question is required", "Passage text must match source"]);
  fireEvent.click(screen.getByTestId("edit-q-birch-001"));
  await save();
  expect(screen.getByTestId("error-list")).toHaveTextContent("Passage text must match source");
  options.writeError = new Response("invalid response", { status: 422 });
  await save();
  expect(screen.getByTestId("error")).toHaveTextContent("Request failed: 422");
  expect(screen.queryByTestId("error-list")).not.toBeInTheDocument();
  options.writeError = Response.json({ detail: "Invalid request" }, { status: 422 });
  await save();
  expect(screen.getByTestId("error")).toHaveTextContent("Request failed: 422");
  options.writeError = new TypeError("connection failed");
  await save();
  expect(screen.getByTestId("error")).toHaveTextContent("Request failed: network");
});

test("runs an absence search and adds it to the form", async () => {
  const options: MockOptions = {};
  const { fetchMock, writes } = mockFetch(options);
  render(<LabelPage />);
  await chooseCity();
  fireEvent.click(screen.getByTestId("answerable"));
  change("question", "Was a quarry approved?");
  change("reason", "entity_not_in_corpus");
  for (const query of ["quarry", "stone mine", "excavation"]) {
    change("absence-query", query);
    fireEvent.click(screen.getByTestId("absence-search"));
    await waitFor(() =>
      expect(screen.getByTestId("absence-result")).toHaveTextContent(`0 units contain "${query}"`),
    );
    await waitFor(() => expect(screen.queryByTestId("loading")).not.toBeInTheDocument());
    fireEvent.click(screen.getByTestId("absence-add"));
  }
  expect(fetchMock).toHaveBeenCalledWith("/api/labels/text-search?city=birch&q=stone+mine");
  fireEvent.click(
    within(screen.getByTestId("absence-searches")).getAllByRole("button", { name: "remove" })[2],
  );
  change("absence-query", "unsearched term");
  fireEvent.click(screen.getByTestId("absence-add"));
  await save();
  expect(writes[0].input).toEqual({
    type: "question",
    city: "birch",
    author: "human",
    note: "",
    payload: {
      question: "Was a quarry approved?",
      answerable: false,
      reason: "entity_not_in_corpus",
      absence_searches: [
        { query: "quarry", hits: 0 },
        { query: "stone mine", hits: 0 },
        { query: "excavation", hits: 0 },
      ],
    },
  });
  fireEvent.click(screen.getByTestId("edit-q-birch-001"));
  expect(screen.getByTestId("answerable")).not.toBeChecked();
  expect(screen.getByTestId("reason")).toHaveValue("entity_not_in_corpus");
  await save();
  expect(writes.at(-1)!.method).toBe("PUT");
  options.searchError = Response.json(
    { error: "validation_error", detail: "Query is too short" },
    { status: 422 },
  );
  change("absence-query", "x");
  fireEvent.click(screen.getByTestId("absence-search"));
  expect(await screen.findByTestId("error")).toHaveTextContent("Request failed: validation_error");
});

test("shows Labels are frozen on a 409", async () => {
  const frozen = Response.json({ error: "labels_frozen" }, { status: 409 });
  mockFetch({ labels: [QUESTION], frozen: true, writeError: frozen, exportError: frozen });
  render(<LabelPage />);
  await chooseCity();
  expect(within(screen.getByTestId("label-progress")).getByText("frozen")).toBeInTheDocument();
  await save();
  expect(screen.getByTestId("error")).toHaveTextContent("Labels are frozen.");
  fireEvent.click(screen.getByTestId("review-q-birch-001"));
  await waitFor(() => expect(screen.queryByTestId("loading")).not.toBeInTheDocument());
  expect(screen.getByTestId("review-q-birch-001")).not.toBeChecked();
  fireEvent.click(screen.getByTestId("delete-q-birch-001"));
  await waitFor(() => expect(screen.queryByTestId("loading")).not.toBeInTheDocument());
  expect(screen.getByTestId("error")).toHaveTextContent("Labels are frozen.");
  fireEvent.click(screen.getByTestId("edit-q-birch-001"));
  await save();
  expect(screen.getByTestId("error")).toHaveTextContent("Labels are frozen.");
  fireEvent.click(screen.getByTestId("label-export"));
  expect(await screen.findByTestId("error")).toHaveTextContent("Labels are frozen.");
});

test("count field equals the number of agenda items entered", async () => {
  const { writes } = mockFetch();
  render(<LabelPage />);
  await chooseCity();
  await chooseType("agenda_count");
  fireEvent.click(within(screen.getByTestId("sample-meetings")).getByRole("button"));
  await screen.findByTestId("page-image");
  expect(screen.getByTestId("meeting_id")).toHaveValue("birch-201");
  expect(screen.getByTestId("label-document")).toHaveValue("birch-agenda-201");
  expect(screen.getByTestId("count")).toHaveValue("0");
  fireEvent.click(screen.getByTestId("add-item"));
  change("identifier", "9.B");
  change("title", "Sidewalk repair");
  change("start_page", "2");
  fireEvent.click(screen.getByTestId("add-item"));
  expect(screen.getByTestId("count")).toHaveValue("2");
  const items = within(screen.getByTestId("label-form")).getAllByRole("group");
  fireEvent.change(within(items[1]).getByTestId("identifier"), { target: { value: "10.A" } });
  fireEvent.change(within(items[1]).getByTestId("title"), { target: { value: "Ordinance" } });
  fireEvent.click(within(items[1]).getByRole("button", { name: "remove" }));
  expect(screen.getByTestId("count")).toHaveValue("1");
  expect(screen.getByTestId("count")).toHaveAttribute("readonly");
  change("meeting_id", "birch-201");
  await save();
  expect(writes[0].input).toEqual({
    type: "agenda_count",
    city: "birch",
    author: "human",
    note: "",
    payload: {
      meeting_id: "birch-201",
      count: 1,
      items: [{ identifier: "9.B", title: "Sidewalk repair", start_page: 2 }],
    },
  });
  expect(screen.getByTestId("sample-meetings")).toHaveTextContent("(labelled)");
  fireEvent.click(screen.getByTestId("edit-c-birch-001"));
  expect(screen.getByTestId("identifier")).toHaveValue("9.B");
  fireEvent.click(screen.getByTestId("add-item"));
  await save();
  const last = writes.at(-1)!;
  expect(last.method).toBe("PUT");
  if (last.input.type !== "agenda_count") throw new Error("expected agenda count");
  expect(last.input.payload.count).toBe(last.input.payload.items.length);
});
