import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";

import {
  ApiError,
  createLabel,
  deleteLabel,
  exportLabels,
  fetchCities,
  fetchDocument,
  fetchDocuments,
  fetchLabels,
  fetchProgress,
  fetchSampleMeetings,
  fetchUnit,
  reviewLabel,
  textSearch,
  updateLabel,
} from "../api";
import type {
  AbsenceSearch,
  AgendaItem,
  AnswerType,
  City,
  DocumentDetail,
  DocumentSummary,
  FactKind,
  FactRecord,
  Label,
  LabelIn,
  LabelProgress,
  LabelType,
  Passage,
  QuestionType,
  Reason,
  SampleMeeting,
  TextSearchResponse,
  Unit,
  VoteMember,
} from "../types";
import { UnitControls, UnitViewer } from "./DocPage";

const LABEL_TYPES: Record<LabelType, string> = {
  question: "Question",
  agenda_count: "Agenda count",
  extraction: "Extraction case",
  agent_task: "Agent task",
};
const ANSWER_TYPES: AnswerType[] = ["number", "name", "date", "identifier", "free_text"];
const QUESTION_TYPES: QuestionType[] = [
  "vote",
  "amount",
  "legislation",
  "person",
  "date",
  "multi_passage",
];
const REASONS: Reason[] = [
  "entity_not_in_corpus",
  "date_outside_corpus",
  "fact_not_recorded",
  "false_premise",
];
const RECORD_DEFAULTS: Record<FactKind, FactRecord> = {
  motion: { text: "", mover: null, seconder: null, outcome: "unknown" },
  vote: {
    subject_identifier: null,
    ayes: 0,
    noes: 0,
    abstain: 0,
    absent: 0,
    members: [],
    outcome: "passed",
  },
  ordinance: { identifier: "", legislation_type: "ordinance", title: "", status: "unknown" },
  amount: { amount_usd: 0, purpose: "", payee: null },
  statement: { speaker: "", role: null, summary: "" },
};

interface RecordField {
  name: string;
  input: "text" | "number" | "select";
  nullable?: boolean;
  options?: string[];
}

const RECORD_FIELDS: Record<FactKind, RecordField[]> = {
  motion: [
    { name: "text", input: "text" },
    { name: "mover", input: "text", nullable: true },
    { name: "seconder", input: "text", nullable: true },
    {
      name: "outcome",
      input: "select",
      options: ["passed", "failed", "withdrawn", "tabled", "unknown"],
    },
  ],
  vote: [
    { name: "subject_identifier", input: "text", nullable: true },
    ...["ayes", "noes", "abstain", "absent"].map((name): RecordField => ({
      name,
      input: "number",
    })),
    { name: "outcome", input: "select", options: ["passed", "failed"] },
  ],
  ordinance: [
    { name: "identifier", input: "text" },
    {
      name: "legislation_type",
      input: "select",
      options: ["ordinance", "resolution", "council_bill", "other"],
    },
    { name: "title", input: "text" },
    {
      name: "status",
      input: "select",
      options: ["introduced", "passed", "adopted", "failed", "referred", "held", "unknown"],
    },
  ],
  amount: [
    { name: "amount_usd", input: "number" },
    { name: "purpose", input: "text" },
    { name: "payee", input: "text", nullable: true },
  ],
  statement: [
    { name: "speaker", input: "text" },
    { name: "role", input: "text", nullable: true },
    { name: "summary", input: "text" },
  ],
};

interface Draft {
  question: string;
  answerable: boolean;
  question_type: QuestionType;
  answer: string;
  answer_type: AnswerType;
  reason: Reason;
  meeting_id: string;
  items: AgendaItem[];
  kind: FactKind;
  record: FactRecord;
  passages: Passage[];
  absence_searches: AbsenceSearch[];
  document_ids: string[];
  note: string;
}

function emptyDraft(): Draft {
  return {
    question: "",
    answerable: true,
    question_type: "vote",
    answer: "",
    answer_type: "number",
    reason: "entity_not_in_corpus",
    meeting_id: "",
    items: [],
    kind: "motion",
    record: { ...RECORD_DEFAULTS.motion },
    passages: [],
    absence_searches: [],
    document_ids: [],
    note: "",
  };
}

function textOffset(root: Element, container: Node, offset: number): number {
  const boundary = document.createRange();
  boundary.setStart(container, offset);
  boundary.collapse(true);
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let length = 0;
  for (let node = walker.nextNode() as Text | null; node; node = walker.nextNode() as Text | null) {
    if (node === container) return length + offset;
    if (boundary.comparePoint(node, 0) >= 0) return length;
    length += node.data.length;
  }
  return length;
}

export default function LabelPage() {
  const [initial, setInitial] = useState<{ cities: City[]; progress: LabelProgress }>();
  const [city, setCity] = useState("");
  const [type, setType] = useState<LabelType>("question");
  const [catalog, setCatalog] = useState<{
    key: string;
    documents: DocumentSummary[];
    labels: Label[];
    samples: SampleMeeting[];
  }>();
  const [documentId, setDocumentId] = useState("");
  const [index, setIndex] = useState(1);
  const [source, setSource] = useState<{ key: string; doc: DocumentDetail; unit: Unit }>();
  const [draft, setDraft] = useState<Draft>(emptyDraft);
  const [editing, setEditing] = useState<string>();
  const [query, setQuery] = useState("");
  const [absence, setAbsence] = useState<{ query: string; result: TextSearchResponse }>();
  const [error, setError] = useState<string>();
  const [messages, setMessages] = useState<string[]>([]);
  const [saved, setSaved] = useState<string>();
  const [exported, setExported] = useState<string>();
  const [busy, setBusy] = useState(false);
  const browser = useRef<HTMLElement>(null);
  const catalogKey = `${city}/${type}`;
  const sourceKey = `${documentId}/${index}`;
  const currentCatalog = catalog?.key === catalogKey ? catalog : undefined;
  const currentSource = source?.key === sourceKey ? source : undefined;

  const showError = useCallback((err: unknown) => {
    if (err instanceof ApiError) {
      setError(err.code === "labels_frozen" ? "Labels are frozen." : `Request failed: ${err.code}`);
      setMessages(err.messages);
    } else {
      setError("Request failed: network");
      setMessages([]);
    }
  }, []);

  useEffect(() => {
    let stale = false;
    Promise.all([fetchCities(), fetchProgress()]).then(
      ([cities, progress]) => {
        if (!stale) setInitial({ cities, progress });
      },
      (err: unknown) => {
        if (!stale) showError(err);
      },
    );
    return () => {
      stale = true;
    };
  }, [showError]);

  useEffect(() => {
    if (!city) return;
    let stale = false;
    async function load() {
      try {
        // The browser must include documents beyond the API's first page.
        const documents: DocumentSummary[] = [];
        let total: number;
        do {
          const page = await fetchDocuments({
            city,
            limit: "500",
            offset: String(documents.length),
          });
          documents.push(...page.documents);
          total = page.total;
        } while (documents.length < total);
        const [labels, samples] = await Promise.all([
          fetchLabels(type, city),
          type === "agenda_count" ? fetchSampleMeetings(city) : Promise.resolve([]),
        ]);
        if (!stale) setCatalog({ key: catalogKey, documents, labels, samples });
      } catch (err) {
        if (!stale) showError(err);
      }
    }
    void load();
    return () => {
      stale = true;
    };
  }, [city, type, catalogKey, showError]);

  useEffect(() => {
    if (!documentId) return;
    let stale = false;
    Promise.all([fetchDocument(documentId), fetchUnit(documentId, index)]).then(
      ([doc, unit]) => {
        if (!stale) setSource({ key: sourceKey, doc, unit });
      },
      (err: unknown) => {
        if (!stale) showError(err);
      },
    );
    return () => {
      stale = true;
    };
  }, [documentId, index, sourceKey, showError]);

  function resetForm() {
    setDraft(emptyDraft());
    setEditing(undefined);
    setSaved(undefined);
    setExported(undefined);
    setError(undefined);
    setMessages([]);
  }

  function openDocument(id: string) {
    setDocumentId(id);
    setIndex(1);
    setError(undefined);
  }

  async function refresh() {
    const [labels, progress, samples] = await Promise.all([
      fetchLabels(type, city),
      fetchProgress(),
      type === "agenda_count" ? fetchSampleMeetings(city) : Promise.resolve([]),
    ]);
    setCatalog((previous) => previous && { ...previous, key: catalogKey, labels, samples });
    setInitial((previous) => previous && { ...previous, progress });
  }

  async function action(work: () => Promise<void>) {
    setBusy(true);
    setError(undefined);
    setMessages([]);
    try {
      await work();
    } catch (err) {
      showError(err);
    } finally {
      setBusy(false);
    }
  }

  function markPassage() {
    const selection = window.getSelection();
    const text = browser.current?.querySelector('[data-testid="unit-text"]');
    if (
      !selection ||
      selection.rangeCount !== 1 ||
      selection.isCollapsed ||
      !text ||
      !currentSource
    ) {
      setError("Select text inside one page or segment.");
      return;
    }
    const range = selection.getRangeAt(0);
    if (!text.contains(range.startContainer) || !text.contains(range.endContainer)) {
      setError("Select text inside one page or segment.");
      return;
    }
    const start = textOffset(text, range.startContainer, range.startOffset);
    const end = textOffset(text, range.endContainer, range.endOffset);
    if (start === end) {
      setError("Select text inside one page or segment.");
      return;
    }
    const passage: Passage = {
      document_id: currentSource.unit.document_id,
      unit_index: currentSource.unit.unit_index,
      start,
      end,
      text: currentSource.unit.text.slice(start, end),
    };
    setDraft({
      ...draft,
      passages: type === "extraction" ? [passage] : [...draft.passages, passage],
    });
    setError(undefined);
  }

  function edit(label: Label) {
    resetForm();
    const next = emptyDraft();
    next.note = label.note;
    switch (label.type) {
      case "question":
        next.question = label.payload.question;
        next.answerable = label.payload.answerable;
        if (label.payload.answerable) {
          next.question_type = label.payload.question_type;
          next.answer = label.payload.answer;
          next.answer_type = label.payload.answer_type;
          next.passages = label.payload.passages;
        } else {
          next.reason = label.payload.reason;
          next.absence_searches = label.payload.absence_searches;
        }
        break;
      case "agenda_count":
        next.meeting_id = label.payload.meeting_id;
        next.items = label.payload.items;
        break;
      case "extraction":
        next.kind = label.payload.kind;
        next.record = label.payload.record;
        next.passages = [label.payload.passage];
        break;
      case "agent_task":
        next.question = label.payload.question;
        next.answer = label.payload.answer;
        next.answer_type = label.payload.answer_type;
        next.document_ids = label.payload.document_ids;
    }
    setDraft(next);
    setEditing(label.id);
    if (next.passages.length) {
      setDocumentId(next.passages[0].document_id);
      setIndex(next.passages[0].unit_index);
    }
  }

  function labelInput(): LabelIn {
    const common = { city, author: "human" as const, note: draft.note };
    switch (type) {
      case "question":
        return {
          ...common,
          type,
          payload: draft.answerable
            ? {
                question: draft.question,
                answerable: true,
                question_type: draft.question_type,
                answer: draft.answer,
                answer_type: draft.answer_type,
                passages: draft.passages,
              }
            : {
                question: draft.question,
                answerable: false,
                reason: draft.reason,
                absence_searches: draft.absence_searches,
              },
        };
      case "agenda_count":
        return {
          ...common,
          type,
          payload: { meeting_id: draft.meeting_id, count: draft.items.length, items: draft.items },
        };
      case "extraction":
        return {
          ...common,
          type,
          payload: { kind: draft.kind, passage: draft.passages[0], record: draft.record },
        };
      case "agent_task":
        return {
          ...common,
          type,
          payload: {
            question: draft.question,
            answer: draft.answer,
            answer_type: draft.answer_type,
            document_ids: draft.document_ids,
          },
        };
    }
  }

  function save(event: FormEvent) {
    event.preventDefault();
    void action(async () => {
      const input = labelInput();
      const label = editing ? await updateLabel(editing, input) : await createLabel(input);
      setSaved(label.id);
      await refresh();
    });
  }

  function field(name: "question" | "answer" | "meeting_id" | "note") {
    return {
      value: draft[name],
      onChange: (event: { target: { value: string } }) =>
        setDraft({ ...draft, [name]: event.target.value }),
    };
  }

  const marked = draft.passages
    .filter((passage) => passage.document_id === documentId && passage.unit_index === index)
    .at(-1);
  const loading =
    busy || (!error && (!initial || (city && !currentCatalog) || (documentId && !currentSource)));
  const recordValues: Record<string, unknown> = { ...draft.record };
  const vote = "members" in draft.record ? draft.record : undefined;

  return (
    <>
      {loading && <p data-testid="loading">Loading...</p>}
      {error && <p data-testid="error">{error}</p>}
      {messages.length > 0 && (
        <ul data-testid="error-list">
          {messages.map((message, i) => (
            <li key={i}>{message}</li>
          ))}
        </ul>
      )}
      <div className="label-columns">
        <aside>
          <label>
            City
            <select
              data-testid="label-city"
              value={city}
              disabled={busy || Boolean(editing)}
              onChange={(event) => {
                setCity(event.target.value);
                setDocumentId("");
                setIndex(1);
                setAbsence(undefined);
                resetForm();
              }}
            >
              <option value="">Select city</option>
              {initial?.cities.map((entry) => (
                <option key={entry.id} value={entry.id}>
                  {entry.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Label type
            <select
              data-testid="label-type"
              value={type}
              disabled={busy || Boolean(editing)}
              onChange={(event) => {
                setType(event.target.value as LabelType);
                resetForm();
              }}
            >
              {Object.entries(LABEL_TYPES).map(([value, name]) => (
                <option key={value} value={value}>
                  {name}
                </option>
              ))}
            </select>
          </label>
          {initial && (
            <div data-testid="label-progress">
              <p>{initial.progress.frozen ? "frozen" : "not frozen"}</p>
              {Object.entries(initial.progress.types).map(([name, progress]) => (
                <div key={name}>
                  <p>
                    {name}: {progress.count} labels,{" "}
                    {initial.progress.human_reviewed[name as LabelType]} reviewed
                  </p>
                  {progress.violations.map((message, i) => (
                    <p key={i}>{message}</p>
                  ))}
                </div>
              ))}
            </div>
          )}
          <ul data-testid="label-list">
            {currentCatalog?.labels.map((label) => (
              <li key={label.id}>
                {label.id}
                <label>
                  Reviewed
                  <input
                    type="checkbox"
                    data-testid={`review-${label.id}`}
                    checked={label.human_reviewed}
                    disabled={busy}
                    onChange={(event) => {
                      const checked = event.target.checked;
                      void action(async () => {
                        await reviewLabel(label.id, checked);
                        await refresh();
                      });
                    }}
                  />
                </label>
                <button
                  data-testid={`edit-${label.id}`}
                  disabled={busy}
                  onClick={() => edit(label)}
                >
                  Edit
                </button>
                <button
                  data-testid={`delete-${label.id}`}
                  disabled={busy}
                  onClick={() =>
                    void action(async () => {
                      await deleteLabel(label.id);
                      if (editing === label.id) {
                        setEditing(undefined);
                        setDraft(emptyDraft());
                      }
                      await refresh();
                    })
                  }
                >
                  Delete
                </button>
              </li>
            ))}
          </ul>
          {type === "agenda_count" && (
            <ul data-testid="sample-meetings">
              {currentCatalog?.samples.map((meeting) => (
                <li key={meeting.meeting_id}>
                  <button
                    onClick={() => {
                      openDocument(meeting.agenda_document_id);
                      setDraft({ ...draft, meeting_id: meeting.meeting_id });
                    }}
                  >
                    {meeting.meeting_id}: {meeting.body}, {meeting.meeting_date}
                    {meeting.labelled ? " (labelled)" : ""}
                  </button>
                </li>
              ))}
            </ul>
          )}
          <button
            data-testid="label-export"
            disabled={busy}
            onClick={() =>
              void action(async () => {
                const result = await exportLabels(type);
                setExported(`Exported ${result.count} labels to ${result.path}`);
              })
            }
          >
            Export this type
          </button>
          {exported && <p>{exported}</p>}
        </aside>
        <section ref={browser}>
          <label>
            Document
            <select
              data-testid="label-document"
              value={documentId}
              onChange={(event) => openDocument(event.target.value)}
            >
              <option value="">Select document</option>
              {currentCatalog?.documents.map((doc) => (
                <option key={doc.id} value={doc.id}>
                  {doc.id}
                </option>
              ))}
            </select>
          </label>
          {currentSource && (
            <>
              <UnitControls
                unit={currentSource.unit}
                count={currentSource.doc.unit_count}
                onChange={(next) => {
                  setIndex(next);
                  setError(undefined);
                }}
              />
              <UnitViewer
                doc={currentSource.doc}
                unit={currentSource.unit}
                start={marked?.start}
                end={marked?.end}
              />
            </>
          )}
          <button data-testid="mark-passage" disabled={!currentSource} onClick={markPassage}>
            Mark passage
          </button>
          <label>
            Absence query
            <input
              data-testid="absence-query"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </label>
          <button
            data-testid="absence-search"
            disabled={!city || busy}
            onClick={() =>
              void action(async () => {
                const sent = query.trim();
                setAbsence({ query: sent, result: await textSearch(city, sent) });
              })
            }
          >
            Search source text
          </button>
          {absence && (
            <div>
              <p data-testid="absence-result">
                {absence.result.unit_count} units contain "{absence.query}"
              </p>
              <button
                data-testid="absence-add"
                onClick={() =>
                  setDraft({
                    ...draft,
                    absence_searches: [
                      ...draft.absence_searches,
                      { query: absence.query, hits: absence.result.unit_count },
                    ],
                  })
                }
              >
                Add absence search
              </button>
            </div>
          )}
        </section>
        <form data-testid="label-form" onSubmit={save}>
          {editing && <p>Editing {editing}</p>}
          <button type="button" onClick={resetForm}>
            New label
          </button>
          {(type === "question" || type === "agent_task") && (
            <label>
              Question
              <textarea data-testid="question" {...field("question")} />
            </label>
          )}
          {type === "question" && (
            <label>
              Answerable
              <input
                data-testid="answerable"
                type="checkbox"
                checked={draft.answerable}
                onChange={(event) => setDraft({ ...draft, answerable: event.target.checked })}
              />
            </label>
          )}
          {type === "question" && draft.answerable && (
            <label>
              Question type
              <select
                data-testid="question_type"
                value={draft.question_type}
                onChange={(event) =>
                  setDraft({ ...draft, question_type: event.target.value as QuestionType })
                }
              >
                {QUESTION_TYPES.map((value) => (
                  <option key={value}>{value}</option>
                ))}
              </select>
            </label>
          )}
          {(type === "agent_task" || (type === "question" && draft.answerable)) && (
            <>
              <label>
                Answer
                <textarea data-testid="label-answer" name="answer" {...field("answer")} />
              </label>
              <label>
                Answer type
                <select
                  data-testid="answer_type"
                  value={draft.answer_type}
                  onChange={(event) =>
                    setDraft({ ...draft, answer_type: event.target.value as AnswerType })
                  }
                >
                  {ANSWER_TYPES.map((value) => (
                    <option key={value}>{value}</option>
                  ))}
                </select>
              </label>
            </>
          )}
          {type === "question" && !draft.answerable && (
            <>
              <label>
                Reason
                <select
                  data-testid="reason"
                  value={draft.reason}
                  onChange={(event) => setDraft({ ...draft, reason: event.target.value as Reason })}
                >
                  {REASONS.map((value) => (
                    <option key={value}>{value}</option>
                  ))}
                </select>
              </label>
              <ul data-testid="absence-searches">
                {draft.absence_searches.map((entry, i) => (
                  <li key={i}>
                    {entry.query}: {entry.hits}
                    <button
                      type="button"
                      onClick={() =>
                        setDraft({
                          ...draft,
                          absence_searches: draft.absence_searches.filter(
                            (_, index) => index !== i,
                          ),
                        })
                      }
                    >
                      remove
                    </button>
                  </li>
                ))}
              </ul>
            </>
          )}
          {type === "agenda_count" && (
            <>
              <label>
                Meeting ID
                <input data-testid="meeting_id" {...field("meeting_id")} />
              </label>
              {draft.items.map((item, i) => (
                <fieldset key={i}>
                  {(["identifier", "title", "start_page"] as const).map((name) => (
                    <label key={name}>
                      {name}
                      <input
                        data-testid={name}
                        type={name === "start_page" ? "number" : "text"}
                        min={name === "start_page" ? 1 : undefined}
                        value={item[name]}
                        onChange={(event) =>
                          setDraft({
                            ...draft,
                            items: draft.items.map((entry, index) =>
                              index === i
                                ? {
                                    ...entry,
                                    [name]:
                                      name === "start_page"
                                        ? Number(event.target.value)
                                        : event.target.value,
                                  }
                                : entry,
                            ),
                          })
                        }
                      />
                    </label>
                  ))}
                  <button
                    type="button"
                    onClick={() =>
                      setDraft({ ...draft, items: draft.items.filter((_, index) => index !== i) })
                    }
                  >
                    remove
                  </button>
                </fieldset>
              ))}
              <button
                type="button"
                data-testid="add-item"
                onClick={() =>
                  setDraft({
                    ...draft,
                    items: [...draft.items, { identifier: "", title: "", start_page: index }],
                  })
                }
              >
                Add item
              </button>
              <label>
                Count
                <input data-testid="count" value={draft.items.length} readOnly />
              </label>
            </>
          )}
          {type === "extraction" && (
            <>
              <label>
                Kind
                <select
                  data-testid="kind"
                  value={draft.kind}
                  onChange={(event) => {
                    const kind = event.target.value as FactKind;
                    setDraft({ ...draft, kind, record: { ...RECORD_DEFAULTS[kind] } });
                  }}
                >
                  {Object.keys(RECORD_FIELDS).map((kind) => (
                    <option key={kind}>{kind}</option>
                  ))}
                </select>
              </label>
              {RECORD_FIELDS[draft.kind].map((entry) => (
                <label key={entry.name}>
                  {entry.name}
                  {entry.input === "select" ? (
                    <select
                      data-testid={entry.name}
                      value={String(recordValues[entry.name])}
                      onChange={(event) =>
                        setDraft({
                          ...draft,
                          record: { ...draft.record, [entry.name]: event.target.value },
                        })
                      }
                    >
                      {entry.options?.map((value) => (
                        <option key={value}>{value}</option>
                      ))}
                    </select>
                  ) : (
                    <input
                      data-testid={entry.name}
                      type={entry.input}
                      min={entry.input === "number" ? 0 : undefined}
                      step={entry.name === "amount_usd" ? "any" : undefined}
                      value={String(recordValues[entry.name] ?? "")}
                      onChange={(event) =>
                        setDraft({
                          ...draft,
                          record: {
                            ...draft.record,
                            [entry.name]:
                              entry.input === "number"
                                ? Number(event.target.value)
                                : entry.nullable && !event.target.value
                                  ? null
                                  : event.target.value,
                          },
                        })
                      }
                    />
                  )}
                </label>
              ))}
              {vote && (
                <>
                  {vote.members.map((member, i) => (
                    <fieldset key={i}>
                      <label>
                        Name
                        <input
                          data-testid="member-name"
                          value={member.name}
                          onChange={(event) => {
                            setDraft({
                              ...draft,
                              record: {
                                ...vote,
                                members: vote.members.map((entry, index) =>
                                  index === i ? { ...entry, name: event.target.value } : entry,
                                ),
                              },
                            });
                          }}
                        />
                      </label>
                      <label>
                        Value
                        <select
                          data-testid="member-value"
                          value={member.value}
                          onChange={(event) => {
                            setDraft({
                              ...draft,
                              record: {
                                ...vote,
                                members: vote.members.map((entry, index) =>
                                  index === i
                                    ? { ...entry, value: event.target.value as VoteMember["value"] }
                                    : entry,
                                ),
                              },
                            });
                          }}
                        >
                          {["aye", "no", "abstain", "absent"].map((value) => (
                            <option key={value}>{value}</option>
                          ))}
                        </select>
                      </label>
                      <button
                        type="button"
                        onClick={() => {
                          setDraft({
                            ...draft,
                            record: {
                              ...vote,
                              members: vote.members.filter((_, index) => index !== i),
                            },
                          });
                        }}
                      >
                        remove
                      </button>
                    </fieldset>
                  ))}
                  <button
                    type="button"
                    data-testid="add-member"
                    onClick={() => {
                      setDraft({
                        ...draft,
                        record: { ...vote, members: [...vote.members, { name: "", value: "aye" }] },
                      });
                    }}
                  >
                    Add member
                  </button>
                </>
              )}
            </>
          )}
          {(type === "extraction" || (type === "question" && draft.answerable)) && (
            <ul data-testid="passages">
              {draft.passages.map((passage, i) => (
                <li key={i}>
                  {passage.document_id} unit {passage.unit_index} [{passage.start}, {passage.end}):{" "}
                  {passage.text}
                  <button
                    type="button"
                    onClick={() =>
                      setDraft({
                        ...draft,
                        passages: draft.passages.filter((_, index) => index !== i),
                      })
                    }
                  >
                    remove
                  </button>
                </li>
              ))}
            </ul>
          )}
          {type === "agent_task" && (
            <>
              <ul data-testid="document-ids">
                {draft.document_ids.map((id, i) => (
                  <li key={i}>
                    {id}
                    <button
                      type="button"
                      onClick={() =>
                        setDraft({
                          ...draft,
                          document_ids: draft.document_ids.filter((_, index) => index !== i),
                        })
                      }
                    >
                      remove
                    </button>
                  </li>
                ))}
              </ul>
              <button
                type="button"
                data-testid="add-document"
                disabled={!currentSource}
                onClick={() =>
                  setDraft({ ...draft, document_ids: [...draft.document_ids, documentId] })
                }
              >
                Add document
              </button>
            </>
          )}
          <label>
            Note
            <textarea data-testid="note" maxLength={1000} {...field("note")} />
          </label>
          <p>Author: human</p>
          <button
            data-testid="label-save"
            disabled={
              !city || !currentCatalog || busy || (type === "extraction" && !draft.passages.length)
            }
          >
            Save
          </button>
          {saved && <p data-testid="saved">Saved {saved}</p>}
        </form>
      </div>
    </>
  );
}
