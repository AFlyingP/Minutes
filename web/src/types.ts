export interface City {
  id: string;
  name: string;
  state: string;
  bodies: string[];
  date_min: string | null;
  date_max: string | null;
  documents: number;
}

export interface SpanRef {
  document_id: string;
  unit_index: number;
  unit_kind: "page" | "segment";
  start_offset: number;
  end_offset: number;
  start_ms: number | null;
  label: string;
}

export interface Hit {
  rank: number;
  score: number;
  document_id: string;
  meeting_id: string;
  city_id: string;
  city_name: string;
  body: string;
  meeting_date: string;
  doc_kind: "agenda" | "minutes" | "transcript";
  item_identifier: string | null;
  item_title: string | null;
  heading: string;
  snippet: string;
  spans: SpanRef[];
}

export interface SearchResponse {
  query: string;
  mode: string;
  chunker: string;
  hits: Hit[];
}

export type DocKind = "agenda" | "minutes" | "transcript";
export type Box = [number, number, number, number, number, number];

export interface DocumentSummary {
  id: string;
  meeting_id: string;
  city_id: string;
  kind: DocKind;
  body: string;
  meeting_date: string;
  unit_kind: "page" | "segment";
  unit_count: number | null;
}

export interface DocumentsResponse {
  total: number;
  documents: DocumentSummary[];
}

export interface DocumentDetail {
  id: string;
  city_id: string;
  kind: DocKind;
  unit_kind: "page" | "segment";
  unit_count: number | null;
  source_url: string;
  meeting: {
    id: string;
    body: string;
    meeting_date: string;
    title: string;
    recording_url: string | null;
  };
}

interface UnitBase {
  document_id: string;
  unit_index: number;
  text: string;
  text_source: "pdf" | "ocr" | "caption";
  speaker: string | null;
  label: string;
  recording_link: string | null;
}

export type Unit = UnitBase &
  (
    | {
        unit_kind: "page";
        boxes: Box[] | null;
        start_ms: null;
        end_ms: null;
        width_pt: number | null;
        height_pt: number | null;
      }
    | {
        unit_kind: "segment";
        boxes: null;
        start_ms: number;
        end_ms: number;
        width_pt: null;
        height_pt: null;
      }
  );

export type LabelType = "question" | "agenda_count" | "extraction" | "agent_task";
export type QuestionType = "vote" | "amount" | "legislation" | "person" | "date" | "multi_passage";
export type AnswerType = "number" | "name" | "date" | "identifier" | "free_text";
export type Reason =
  "entity_not_in_corpus" | "date_outside_corpus" | "fact_not_recorded" | "false_premise";
export type FactKind = "motion" | "vote" | "ordinance" | "amount" | "statement";

export interface Passage {
  document_id: string;
  unit_index: number;
  start: number;
  end: number;
  text: string;
}

export interface AbsenceSearch {
  query: string;
  hits: number;
}

export interface AgendaItem {
  identifier: string;
  title: string;
  start_page: number;
}

export interface VoteMember {
  name: string;
  value: "aye" | "no" | "abstain" | "absent";
}

export interface Motion {
  text: string;
  mover: string | null;
  seconder: string | null;
  outcome: "passed" | "failed" | "withdrawn" | "tabled" | "unknown";
}

export interface Vote {
  subject_identifier: string | null;
  ayes: number;
  noes: number;
  abstain: number;
  absent: number;
  members: VoteMember[];
  outcome: "passed" | "failed";
}

export interface Ordinance {
  identifier: string;
  legislation_type: "ordinance" | "resolution" | "council_bill" | "other";
  title: string;
  status: "introduced" | "passed" | "adopted" | "failed" | "referred" | "held" | "unknown";
}

export interface Amount {
  amount_usd: number;
  purpose: string;
  payee: string | null;
}

export interface Statement {
  speaker: string;
  role: string | null;
  summary: string;
}

export type FactRecord = Motion | Vote | Ordinance | Amount | Statement;

export type LabelData =
  | {
      type: "question";
      payload:
        | {
            question: string;
            answerable: true;
            question_type: QuestionType;
            answer: string;
            answer_type: AnswerType;
            passages: Passage[];
          }
        | {
            question: string;
            answerable: false;
            reason: Reason;
            absence_searches: AbsenceSearch[];
          };
    }
  | { type: "agenda_count"; payload: { meeting_id: string; count: number; items: AgendaItem[] } }
  | { type: "extraction"; payload: { kind: FactKind; passage: Passage; record: FactRecord } }
  | {
      type: "agent_task";
      payload: {
        question: string;
        answer: string;
        answer_type: AnswerType;
        document_ids: string[];
      };
    };

export type LabelIn = LabelData & {
  city: string;
  author: "agent" | "human";
  note: string;
};

export type Label = LabelIn & {
  id: string;
  created_utc: string;
  human_reviewed: boolean;
};

export interface LabelProgress {
  frozen: boolean;
  types: Record<LabelType, { count: number; violations: string[] }>;
  human_reviewed: Record<LabelType, number>;
}

export interface TextSearchResponse {
  unit_count: number;
  matches: { document_id: string; unit_index: number; snippet: string }[];
}

export interface SampleMeeting {
  meeting_id: string;
  body: string;
  meeting_date: string;
  agenda_document_id: string;
  labelled: boolean;
}

export interface LabelExportResponse {
  path: string;
  count: number;
}
