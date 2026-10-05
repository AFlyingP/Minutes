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
