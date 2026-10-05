import type {
  City,
  DocumentDetail,
  DocumentsResponse,
  Label,
  LabelExportResponse,
  LabelIn,
  LabelProgress,
  LabelType,
  SampleMeeting,
  SearchResponse,
  TextSearchResponse,
  Unit,
} from "./types";

export type Params = Record<string, string | undefined>;

export class ApiError extends Error {
  code: string;
  status: number;
  messages: string[];

  constructor(code: string, status: number, messages: string[] = []) {
    super(`Request failed: ${code}`);
    this.code = code;
    this.status = status;
    this.messages = messages;
  }
}

async function responseError(response: Response): Promise<ApiError> {
  try {
    const body = await response.json();
    const code = typeof body.error === "string" ? body.error : String(response.status);
    const messages =
      (response.status === 422 || response.status === 409) && Array.isArray(body.detail)
        ? body.detail.filter((message: unknown): message is string => typeof message === "string")
        : [];
    return new ApiError(code, response.status, messages);
  } catch {
    // the body is not JSON, so the status is all there is to report
  }
  return new ApiError(String(response.status), response.status);
}

export async function getJson<T>(path: string, params?: Params): Promise<T> {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params ?? {})) {
    if (value) query.set(key, value);
  }
  const text = query.toString();
  const response = await fetch(text ? `${path}?${text}` : path);
  if (!response.ok) throw await responseError(response);
  return (await response.json()) as T;
}

export async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw await responseError(response);
  return (await response.json()) as T;
}

export async function putJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw await responseError(response);
  return (await response.json()) as T;
}

export async function fetchCities(): Promise<City[]> {
  return (await getJson<{ cities: City[] }>("/api/cities")).cities;
}

export function search(q: string, filters: Params): Promise<SearchResponse> {
  return getJson<SearchResponse>("/api/search", {
    q,
    ...filters,
    mode: "keyword",
    chunker: "fixed",
    k: "10",
  });
}

export function fetchDocuments(params: Params = {}): Promise<DocumentsResponse> {
  return getJson<DocumentsResponse>("/api/documents", params);
}

export function fetchDocument(id: string): Promise<DocumentDetail> {
  return getJson<DocumentDetail>(`/api/documents/${encodeURIComponent(id)}`);
}

export function fetchUnit(id: string, index: number): Promise<Unit> {
  return getJson<Unit>(`/api/documents/${encodeURIComponent(id)}/units/${index}`);
}

export function unitImageUrl(id: string, index: number): string {
  return `/api/documents/${encodeURIComponent(id)}/units/${index}/image?dpi=110`;
}

export async function fetchLabels(type: LabelType, city: string): Promise<Label[]> {
  return (await getJson<{ labels: Label[] }>("/api/labels", { type, city })).labels;
}

export function createLabel(label: LabelIn): Promise<Label> {
  return postJson<Label>("/api/labels", label);
}

export function updateLabel(id: string, label: LabelIn): Promise<Label> {
  return putJson<Label>(`/api/labels/${encodeURIComponent(id)}`, label);
}

export async function deleteLabel(id: string): Promise<void> {
  const response = await fetch(`/api/labels/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!response.ok) throw await responseError(response);
}

export function reviewLabel(id: string, human_reviewed: boolean): Promise<Label> {
  return postJson<Label>(`/api/labels/${encodeURIComponent(id)}/review`, { human_reviewed });
}

export function fetchProgress(): Promise<LabelProgress> {
  return getJson<LabelProgress>("/api/labels/progress");
}

export function textSearch(city: string, q: string): Promise<TextSearchResponse> {
  return getJson<TextSearchResponse>("/api/labels/text-search", { city, q });
}

export async function fetchSampleMeetings(city: string): Promise<SampleMeeting[]> {
  return (await getJson<{ meetings: SampleMeeting[] }>("/api/labels/sample-meetings", { city }))
    .meetings;
}

export function exportLabels(type: LabelType): Promise<LabelExportResponse> {
  return postJson<LabelExportResponse>("/api/labels/export", { type });
}
