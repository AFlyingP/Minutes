import type { City, SearchResponse } from "./types";

export type Params = Record<string, string | undefined>;

export class ApiError extends Error {
  code: string;
  status: number;

  constructor(code: string, status: number) {
    super(`Request failed: ${code}`);
    this.code = code;
    this.status = status;
  }
}

async function errorCode(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body.error === "string") return body.error;
  } catch {
    // the body is not JSON, so the status is all there is to report
  }
  return String(response.status);
}

export async function getJson<T>(path: string, params?: Params): Promise<T> {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params ?? {})) {
    if (value) query.set(key, value);
  }
  const text = query.toString();
  const response = await fetch(text ? `${path}?${text}` : path);
  if (!response.ok) throw new ApiError(await errorCode(response), response.status);
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
