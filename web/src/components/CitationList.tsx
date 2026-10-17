import { Link } from "react-router-dom";

import type { Citation } from "../types";

export default function CitationList({
  citations,
  cityNames,
}: {
  citations: Citation[];
  cityNames: Record<string, string>;
}) {
  return (
    <ol data-testid="citations">
      {citations.map((citation) => (
        <li key={citation.number}>
          <Link
            data-testid={`citation-${citation.number}`}
            to={`/doc/${citation.document_id}?unit=${citation.unit_index}&start=${citation.start_offset}&end=${citation.end_offset}`}
          >
            [{citation.number}] {cityNames[citation.city_id]} {citation.body},{" "}
            {citation.meeting_date}, {citation.doc_kind}, {citation.label}
          </Link>
        </li>
      ))}
    </ol>
  );
}
