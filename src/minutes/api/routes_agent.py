from dataclasses import asdict
from typing import Annotated, Literal, cast

import psycopg
from fastapi import APIRouter, Depends
from psycopg.rows import TupleRow

from minutes.agent.agent import MAX_STEPS, run_agent
from minutes.api.routes_core import get_conn
from minutes.api.schemas import AgentCitationOut, AgentIn, AgentOut, AgentStepOut
from minutes.retrieval.search import span_label

router = APIRouter(prefix="/api")
Conn = Annotated[psycopg.Connection[TupleRow], Depends(get_conn)]


@router.post("/agent", response_model=AgentOut)
def agent_endpoint(conn: Conn, request: AgentIn) -> AgentOut:
    result = run_agent(conn, request.question)
    labels: dict[tuple[str, int], str] = {}
    if result.citations:
        keys = [
            (str(citation["document_id"]), int(cast(int, citation["unit_index"])))
            for citation in result.citations
        ]
        rows = conn.execute(
            "SELECT u.document_id, u.unit_index, u.unit_kind, u.start_ms "
            "FROM units u JOIN unnest(%s::text[], %s::int[]) "
            "AS cited(document_id, unit_index) USING (document_id, unit_index)",
            ([key[0] for key in keys], [key[1] for key in keys]),
        ).fetchall()
        labels = {
            (document_id, unit_index): span_label(unit_kind, unit_index, start_ms)
            for document_id, unit_index, unit_kind, start_ms in rows
        }
    citations = [
        AgentCitationOut(
            document_id=str(citation["document_id"]),
            unit_index=int(cast(int, citation["unit_index"])),
            label=labels.get(
                (str(citation["document_id"]), int(cast(int, citation["unit_index"]))),
                f"p. {citation['unit_index']}",
            ),
        )
        for citation in result.citations
    ]
    return AgentOut(
        status=result.status,
        answer=result.answer,
        citations=citations,
        steps=[AgentStepOut(**asdict(step)) for step in result.steps],
        cost_usd=result.cost_usd,
        limit=cast_limit(result.limit),
        max_steps=MAX_STEPS,
    )


def cast_limit(value: str | None) -> Literal["steps", "cost", "time"] | None:
    if value in ("steps", "cost", "time"):
        return cast(Literal["steps", "cost", "time"], value)
    return None
