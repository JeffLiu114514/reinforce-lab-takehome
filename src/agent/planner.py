from __future__ import annotations

from typing import List, Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .prompts import PLANNER_USER, SYSTEM_JSON_ONLY
from .schemas import ClaimType
from .utils import get_llm


class QuerySpec(BaseModel):
    q: str
    claim_types: List[ClaimType]
    provider_hint: Literal["academic", "industry", "general"] = "general"


class PlanOutput(BaseModel):
    angles: List[str] = Field(min_length=4)
    queries: List[QuerySpec]
    constraints: List[str] = Field(
        default_factory=lambda: ["data quality", "bias", "evaluation"]
    )


def plan_research(prompt: str) -> PlanOutput:
    llm = get_llm()
    prompt_tmpl = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_JSON_ONLY),
            ("user", PLANNER_USER),
        ]
    )
    messages = prompt_tmpl.format_messages(prompt=prompt)
    structured = llm.with_structured_output(PlanOutput)
    result = structured.invoke(messages)
    required = {"data quality", "bias", "evaluation"}
    if not required.issubset(set(map(str.lower, result.constraints))):
        result.constraints = sorted(required)
    return result

