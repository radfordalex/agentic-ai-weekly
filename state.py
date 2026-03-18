from typing import TypedDict, Optional


class ResearchState(TypedDict):
    query: str
    mode: str
    date_range: Optional[str]
    sub_questions: Optional[str]
    research_findings: Optional[str]
    quality_assessment: Optional[str]
    quality_score: Optional[int]
    final_report: Optional[str]
    linkedin_draft: Optional[str]
    retry_count: Optional[int]
    unverified_items: Optional[str]
