from langgraph.graph import StateGraph, END
from state import ResearchState
from agents import planner, researcher, quality_checker, report_writer, linkedin_drafter


def should_retry(state):
    score = state.get("quality_score", 0)
    retry_count = state.get("retry_count", 0)
    if score >= 7:
        return "report_writer"
    elif retry_count >= 3:
        return "force_report"
    else:
        return "researcher"


def add_unverified_items(state):
    quality = state.get("quality_assessment", "")
    flagged = []
    gaps = []
    current_section = None

    for line in quality.split("\n"):
        line_stripped = line.strip()

        if "FLAGGED" in line_stripped and ":" in line_stripped:
            current_section = "flagged"
            continue
        elif "GAPS" in line_stripped and ":" in line_stripped:
            current_section = "gaps"
            continue
        elif any(section in line_stripped for section in ["CONFIRMED", "CONTRADICTIONS", "SOURCE RATINGS", "QUALITY_SCORE"]):
            current_section = None
            continue

        if current_section == "flagged" and line_stripped and line_stripped.startswith("-"):
            flagged.append(line_stripped)
        elif current_section == "gaps" and line_stripped and line_stripped.startswith("-"):
            gaps.append(line_stripped)

    unverified = []
    if flagged:
        unverified.append("FLAGGED ITEMS:\n" + "\n".join(flagged))
    if gaps:
        unverified.append("RESEARCH GAPS:\n" + "\n".join(gaps))

    return {"unverified_items": "\n\n".join(unverified) if unverified else "Quality checker did not identify specific unverified items, but overall research quality was below threshold."}


workflow = StateGraph(ResearchState)
workflow.add_node("planner", planner)
workflow.add_node("researcher", researcher)
workflow.add_node("quality_checker", quality_checker)
workflow.add_node("force_report", add_unverified_items)
workflow.add_node("report_writer", report_writer)
workflow.add_node("linkedin_drafter", linkedin_drafter)
workflow.set_entry_point("planner")
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "quality_checker")
workflow.add_conditional_edges("quality_checker", should_retry)
workflow.add_edge("force_report", "report_writer")
workflow.add_edge("report_writer", "linkedin_drafter")
workflow.add_edge("linkedin_drafter", END)
app = workflow.compile()
