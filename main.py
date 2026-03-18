from graph import app
from dotenv import load_dotenv
import os
import sys
from datetime import datetime

load_dotenv()


def get_versioned_path(base_path):
    """If file exists, add _v2, _v3, etc. Never overwrite."""
    if not os.path.exists(base_path):
        return base_path

    name, ext = os.path.splitext(base_path)
    version = 2
    while os.path.exists(f"{name}_v{version}{ext}"):
        version += 1
    return f"{name}_v{version}{ext}"


def run(mode="custom", query=None):
    default_queries = {
        "weekly": "What are the most significant developments in agentic AI this week? Include new framework releases, major research papers, product launches, and industry adoption news.",
        "monthly": "What were the major trends and developments in agentic AI this month? Identify key themes, framework updates, notable research, and shifts in industry adoption.",
    }

    if not query:
        query = default_queries.get(mode, "What is agentic AI and what is the current state of production agentic AI systems?")

    date_ranges = {
        "weekly": "past 7 days",
        "monthly": "past 30 days",
        "custom": "no date restriction"
    }

    inputs = {
        "query": query,
        "mode": mode,
        "date_range": date_ranges.get(mode, "no date restriction"),
        "sub_questions": None,
        "research_findings": None,
        "quality_assessment": None,
        "quality_score": None,
        "final_report": None,
        "linkedin_draft": None,
        "retry_count": 0,
        "unverified_items": None,
    }

    print(f"\n{'='*60}")
    print(f"Agentic AI Research Assistant")
    print(f"Mode: {mode}")
    print(f"Query: {query}")
    print(f"Date range: {date_ranges.get(mode)}")
    print(f"{'='*60}\n")

    accumulated_state = {}
    for step in app.stream(inputs):
        node_name = list(step.keys())[0]
        node_output = step[node_name]
        accumulated_state.update(node_output)

        print(f"\n{'='*60}")
        print(f"Completed: {node_name}")

        if "quality_score" in node_output and node_output["quality_score"]:
            score = node_output["quality_score"]
            retry = node_output.get("retry_count", 0)
            print(f"Quality Score: {score}/10 (Attempt {retry})")
            if score < 7 and retry < 3:
                print("Score below 7 -- looping back to researcher...")
            elif score < 7 and retry >= 3:
                print("Retry limit reached -- proceeding with unverified items noted.")

        for key, value in node_output.items():
            if value and key not in ["quality_score", "retry_count"]:
                preview = str(value)[:500]
                print(f"\n{key}:\n{preview}...")

        print(f"{'='*60}")

    os.makedirs("reports", exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")

    if accumulated_state.get("final_report"):
        report_path = get_versioned_path(f"reports/{date_str}_{mode}_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(accumulated_state["final_report"])
        print(f"\nReport saved to {report_path}")

    if accumulated_state.get("linkedin_draft"):
        draft_path = get_versioned_path(f"reports/{date_str}_{mode}_linkedin_draft.md")
        with open(draft_path, "w", encoding="utf-8") as f:
            f.write(accumulated_state["linkedin_draft"])
        print(f"LinkedIn draft saved to {draft_path}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "custom"
    query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None
    run(mode=mode, query=query)
