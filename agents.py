from langchain_openai import ChatOpenAI
from tools import search_tool, fetch_full_page
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_strong = ChatOpenAI(model="gpt-4o", temperature=0)


def planner(state):
    query = state["query"]
    mode = state["mode"]
    date_range = state.get("date_range", "past 7 days")

    mode_instructions = {
        "weekly": f"""Focus on NEW developments within the {date_range}.
        Frame sub-questions around: What launched? What was announced?
        What research was published? What changed in the ecosystem?
        Every sub-question should include time-specific language like
        'in the past week' or 'recently announced'.""",

        "monthly": f"""Summarize trends and major developments over the {date_range}.
        Frame sub-questions around: What were the biggest themes?
        What frameworks or tools gained traction? What shifted in adoption?
        Focus on patterns and trends, not individual announcements.""",

        "custom": """Treat this as a deep-dive research query.
        Frame sub-questions to cover the topic comprehensively
        from multiple angles."""
    }

    response = llm.invoke(
        f"""You are a Research Strategist specializing in AI and technology.

        MODE: {mode}
        {mode_instructions.get(mode, mode_instructions["custom"])}

        Analyze this research query and decompose it into 3-5 focused
        sub-questions that together would provide a comprehensive answer:

        QUERY: {query}

        For each sub-question provide:
        1. The sub-question itself
        2. Why this aspect matters
        3. Three SPECIFIC search queries optimized for finding high-quality sources.
           Format each search query on its own line starting with SEARCH:
           Example:
           SEARCH: CrewAI vs LangGraph benchmark comparison 2026
           SEARCH: agentic AI production deployment case study
           SEARCH: multi-agent orchestration arxiv paper 2025 2026

        CRITICAL: Your search queries must target SPECIFIC, CONCRETE information:
        - Name specific frameworks, companies, tools, and versions
        - Ask for numbers: adoption rates, GitHub stars, benchmark results, funding amounts
        - Target primary sources: arXiv papers, GitHub repos, company engineering blogs, official docs
        - Avoid queries that would return generic marketing content or definitions
        - Include year (2025 or 2026) in at least one search query per sub-question

        Format as a numbered list."""
    )
    return {"sub_questions": response.content}


def researcher(state):
    sub_questions = state["sub_questions"]
    mode = state["mode"]
    date_range = state.get("date_range", "past 7 days")
    query = state["query"]
    quality_feedback = state.get("quality_assessment", None)
    retry_count = state.get("retry_count", 0)

    retry_instructions = ""
    if quality_feedback and retry_count > 0:
        retry_instructions = f"""
        IMPORTANT: This is retry attempt {retry_count + 1}. The previous research
        was rated poorly. Here is what the Quality Checker flagged:

        {quality_feedback}

        You MUST address these issues:
        - Search for DIFFERENT sources than before. Do not repeat the same URLs.
        - If claims were flagged as unsupported, find primary sources that confirm or deny them.
        - If gaps were identified, focus your searches on filling those gaps.
        - Prioritize: arXiv papers, official documentation, GitHub repositories,
          company engineering blogs, and peer-reviewed content over generic blog posts.
        """

    all_findings = []

    current_subq = ""
    search_queries = []

    for line in sub_questions.split("\n"):
        stripped = line.strip()

        if stripped and stripped[0].isdigit() and ("Sub-question" in stripped or "**" in stripped):
            if current_subq and search_queries:
                findings = _run_searches(current_subq, search_queries, query, mode, date_range, retry_count)
                all_findings.append(findings)
            current_subq = stripped
            search_queries = []
        elif stripped.startswith("SEARCH:"):
            search_queries.append(stripped.replace("SEARCH:", "").strip())

    if current_subq and search_queries:
        findings = _run_searches(current_subq, search_queries, query, mode, date_range, retry_count)
        all_findings.append(findings)

    if not all_findings:
        for line in sub_questions.split("\n"):
            if not (line.strip() and line.strip()[0].isdigit()):
                continue

            search_query = line.strip()[:200]
            if mode in ["weekly", "monthly"]:
                search_query = f"{search_query} {date_range}"

            results = search_tool.invoke(search_query)

            fetched_content = []
            urls_fetched = []
            if isinstance(results, list):
                for result in results[:10]:
                    url = result.get("url", "")
                    if url and url not in urls_fetched:
                        content = fetch_full_page(url, query)
                        fetched_content.append(f"Source: {url}\nContent:\n{content}")
                        urls_fetched.append(url)

            all_findings.append(
                f"Sub-question: {line.strip()}\n"
                f"Search snippets: {results}\n"
                f"Full page content:\n" + "\n---\n".join(fetched_content)
            )

    findings_text = "\n\n===\n\n".join(all_findings)

    response = llm.invoke(
        f"""You are an Information Retrieval Specialist focused on AI and
        technology research. You prioritize primary sources and authoritative
        publications.

        {retry_instructions}

        Organize these raw search results and full-page content into
        structured findings. For each sub-question provide:
        - Source name and URL
        - Key findings with SPECIFIC facts, numbers, dates - not vague summaries
        - Publication date if available
        - Confidence level (high/medium/low) based on source quality

        CRITICAL RULES:
        - NEVER fabricate URLs. Only report URLs from the search results.
        - NEVER invent statistics or claims not in the source material.
        - If information is thin for a sub-question, say so explicitly.
        - Prioritize CONCRETE data: version numbers, GitHub stars, funding rounds,
          adoption numbers, benchmark scores, specific company names and use cases.
        - Generic statements like "AI is transforming industries" are WORTHLESS.
          Only include claims backed by specific evidence from the sources.

        RAW RESULTS:
        {findings_text}"""
    )
    return {"research_findings": response.content}


def _run_searches(sub_question, search_queries, query, mode, date_range, retry_count):
    """Run multiple targeted searches for a single sub-question."""
    all_results = []
    fetched_content = []
    urls_fetched = []

    for sq in search_queries:
        if mode in ["weekly", "monthly"]:
            sq = f"{sq} {date_range}"

        results = search_tool.invoke(sq)
        if isinstance(results, list):
            all_results.extend(results)

    if retry_count > 0:
        for sq in search_queries[:2]:
            targeted = search_tool.invoke(f"{sq} arxiv OR github OR official documentation")
            if isinstance(targeted, list):
                all_results.extend(targeted)

    seen_urls = set()
    unique_results = []
    for result in all_results:
        url = result.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)

    for result in unique_results[:10]:
        url = result.get("url", "")
        if url and url not in urls_fetched:
            content = fetch_full_page(url, query)
            fetched_content.append(f"Source: {url}\nContent:\n{content}")
            urls_fetched.append(url)

    return (
        f"Sub-question: {sub_question}\n"
        f"Search queries used: {search_queries}\n"
        f"Total unique sources found: {len(unique_results)}\n"
        f"Search snippets: {all_results}\n"
        f"Full page content:\n" + "\n---\n".join(fetched_content)
    )


def quality_checker(state):
    findings = state["research_findings"]
    retry_count = state.get("retry_count", 0)
    query = state["query"]

    response = llm_strong.invoke(
        f"""You are a senior Fact-Checker and Source Evaluator. You are
        skeptical by nature and hold research to high standards.

        ORIGINAL QUERY: {query}

        Review these research findings and provide:

        CONFIRMED: Findings that are well-supported by authoritative sources.
        List each confirmed finding on its own line starting with a dash.

        FLAGGED: Claims that need verification or appear unsupported.
        List each flagged item on its own line starting with a dash.

        CONTRADICTIONS: Conflicting information between sources.
        List each contradiction on its own line starting with a dash.

        SOURCE RATINGS: Rate each source 1-5 with justification.
        List each rating on its own line starting with a dash.

        GAPS: Important aspects not adequately covered.
        List each gap on its own line starting with a dash.

        QUALITY SCORE: Rate the overall research quality 1-10.
        
        Calibrate your score based on what is ACHIEVABLE for the query type:
        
        For OVERVIEW queries (e.g. "What is X", "current state of X"):
        - Authoritative corporate sources (IBM, Google, Microsoft docs) ARE 
          appropriate and should score well
        - Industry analyst reports (Gartner, Forrester, Deloitte) are high quality
        - A mix of 5+ credible corporate/analyst sources covering the topic 
          comprehensively deserves a 7-8
        - Score 9-10 if it also includes academic sources and specific metrics
        
        For TECHNICAL queries (e.g. "compare frameworks", "benchmark results"):
        - arXiv papers, GitHub repos, and engineering blogs are expected
        - Corporate marketing pages should score lower
        - Specific benchmarks, version numbers, and code examples are needed for 7+
        
        For NEWS queries (e.g. "this week's developments"):
        - Recency matters most - sources should be from the specified time period
        - News outlets, official announcements, and release notes are appropriate
        - Coverage of multiple developments deserves 7+

        You MUST end your response with exactly this format on its own line:
        QUALITY_SCORE: [number]

        This is retry attempt {retry_count + 1}. If this is attempt 2 or higher,
        be especially critical about whether the new research actually improved
        on previous gaps.

        FINDINGS TO REVIEW:
        {findings}"""
    )

    content = response.content
    score = 5
    for line in content.split("\n"):
        if "QUALITY_SCORE:" in line:
            try:
                score = int(line.split(":")[-1].strip())
            except ValueError:
                score = 5

    return {
        "quality_assessment": content,
        "quality_score": score,
        "retry_count": retry_count + 1
    }


def report_writer(state):
    findings = state["research_findings"]
    quality = state["quality_assessment"]
    query = state["query"]
    mode = state["mode"]
    unverified = state.get("unverified_items", "")

    mode_instructions = {
        "weekly": """Structure as a weekly briefing. Lead with the biggest
        development, then cover other notable updates. Keep it scannable
        with clear section headers.""",

        "monthly": """Structure as a monthly trend report. Identify 2-3
        major themes, then detail developments under each theme. Include
        a 'Looking Ahead' section.""",

        "custom": """Structure based on the natural organization of the
        research findings. Use clear sections that address each major
        aspect of the query."""
    }

    response = llm.invoke(
        f"""You are a Research Report Writer specializing in AI and technology.

        {mode_instructions.get(mode, mode_instructions["custom"])}

        Requirements:
        - Clear title and executive summary (2-3 sentences)
        - Organized sections with specific findings, not vague summaries
        - Inline citations linking to source URLs
        - A Limitations and Uncertainties section using the quality assessment
        - If there are unverified items, include them clearly: {unverified}
        - A Sources section at the end with all referenced URLs
        - Write for a technical audience but keep it accessible
        - Include specific numbers, dates, and facts wherever available
        - NO BUZZWORDS. No "transformative", "revolutionary", "game-changing".
          Let the specific facts speak for themselves.

        ORIGINAL QUERY: {query}
        RESEARCH FINDINGS: {findings}
        QUALITY ASSESSMENT: {quality}"""
    )
    return {"final_report": response.content}


def linkedin_drafter(state):
    report = state["final_report"]
    mode = state["mode"]
    query = state["query"]

    mode_instructions = {
        "weekly": """Create a LinkedIn post highlighting this week's most
        important developments. Pull the 4-5 most significant specific 
        facts from the report.""",

        "monthly": """Create a LinkedIn post summarizing this month's key
        trends. Pull the 4-5 most significant specific facts from the report.""",

        "custom": """Create a LinkedIn post about this topic for a mixed
        audience of AI engineers and non-technical business professionals.
        Pull the 5-6 most significant specific facts from the report."""
    }

    response = llm.invoke(
        f"""You are extracting the most important facts from a research 
        report and formatting them as a LinkedIn post.

        {mode_instructions.get(mode, mode_instructions["custom"])}

        STRUCTURE (follow this exactly):

        [One sentence hook: the single most surprising or important fact 
        from the report, with a specific number or name]

        [2-3 sentence paragraph explaining what agentic AI actually is. 
        Use a simple analogy a non-technical person would understand. 
        Compare it to something familiar.]

        Key findings from this research:

        [Fact 1 - pulled directly from report with source name]
        [Fact 2 - pulled directly from report with source name]  
        [Fact 3 - pulled directly from report with source name]
        [Fact 4 - pulled directly from report with source name]
        [Fact 5 - pulled directly from report with source name if available]

        [PERSONAL_INSERT: Add 1-2 sentences about your hands-on experience with this topic]

        [One sentence closing question to drive engagement]

        STRICT RULES:
        - No hashtags
        - No emojis
        - BANNED PHRASES: "fast-paced", "transformative", "revolutionary",
          "game-changing", "cutting-edge", "in today's world", "it's no secret",
          "I'm excited to share", "buckle up", "the future is here",
          "not just a buzzword", "paradigm shift", "as we stand on the brink",
          "as we embrace", "in today's business environment", "imagine a world",
          "did you know"
        - Every fact MUST include a specific company name, framework name, 
          number, percentage, or date
        - Every fact MUST name its source in parentheses
        - If a fact does not have a concrete detail, DO NOT include it
        - Keep total post under 250 words
        - Write in first person as an AI/ML engineer

        REPORT TO EXTRACT FROM:
        {report}"""
    )
    return {"linkedin_draft": response.content}
