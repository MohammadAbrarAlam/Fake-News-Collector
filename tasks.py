def create_tasks(agents):
    from crewai import Task

    review = Task(
        description=(
            "Review the supplied claims and search results. Use only the evidence provided. "
            "For every claim, state whether evidence supports it, contradicts it, or is insufficient. "
            "Retain source URLs exactly as supplied and explain uncertainty. Input: {case_data}"
        ),
        expected_output="A claim-by-claim evidence assessment with source URLs.",
        agent=agents["fact_checker"],
    )
    report = Task(
        description=(
            "Turn the preceding evidence assessment into valid JSON in the requested language. "
            "Return an object with a summary and claims array. Each claim must have claim, verdict, "
            "confidence (0-100), explanation, and evidence (array of title, url, source, snippet). "
            "Use only supplied URLs; do not invent facts or citations."
        ),
        expected_output="Valid JSON containing a cautious, source-linked fact-check report.",
        agent=agents["reporter"],
        context=[review],
    )
    return [review, report]