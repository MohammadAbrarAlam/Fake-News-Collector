def create_agents(llm):
    from crewai import Agent

    return {
        "collector": Agent(
            role="Claim Analyst",
            goal="Identify checkable claims and separate them from opinions",
            backstory="You are a careful analyst who never treats a claim as proven without evidence.",
            llm=llm,
            verbose=False,
        ),
        "fact_checker": Agent(
            role="Evidence Reviewer",
            goal="Assess claims using only the supplied source material",
            backstory="You prioritize primary and reputable sources, preserve uncertainty, and never invent citations.",
            llm=llm,
            verbose=False,
        ),
        "reporter": Agent(
            role="Fact-check Reporter",
            goal="Return a concise, balanced fact-check with traceable evidence",
            backstory="You communicate findings clearly in the requested language and distinguish evidence from inference.",
            llm=llm,
            verbose=False,
        ),
    }