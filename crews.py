def run_fact_check(case_data, api_key):
    from crewai import Crew, LLM, Process

    from agents import create_agents
    from tasks import create_tasks

    llm = LLM(
        model="gemini/gemini-2.5-flash",
        api_key=api_key,
        temperature=0.1,
    )
    agents = create_agents(llm)
    tasks = create_tasks(agents)
    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )
    return str(crew.kickoff(inputs={"case_data": case_data}))