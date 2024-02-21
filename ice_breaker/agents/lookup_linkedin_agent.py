from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.agents import initialize_agent, Tool, AgentType

from tools.tools import get_profile_url


def lookup_linkedin(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    template = """
 		Given a full name {person_name}, I want you to look up for me their LinkedIn profile.
   		Your answer should contain only the LinkedIn URL, no additional text."""

    tools = [
        Tool(
            name="Search Google for the LinkedIn profile page",
            func=get_profile_url,
            description="useful for finding LinkedIn profiles URLs",
        )
    ]

    agent = initialize_agent(
        tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    prompt_template = PromptTemplate(template=template, input_variables=["person_name"])
    profile_url = agent.run(prompt_template.format_prompt(person_name="Lex Fridman"))
    return profile_url
