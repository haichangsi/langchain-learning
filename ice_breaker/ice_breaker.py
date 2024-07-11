import os
import json
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFacePipeline

from agents.lookup_linkedin_agent import lookup_linkedin
from third_parties.linkedin import scrape_linkedin_profile
from output_parser import PersonIntel, person_intel_parser


def ice_breaker(model: str, name: str = "mock"):
    summary_prompt_template = create_prompt_template()

    if name == "mock":
        profile_url = "mock"
    else:
        profile_url = lookup_linkedin(name)

    linkedin_data = scrape_linkedin_profile(url=profile_url)

    model_functions = {
        "gpt": gpt_summary,
        "hf_inference": hf_inference_client,
        "ollama": ollama_summary,
    }

    if model not in model_functions:
        model = "gpt"

    result = model_functions[model](summary_prompt_template, linkedin_data)
    print(result)


def create_prompt_template():
    summary_template = """
	given the following information {information} about a person I want you to create:
	1. a short summary
	2. two interesting facts about them
    3. a topic that may interest the person
    4. 2 creative ice breakers to open a conversation with the person
                \n{format_instructions}
	"""
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
        partial_variables={
            "format_instructions": person_intel_parser.get_format_instructions()
        },
    )

    return summary_prompt_template


def gpt_summary(prompt_template, data) -> PersonIntel:
    openai_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = prompt_template | openai_llm | StrOutputParser()
    res = chain.invoke(input={"information": data})
    print(res)
    # parsed_res = person_intel_parser.parse(res)
    return res


def hf_inference_client(prompt_template, data):

    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    llm_client = InferenceClient(model=repo_id, timeout=120)

    summary_prompt = prompt_template.format(information=data)

    response = llm_client.post(
        json={
            "inputs": summary_prompt,
            "parameters": {"max_new_tokens": 500},
            "task": "text-generation",
        },
    )

    response_text = json.loads(response.decode())[0]["generated_text"]
    print(response_text)


def ollama_summary(prompt_template, data, model="llama3"):
    # Ollama supports multiple models like Llama3 or Mistral
    llama_llm = ChatOllama(model=model)

    chain = prompt_template | llama_llm | StrOutputParser

    res = chain.invoke(input={"information": data})
    print(res)


# def hf_summary_from_model_id():
#     llm = HuggingFacePipeline.from_model_id(
#     model_id="microsoft/Phi-3-mini-4k-instruct",
#     task="text-generation",
#     pipeline_kwargs={
#         "max_new_tokens": 100,
#         "top_k": 50,
#         "temperature": 0.1,
#     },
#     )
#     return llm.invoke("Short story")


if __name__ == "__main__":
    load_dotenv()
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HFH_API_TOKEN")
    ice_breaker("gpt", "mock")
    # ice_breaker("hf_inference", "mock")
    # ice_breaker("ollama", "mock")
