import os
import json
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFacePipeline

from ice_breaker.agents.lookup_linkedin_agent import lookup_linkedin
from ice_breaker.third_parties.linkedin import scrape_linkedin_profile
from ice_breaker.output_parser import PersonIntel, person_intel_parser

from typing import Tuple


def ice_breaker_main(model: str, name: str = "mock") -> Tuple[PersonIntel, str]:
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
    return result, linkedin_data.get("profile_pic_url")


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

    chain = prompt_template | openai_llm | person_intel_parser
    res: PersonIntel = chain.invoke(input={"information": data})

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

    return response_text


def ollama_summary(prompt_template, data, model="llama3") -> PersonIntel:
    # Ollama supports multiple models like Llama3 or Mistral
    llama_llm = ChatOllama(model=model)

    chain = prompt_template | llama_llm | person_intel_parser

    res: PersonIntel = chain.invoke(input={"information": data})

    return res


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
    ice_breaker_main("gpt", "mock")
    # ice_breaker_main("hf_inference", "mock")
    # ice_breaker_main("ollama", "mock")
