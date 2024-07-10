import os
import json
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFacePipeline

from agents.lookup_linkedin_agent import lookup_linkedin
from third_parties.linkedin import scrape_linkedin_profile
from output_parser import PersonIntel, person_intel_parser

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

def simple_summary() -> PersonIntel:
    summary_prompt_template = create_prompt_template()

    openai_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    # hf_chain = LLMChain(prompt=summary_prompt_template, llm=llm, verbose=True)
    openai_chain = LLMChain(llm=openai_llm, prompt=summary_prompt_template)

    linkedin_profile_url = "mock"
    linkedin_data = scrape_linkedin_profile(url=linkedin_profile_url)
    res = openai_chain.invoke(input={"information": linkedin_data})
    print(res)
    # parsed_res = person_intel_parser.parse(res)
    return res

def hf_inference_client():
    summary_template = """
	given the following information {information} about a person I want you to create:
	1. a short summary
	2. two interesting facts about them
	"""
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )
    
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    llm_client = InferenceClient(model=repo_id, timeout=120)
    
    linkedin_profile_url = "mock"
    linkedin_data = scrape_linkedin_profile(url=linkedin_profile_url)
    
    summary_prompt = summary_template.format(
        information=linkedin_data
    )
    
    response = llm_client.post(
        json={
            "inputs": summary_prompt,
            "parameters": {"max_new_tokens": 500},
            "task": "text-generation",
        },
    )

    response_text = json.loads(response.decode())[0]["generated_text"]
    print(response_text)
    
def summary_ollama():
    summary_prompt_template = create_prompt_template()
    llama_llm = ChatOllama(model="llama3")
    
    chain = summary_prompt_template | llama_llm
    
    linkedin_profile_url = "mock"
    linkedin_data = scrape_linkedin_profile(url=linkedin_profile_url)
    
    res = chain.invoke(input={"information": linkedin_data})
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
    # simple_summary()
    # print(hf_inference_client())
    summary_ollama()
