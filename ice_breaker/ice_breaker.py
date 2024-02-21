from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain import HuggingFacePipeline
import os

import torch
import transformers
from transformers import AutoTokenizer
from agents.lookup_linkedin_agent import lookup_linkedin

from third_parties.linkedin import scrape_linkedin_profile, get_sample_linkedin_profile
from output_parser import PersonIntel, person_intel_parser

# def hf_pipeline_helper():

# hf_model = "HuggingFaceM4/tiny-random-LlamaForCausalLM"
# tokenizer = AutoTokenizer.from_pretrained(hf_model)

# pipeline = transformers.pipeline(
# 	"text-generation",
# 	model=hf_model,
# 	tokenizer=tokenizer,
# 	torch_dtype=torch.bfloat16,
# 	trust_remote_code=True,
# 	device_map="auto",
# 	max_length=3000,
# 	do_sample=True,
# 	top_k=10,
# 	num_return_sequences=1,
# 	eos_token_id=tokenizer.eos_token_id
# )
# hf_llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature':0})
# hf_llm = VertexAI()


def simple_summary() -> PersonIntel:
    summary_template = """
	given the following information {information} about a person I want you to create:
	1. a short summary of the person
	2. two interesting facts about the person
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
    # TO DO: download a model from hfh/ use huggingchat api/
    repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.2, "max_length": 8192}
    )

    openai_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    hf_chain = LLMChain(prompt=summary_prompt_template, llm=llm, verbose=True)
    openai_chain = LLMChain(llm=openai_llm, prompt=summary_prompt_template)

    linkedin_profile_url = lookup_linkedin("Lex Fridman")
    linkedin_data = scrape_linkedin_profile(url=linkedin_profile_url)
    # linkedin_data = get_sample_linkedin_profile()
    # res = hf_chain.run(information=linkedin_data)
    # print(res)
    res = openai_chain.run(information=linkedin_data)
    parsed_res = person_intel_parser.parse(res)
    return parsed_res
    print(res)


if __name__ == "__main__":
    load_dotenv()
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HFH_API_TOKEN")
    simple_summary()
