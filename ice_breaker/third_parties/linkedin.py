import os
import requests

def scrape_linkedin_profile(url: str):
	"""
	Scrapes a LinkedIn profile.
	Manually scrapes the information from the LinkedIn profile.
	"""
	api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
	header_dict = {"Authorization": f"Bearer {os.getenv('PROXYCURL_TOKEN')}"}
	params = {
		'linkedin_profile_url': 'https://www.linkedin.com/in/lexfridman/',
		'extra': 'include',
		'github_profile_id': 'include',
		'facebook_profile_id': 'include',
		'twitter_profile_id': 'include',
		'personal_contact_number': 'include',
		'personal_email': 'include',
		'inferred_salary': 'include',
		'skills': 'include',
		'use_cache': 'if-present',
		'fallback_to_cache': 'on-error',
	}	
	response = requests.get(api_endpoint, params=params, headers=header_dict)
	response = filter_response(response)
	return response

def get_sample_linkedin_profile():
	"""
	Uses a predownloaded JSON LinkedIn profile.
	"""
	lf_gist = "https://gist.githubusercontent.com/haichangsi/8c5dd133aaeb56afc5c4f0fff5b72626/raw/a529da2dc0e0b874a5d3526de033edab8a1629ff/lex-friedman-langchain"
	response = requests.get(lf_gist)
	response = filter_response(response)
	return response

def filter_response(response):
	"""
	Filter out the response to only include the relevant information.
	"""
	data = response.json()

	data = {
		k:v for k,v in data.items() if v not in([], "", "", None)
		and k not in ["people_also_viewed", "certifications"]
	}
	if data.get("groups"):
		for group_dict in data.get("groups"):
			group_dict.pop("profile_pic_url")
   
	return data

