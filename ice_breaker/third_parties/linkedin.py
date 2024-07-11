import os
import requests


def scrape_linkedin_profile(url: str, mock: bool = True):
    """
    Scrapes a LinkedIn profile.
    Manually scrapes the information from the LinkedIn profile.
    """

    if mock:
        lf_gist = "https://gist.githubusercontent.com/emarco177/0d6a3f93dd06634d95e46a2782ed7490/raw/fad4d7a87e3e934ad52ba2a968bad9eb45128665/eden-marco.json"
        response = requests.get(lf_gist, timeout=10)
    else:
        api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
        header_dict = {"Authorization": f"Bearer {os.getenv('PROXYCURL_TOKEN')}"}
        params = {
            "linkedin_profile_url": url,
            # "extra": "include",
            # "github_profile_id": "include",
            # "facebook_profile_id": "include",
            # "twitter_profile_id": "include",
            # "personal_contact_number": "include",
            # "personal_email": "include",
            # "inferred_salary": "include",
            # "skills": "include",
            # "use_cache": "if-present",
            # "fallback_to_cache": "on-error",
        }
        response = requests.get(
            api_endpoint, params=params, headers=header_dict, timeout=10
        )
    response = filter_response(response)
    return response


def filter_response(response):
    """
    Filter out the response to only include the relevant information.
    """
    data = response.json()

    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None)
        and k not in ["people_also_viewed", "certifications"]
    }
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url", None)

    return data


# print(scrape_linkedin_profile("https://www.linkedin.com/in/lexfridman/"))
