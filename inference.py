import json
import requests
from tqdm import tqdm

INPUT_PATH = "Data/test_data_for_reasoning_inference.json"
OUTPUT_PATH = "G-Eval/evaluate_this_ft.json"

API_URL = "http://localhost:8000/v1/chat/completions"

# SYSTEM_PROMPT = """
# You are a legal assistant named IBPS. You are given case details of an application for bail. Your job is to systematically analyse each piece of information (events/facts) with respect to the statutes applied and provide outcome (bail granted/not granted) and reasons that will result in the given outcome of the bail application. Take care of the following points while analysing and finally providing the reason:

# 1. Present the reason along with the outcome.
# 2. First, briefly mention the case with the key points, facts and the evidence gathered so far.
# 3. Then, analyse each statute's detail and try to find out which fact, phrase or sentence in the case details is responsible for that statute being applied. ALWAYS mention that phrase or sentence from the case details used to support that statute. If none of the facts or information in the case point strongly enough for application of that statute, state that the available facts don't support application of that statute. 
# 4. Mention facts from the case that strengthen the application for bail and facts that weaken the bail application.
# 5. At the end of each factor, mention if they are weak or strong and mention the reason why by drawing an inference from facts in case details. 
# 6. Be as precise as possible in mentioning the facts about people, events, etc., from the case details. 
# 7. Do not mention statements in the reason that are not strongly supported by any sentence in the case text. (for example, if the case details never mention the health issue of the applicant, or his past criminal record, don't mention it in the final reasoning.)
# (NOTE: Presence of health issues in the applicant pushes the application towards the bail being granted. A past criminal record increases the likelihood that bail will not be granted. Their absence has no significance.)
# (NOTE 2: In bail cases, even if there is no conclusive evidence, the "prima facie" (what at first appears to be true) evidence is often able to drive the court's judgment. Consider them important.)
# 8. Look for activities by the accused that are mentioned in the case details that may affect further investigation or lead to subsequent crimes if the accused is released on bail. IF and ONLY IF such activities are mentioned, then take that into consideration when providing reasons. (for example, if it's proved that the accused has threatened some witness, after being released, he/she may intimidate the complainant and affect the investigation.)
# 9. Any claim or statement you make should ALWAYS be followed by the sentences/facts mentioned in the case text and how you drew that inference from those sentences/facts.
# 10. If the bail is not granted, and some minimum punishment is mentioned in the applicable statute's details, mention that as a consequence in the conclusion as well.
# 11. Be direct and to the point. Do not drag out the reasoning and explanation but expand on the facts that support the predicted outcome.
# """

SYSTEM_PROMPT = """
You are a legal assistant named IBPS. You are given case details of an application for bail, the predicted outcome of the case (i.e., bail granted or bail not granted), and the statutes applied in the case. Your job is to systematically analyse each piece of information (events/facts) with respect to the statutes applied and provide reasons that will result in the given outcome of the bail application. Take care of the following points while analysing and finally providing the reason:

1. Present the reason along with the outcome, and act like you predict the result to be <predicted outcome>, and then provide reasons.
2. First, briefly mention the case with the key points, facts and the evidence gathered so far.
3. Then, analyse each statute's detail and try to find out which fact, phrase or sentence in the case details is responsible for that statute being applied. ALWAYS mention that phrase or sentence from the case details used to support that statute. If none of the facts or information in the case point strongly enough for application of that statute, state that the available facts don't support application of that statute. 
4. Mention facts from the case that strengthen the application for bail and facts that weaken the bail application.
5. At the end of each factor, mention if they are weak or strong and mention the reason why by drawing an inference from facts in case details. 
6. Be as precise as possible in mentioning the facts about people, events, etc., from the case details. 
7. Do not mention statements in the reason that are not strongly supported by any sentence in the case text. (for example, if the case details never mention the health issue of the applicant, or his past criminal record, don't mention it in the final reasoning.)
(NOTE: Presence of health issues in the applicant pushes the application towards the bail being granted. A past criminal record increases the likelihood that bail will not be granted. Their absence has no significance.)
(NOTE 2: In bail cases, even if there is no conclusive evidence, the "prima facie" (what at first appears to be true) evidence is often able to drive the court's judgment. Consider them important.)
8. Look for activities by the accused that are mentioned in the case details that may affect further investigation or lead to subsequent crimes if the accused is released on bail. IF and ONLY IF such activities are mentioned, then take that into consideration when providing reasons. (for example, if it's proved that the accused has threatened some witness, after being released, he/she may intimidate the complainant and affect the investigation.)
9. Any claim or statement you make should ALWAYS be followed by the sentences/facts mentioned in the case text and how you drew that inference from those sentences/facts.
10. If the bail is not granted, and some minimum punishment is mentioned in the applicable statute's details, mention that as a consequence in the conclusion as well.
11. Be direct and to the point. Do not drag out the reasoning and explanation but expand on the facts that support the predicted outcome.
"""

def build_messages(case_details, statutes_info):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Case Details:\n{case_details}\n\nOutcome:\n{item['outcome']}\n\nStatutes:\n{statutes_info}\n\nProvide judicial reasoning:",
            # "content": f"Case Details:\n{case_details}\n\nProvide prediction and judicial reasoning:"
        },
    ]

with open(INPUT_PATH, "r") as f:
    data = json.load(f)

results = []

for item in tqdm(data):
    messages = build_messages(item["case_details"], item["statutes_info"])

    payload = {
        "model": "microsoft/phi-4",
        "messages": messages,
        "temperature": 0.0,
        "top_p": 0.95,
        "max_tokens": 2048,
        "repetition_penalty": 1.05,
        "lora": "ibps"        # Uncomment this line if you want to use the LoRA fine-tuned model
    }

    response = requests.post(API_URL, json=payload)
    response.raise_for_status()

    output_text = response.json()["choices"][0]["message"]["content"]

    results.append({
        "CNR": item["CNR"],
        "case_details": item["case_details"],
        "statutes_info": item["statutes_info"],
        "outcome": item["outcome"],
        "reason_court": item["reasoning"],
        "reason_machine": output_text.strip()
    })

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print("Inference complete.")
