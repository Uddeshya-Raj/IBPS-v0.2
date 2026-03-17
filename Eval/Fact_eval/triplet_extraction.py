import json
from openai import OpenAI
from tqdm import tqdm

with open("API_Key.txt") as f:
    api_key = f.read().strip()

with open("triplet_extraction_prompt.txt") as f:
    SYSTEM_PROMPT = f.read()

client = OpenAI(api_key=api_key)

def extract_triplets(text):

    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ]
    )

    content = response.choices[0].message.content

    try:
        return json.loads(content)
    except:
        return {
            "resolved_text": "",
            "entities": [],
            "triplets": []
        }


def process_cases(input_file, output_file):

    with open(input_file) as f:
        data = json.load(f)

    results = []

    for case in tqdm(data):

        case_result = case.copy()

        for key in ["reason_court", "reason_ft", "reason_baseline"]:

            if key in case and case[key]:

                extraction = extract_triplets(case[key])

                case_result[key + "_kg"] = extraction

        results.append(case_result)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


process_cases("cases.json", "cases_with_triplets.json")