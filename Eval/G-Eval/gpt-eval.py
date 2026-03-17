from openai import OpenAI
import json
import argparse
import tqdm
import time
import statistics

def parse_response(text):
    try:
        parts = [x.strip() for x in text.split(",")]
        fa = float(parts[0])
        c1 = float(parts[1])
        c2 = float(parts[2])
        flag = int(parts[3])
        return fa, c1, c2, flag
    except:
        return None


if __name__ == '__main__':
    with open('./G-Eval/API_Key.txt', 'r') as file:
        API_Key = file.readline().strip()
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--prompt', type=str, default='./G-Eval/eval_prompt.txt')
    argparser.add_argument('--save_to', type=str, default='./G-Eval/results.json') # <----- change this accordingly
    argparser.add_argument('--input', type=str, default='./G-Eval/evaluate_this.json') # <----- change this accordingly
    argparser.add_argument('--key', type=str, default=API_Key)
    argparser.add_argument('--model', type=str, default='gpt-4.1')
    args = argparser.parse_args()
    
    client = OpenAI(api_key=args.key)
    system_prompt = open('./G-Eval/system_prompt.txt').read()

    input_cases = json.load(open(args.input))
    user_prompt = open(args.prompt).read()

    ct, ignore = 0, 0

    new_json = []
    for instance in tqdm.tqdm(input_cases):
        full_case = instance['case_details'] + '\n\n' + instance['statutes_info']
        source = instance['reason_court']
        system_output = instance['reason_machine']
        cur_prompt = user_prompt.replace('{{full_case}}', full_case).replace('{{Actual_reason}}', source).replace('{{Generated_reason}}', system_output)
        instance['prompt'] = cur_prompt
        while True:
            try:
                _response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": cur_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=15,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    # logprobs=40,
                    n=3
                )
                time.sleep(0.5)

                all_responses = [choice.message.content for choice in _response.choices]

                parsed = []
                for resp in all_responses:
                    result = parse_response(resp)
                    if result is not None:
                        parsed.append(result)

                if len(parsed) == 0:
                    ignore += 1
                    break

                fa_scores = [p[0] for p in parsed]
                c1_scores = [p[1] for p in parsed]
                c2_scores = [p[2] for p in parsed]
                flags = [p[3] for p in parsed]

                avg_fa = sum(fa_scores) / len(fa_scores)
                avg_c1 = sum(c1_scores) / len(c1_scores)
                avg_c2 = sum(c2_scores) / len(c2_scores)

                # majority vote for flag
                flag_final = max(set(flags), key=flags.count)

                instance['all_responses'] = all_responses
                instance['avg_FA'] = avg_fa
                instance['avg_C1'] = avg_c1
                instance['avg_C2'] = avg_c2
                instance['std_FA'] = statistics.stdev(fa_scores) if len(fa_scores) > 1 else 0
                instance['std_C1'] = statistics.stdev(c1_scores) if len(c1_scores) > 1 else 0
                instance['std_C2'] = statistics.stdev(c2_scores) if len(c2_scores) > 1 else 0
                instance['final_Flag'] = flag_final
                                
                new_json.append(instance)
                ct += 1
                break
            except Exception as e:
                print(e)
                if ("limit" in str(e)):
                    time.sleep(2)
                else:
                    ignore += 1
                    print('ignored', ignore)

                    break

    print('ignored total', ignore)
    with open(args.save_to, 'w') as f:
        json.dump(new_json, f, indent=4)
