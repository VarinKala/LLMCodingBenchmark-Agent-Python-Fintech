import json
import os
import time

from openai import OpenAI

# Setup NIM Client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY")
)

# Models to evaluate (NIM Free Tier options)
MODELS = ["meta/llama-3.1-70b-instruct", "qwen/qwen2.5-coder-32b-instruct"]

def run_zero_shot_baseline(benchmark_file):
    with open(benchmark_file, "r") as f:
        tasks = json.load(f)

    all_results = {}

    for model_name in MODELS:
        print(f"🧬 Evaluating Model: {model_name}")
        model_results = []
        
        for task in tasks:
            print(f"  Task {task['id']}: {task['task']}")
            success_count = 0
            
            # PROJECT REQUIREMENT: Prompt at least 10 times for Pass@1
            for i in range(10):
                try:
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": f"Write a Python function for: {task['prompt']}. Respond with ONLY the code block."}],
                        temperature=0.7 # Standard for Pass@k
                    )
                    code = completion.choices[0].message.content.replace("```python", "").replace("```", "").strip()
                    
                    # Verify execution
                    try:
                        exec(code, {"np": __import__("numpy"), "pd": __import__("pandas"), "norm": __import__("scipy.stats").stats.norm})
                        success_count += 1
                    except:
                        pass
                except Exception as e:
                    print(f"    ⚠️ API Error: {e}")
                    time.sleep(5)
                
                time.sleep(1) # Rate limit safety

            pass_at_1 = (success_count / 10) * 100
            model_results.append({
                "id": task['id'],
                "pass_at_1": pass_at_1,
                "sample_code": code # Store last attempt for CodeBLUE
            })
            print(f"    📊 Pass@1: {pass_at_1}%")

        all_results[model_name] = model_results

    with open("nim_zero_shot_results_2.json", "w") as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    run_zero_shot_baseline("fintech_benchmark.json")
