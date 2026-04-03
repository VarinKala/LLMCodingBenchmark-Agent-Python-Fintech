import json
import os
import time

import google.generativeai as genai
from google.api_core import exceptions

# Setup Gemini
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

def run_gemini_10x_baseline(benchmark_file):
    with open(benchmark_file, "r") as f:
        tasks = json.load(f)

    results = []

    for task in tasks:
        print(f"🚀 Gemini Baseline: Task {task['id']}")
        success_count = 0
        
        for i in range(10):
            try:
                # Temperature 0.7 allows for the 'variance' needed for Pass@k
                prompt = f"Write a Python function for: {task['prompt']}. Respond with ONLY the code block."
                response = model.generate_content(prompt, generation_config={"temperature": 0.7})
                code = response.text.replace("```python", "").replace("```", "").strip()
                
                # Execution check
                try:
                    exec_env = {
                        "np": __import__("numpy"), 
                        "pd": __import__("pandas"), 
                        "norm": __import__("scipy.stats").stats.norm
                    }
                    exec(code, exec_env)
                    success_count += 1
                except:
                    pass
            except exceptions.ResourceExhausted:
                print("  ⚠️ Rate limit! Sleeping 30s...")
                time.sleep(30)
            except Exception as e:
                print(f"  ❌ Error: {e}")
            
            time.sleep(2) # Standard cool-down

        pass_at_1 = (success_count / 10) * 100
        print(f"  📊 Task {task['id']} Pass@1: {pass_at_1}%")
        results.append({
            "id": task['id'],
            "pass_at_1": pass_at_1,
            "sample_code": code
        })

    with open("gemini_zero_shot_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_gemini_10x_baseline("fintech_benchmark.json")
