import json
import os
import subprocess
import time

import google.generativeai as genai
from openai import OpenAI


class FintechMultiAgentSystem:
    def __init__(self):
        # Clients
        self.nim_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ.get("NVIDIA_API_KEY")
        )
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.orchestrator = genai.GenerativeModel('gemini-2.0-flash')
        
        # Model Selection
        self.critic_model = "qwen/qwen2.5-coder-32b-instruct"
        self.programmer_model = "meta/llama-3.1-70b-instruct"

    def safe_nim_call(self, model, messages, retries=4):
        """Standardized wrapper for Free Tier stability"""
        delay = 5
        for i in range(retries):
            try:
                response = self.nim_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    timeout=60 # Extended timeout for slower free-tier queues
                )
                return response.choices[0].message.content
            except Exception as e:
                # Catching Rate Limits (429) and Server Overload (503/504)
                print(f"  ⚠️ NIM API Error: {e}. Retry {i+1}/{retries} in {delay}s...")
                time.sleep(delay)
                delay *= 2 # Exponential backoff
        return None

    def executor_agent(self, code):
        """Act/Observe step with local environment isolation"""
        with open("temp_agent_code.py", "w") as f:
            f.write(code)
        try:
            # Captures both output and errors for the Critic
            result = subprocess.run(
                ["python3", "temp_agent_code.py"], 
                capture_output=True, text=True, timeout=5
            )
            return (True, "SUCCESS") if result.returncode == 0 else (False, result.stderr)
        except subprocess.TimeoutExpired:
            return False, "Error: Execution timed out (potential infinite loop)."
        except Exception as e:
            return False, str(e)

    def run_react_iteration(self, task_prompt):
        """Multi-Agent ReAct trajectory"""
        feedback = None
        current_code = ""
        
        for attempt in range(3): # ReAct allows up to 3 self-corrections
            prog_prompt = task_prompt
            if feedback:
                prog_prompt += f"\n\nCRITIC FEEDBACK: {feedback}\nFix the code based on this feedback."
            
            # Programmer ACTS
            res = self.safe_nim_call(self.programmer_model, [{"role": "user", "content": prog_prompt}])
            if not res: continue
            
            current_code = res.replace("```python", "").replace("```", "").strip()
            
            # Executor OBSERVES
            success, logs = self.executor_agent(current_code)
            if success:
                return True, current_code
            
            # Critic REASONS
            critic_prompt = f"CODE:\n{current_code}\n\nERROR:\n{logs}\nProvide a 1-sentence fix."
            feedback = self.safe_nim_call(self.critic_model, [{"role": "user", "content": critic_prompt}])
            
            time.sleep(2) # Buffer between agent handoffs
            
        return False, current_code

    def run_full_benchmark(self, benchmark_file):
        with open(benchmark_file, "r") as f:
            tasks = json.load(f)

        final_results = []
        for task in tasks:
            print(f"🚀 Processing Task {task['id']}")
            success_count = 0
            
            # 10x loop for Pass@1 calculation
            for i in range(10):
                passed, code = self.run_react_iteration(task['prompt'])
                if passed: success_count += 1
                time.sleep(2) # Mandatory Free Tier cool-down

            pass_at_1 = (success_count / 10) * 100
            print(f"  📊 Agentic Pass@1: {pass_at_1}%")
            final_results.append({"id": task['id'], "pass_at_1": pass_at_1})

        with open("agentic_mas_results.json", "w") as f:
            json.dump(final_results, f, indent=4)

if __name__ == "__main__":
    mas_system = FintechMultiAgentSystem()
    mas_system.run_full_benchmark("fintech_benchmark.json")
