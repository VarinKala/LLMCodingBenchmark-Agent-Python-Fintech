import json
import os
import subprocess
import time

import google.generativeai as genai
from openai import OpenAI


class FintechMultiAgentSystem:
    def __init__(self):
        # 1. Orchestrator: Gemini 2.0 Flash
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.orchestrator = genai.GenerativeModel('gemini-2.0-flash')
        
        # 2. NIM Clients: Qwen (Critic) & Llama (Programmer)
        # Note: Roles are swapped to allow for MAS improvement observation
        self.nim_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ.get("NVIDIA_API_KEY")
        )
        self.critic_model = "qwen/qwen2.5-coder-32b-instruct"
        self.programmer_model = "meta/llama-3.1-70b-instruct"

    def safe_nim_call(self, model, messages, retries=6):
        """NIM Free Tier Hardened Wrapper with exponential backoff."""
        delay = 10 
        for i in range(retries):
            try:
                response = self.nim_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    timeout=120 # High timeout for Free Tier congestion
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"  ⚠️ NIM API Error ({model}): {e}. Retry {i+1}/{retries} in {delay}s...")
                time.sleep(delay)
                delay *= 2 
        return None

    def executor_agent(self, code):
        """Act/Observe step: Local environment code execution."""
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

    def critic_agent(self, code, error_logs, original_task):
        """
        The Judge Agent: Analyzes failure against original requirements.
        This follows the 'Agent-as-a-Judge' framework (arXiv:2408.08927).
        """
        prompt = f"""
        ROLE: Financial Code Judge
        ORIGINAL REQUIREMENT: {original_task}
        DEVELOPER'S ATTEMPT:
        {code}
        
        EXECUTION ERROR:
        {error_logs}
        
        INSTRUCTION: Identify the mathematical or logical mismatch between the code 
        and the requirement. Provide a clear, 1-sentence correction for the developer.
        """
        return self.safe_nim_call(self.critic_model, [{"role": "user", "content": prompt}])

    def run_react_iteration(self, task):
        """The ReAct trajectory: Act -> Observe -> Reason (Judge) -> Re-Act."""
        feedback = None
        current_code = ""
        
        for attempt in range(3): # Maximum 3 correction attempts per iteration
            # 1. ACT: Programmer generates code
            prog_prompt = 'Avoid writing comments for this coding task.\n' + task['prompt']
            if feedback:
                prog_prompt += f"\n\nJUDGE FEEDBACK: {feedback}\nCorrect the logic based on this feedback."
            
            res = self.safe_nim_call(self.programmer_model, [{"role": "user", "content": prog_prompt}])
            if not res: continue
            
            # Clean Markdown formatting
            current_code = res.replace("```python", "").replace("```", "").strip()
            
            # 2. OBSERVE: Executor runs code
            success, logs = self.executor_agent(current_code)
            if success:
                return True, current_code
            
            # 3. REASON: Judge (Critic) evaluates the failure
            feedback = self.critic_agent(current_code, logs, task['prompt'])
            print(f"    💡 Judge Advice (Attempt {attempt+1}): {feedback}")
            
            time.sleep(2) # Buffer between agent handoffs
            
        return False, current_code

    def run_full_benchmark(self, benchmark_file):
        """Executes 10x Pass@1 benchmark across all tasks."""
        with open(benchmark_file, "r") as f:
            tasks = json.load(f)

        final_results = []
        for task in tasks:
            print(f"🚀 Benchmarking Task {task['id']}: {task['task']}")
            success_count = 0
            last_code_sample = ""
            
            # 10x loop to calculate statistically valid Pass@1
            for i in range(10):
                passed, code = self.run_react_iteration(task)
                if passed: 
                    success_count += 1
                last_code_sample = code
                time.sleep(2) # Mandatory Free Tier cool-down

            pass_at_1 = (success_count / 10) * 100
            print(f"  📊 Final Pass@1 for {task['id']}: {pass_at_1}%")
            
            final_results.append({
                "id": task['id'],
                "pass_at_1": pass_at_1,
                "sample_code": last_code_sample
            })

        # Final save of Agentic Results
        with open("agentic_mas_results.json", "w") as f:
            json.dump(final_results, f, indent=4)
        print("\n✅ Benchmark Complete. Results saved to agentic_mas_results.json")

if __name__ == "__main__":
    # Ensure environment variables are set before running
    # NVIDIA_API_KEY and GOOGLE_API_KEY
    mas_system = FintechMultiAgentSystem()
    mas_system.run_full_benchmark("fintech_benchmark.json")
