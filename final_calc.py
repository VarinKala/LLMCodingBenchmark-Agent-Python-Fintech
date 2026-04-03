import difflib
import json
import os
import re


def calculate_dws(reference, prediction):
    """
    Custom Domain-Weighted Similarity (DWS) 
    Calculates structural similarity and domain keyword accuracy.
    """
    if not prediction:
        return 0.0
        
    # 1. Structural Similarity (Gestalt Pattern Matching)
    struct_score = difflib.SequenceMatcher(None, reference, prediction).ratio()
    
    # 2. Fintech Keyword Weighting
    keywords = ["numpy", "pandas", "norm", "cdf", "rolling", "std", "mean", "np", "pd", "scipy"]
    
    ref_keys = set(re.findall(r'\w+', reference.lower()))
    pred_keys = set(re.findall(r'\w+', prediction.lower()))
    
    important_ref = [k for k in keywords if k in ref_keys]
    
    if not important_ref:
        keyword_score = struct_score 
    else:
        matches = sum(1 for k in important_ref if k in pred_keys)
        keyword_score = matches / len(important_ref)
    
    # Final Metric: 60% Structure, 40% Domain Accuracy
    return (0.6 * struct_score) + (0.4 * keyword_score)

def generate_report(benchmark_file, results_files_list):
    """
    Processes multiple result files and creates a consolidated JSON report.
    """
    with open(benchmark_file, "r") as f:
        bench_data = json.load(f)
        bench_dict = {t["id"]: t["reference_code"] for t in bench_data}

    consolidated_report = {}

    for file_path in results_files_list:
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: {file_path} not found. Skipping.")
            continue

        model_name = os.path.basename(file_path).replace(".json", "")
        print(f"📊 Processing {model_name}...")

        with open(file_path, "r") as f:
            results_data = json.load(f)

        total_pass_score = 0
        total_dws_score = 0
        task_count = len(results_data)

        for res in results_data:
            task_id = res["id"]
            ref_code = bench_dict.get(task_id)
            pred_code = res.get("sample_code", "")

            # Accumulate Pass@1
            total_pass_score += res.get("pass_at_1", 0)

            # Calculate and accumulate DWS
            if ref_code:
                dws = calculate_dws(ref_code, pred_code)
                total_dws_score += dws

        # Aggregates
        final_pass_at_1 = total_pass_score / task_count if task_count > 0 else 0
        avg_dws = total_dws_score / task_count if task_count > 0 else 0

        consolidated_report[model_name] = {
            "Pass@1": f"{final_pass_at_1:.2f}%",
            "Extended_CodeBleu_DWS": round(avg_dws, 4),
            "Metadata": {
                "Tasks_Count": task_count,
                "Source_File": file_path
            }
        }

    # Write consolidated report to JSON
    output_file = "final_benchmark_comparison.json"
    with open(output_file, "w") as f:
        json.dump(consolidated_report, f, indent=4)
    
    print(f"\n✅ Consolidated Report saved to: {output_file}")
    
    # Print Table for immediate review
    print("\n" + "="*60)
    print(f"{'Model Name':<35} | {'Pass@1':<10} | {'DWS Score'}")
    print("-" * 60)
    for model, stats in consolidated_report.items():
        print(f"{model:<35} | {stats['Pass@1']:<10} | {stats['Extended_CodeBleu_DWS']}")
    print("="*60)

if __name__ == "__main__":
    # List your result files here
    results_to_process = [
        "gemini_zero_shot_results.json",
        "llama_zero_shot_results.json",
        "qwen_zero_shot_results.json"
#        "agentic_mas_results.json"
    ]
    
    generate_report("fintech_benchmark.json", results_to_process)
