import os

def run_step(title, script_name):
    print(f"\n🔧 {title}")
    os.system(f"python3 {script_name}.py")

def main():
    print("🚀 Starting Resume-Job Matcher Pipeline")

    run_step("Step 1: Preprocessing data", "preprocess")
    run_step("Step 2: Generating embeddings", "embedder")
    run_step("Step 3: Matching resumes to jobs", "matcher")

    print("\n✅ Pipeline complete. Check 'data/match_results.csv' for results.")

if __name__ == "__main__":
    main()
