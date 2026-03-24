import sys
import subprocess
import os
import argparse


def run_step(script_path, description, *args):
    print(f"\n--- Running Step: {description} ---")

    cmd = [sys.executable, script_path] + list(args)

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="")

        process.wait()
        if process.returncode != 0:
            print(f"Error: {description} failed.")
            sys.exit(1)
        print(f"Success: {description} completed.")
    except Exception as e:
        print(f"Error executing {script_path}: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="MalariaGEN Taxon Classifier Pipeline")
    parser.add_argument('--k', type=int, default=15, help="K-mer size for the pipeline")
    parser.add_argument('--skip-extract', action='store_true', help="Skip feature extraction")
    parser.add_argument('--skip-train', action='store_true', help="Skip model training")
    args = parser.parse_args()

    k = str(args.k)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    extract_script = os.path.join(script_dir, "scripts", "extract_kmers.py")
    train_script = os.path.join(script_dir, "scripts", "train_models.py")
    eval_script = os.path.join(script_dir, "evaluation", "evaluate_models.py")

    print(f"Starting MalariaGEN Taxon Classifier Pipeline (k={k})...\n")

    if not args.skip_extract:
        run_step(extract_script, "Feature Extraction", "--k", k)
    else:
        print("Skipping feature extraction.")

    if not args.skip_train:
        run_step(train_script, "Model Training", "--k", k)
    else:
        print("Skipping model training.")

    run_step(eval_script, "Evaluation", "--k", k)

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
