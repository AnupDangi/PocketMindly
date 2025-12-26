from huggingface_hub import list_repo_files

repos_to_check = [
    "onnx-community/silero-vad",
    "silero/silero-vad",
    "snakers4/silero-vad",
    "speechbrain/vad-crdnn-libriparty"
]

for repo in repos_to_check:
    print(f"Checking {repo}...")
    try:
        files = list_repo_files(repo_id=repo)
        print(f"Found {len(files)} files. Top 10:")
        for f in files[:10]:
            print(f" - {f}")
    except Exception as e:
        print(f"Error accessing {repo}: {e}")
    print("-" * 20)
