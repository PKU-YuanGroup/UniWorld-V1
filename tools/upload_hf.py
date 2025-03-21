import os
from huggingface_hub import HfApi, upload_file

REPO_NAME = "LanguageBind/Cambrian737k"
LOCAL_FOLDER = "/storage/lb/dataset/Cambrian737k"

success_flag = False
while not success_flag:
    try:
        api = HfApi()
        existing_files = set(api.list_repo_files(REPO_NAME, repo_type="dataset"))
        for root, _, files in os.walk(LOCAL_FOLDER):
            for file in files:
                if (not file.endswith('.tar')) and (not file.endswith('.json')):
                    continue
                local_path = os.path.join(root, file)
                remote_path = os.path.relpath(local_path, LOCAL_FOLDER)

                if remote_path in existing_files:
                    print(f"Skipping {remote_path}, already exists.")
                    continue

                print(f"Uploading <{local_path}> to <{remote_path}>...")
                upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=remote_path,
                    repo_id=REPO_NAME,
                    repo_type="dataset",
                )
        success_flag = True
    except Exception as e:
        print(e)
        success_flag = False
print("Upload completed.")
