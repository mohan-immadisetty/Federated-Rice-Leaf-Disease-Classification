"""Create the Hugging Face Space and upload `dist/space` in one step.

Requires a Hugging Face token with **write** access:

    hf auth login          # choose a Write token (or a fine-grained token with
                           # "Repositories: create/write" + "Spaces" permission)

Then:

    python scripts/build_space.py
    python scripts/push_space.py

If the token is read-only you will get a 403 with instructions.
"""
from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = ROOT / "dist" / "space"
REPO_NAME = "rice-e5"


def main() -> None:
    if not SPACE_DIR.exists():
        sys.exit("dist/space does not exist. Run `python scripts/build_space.py` first.")

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception as exc:
        sys.exit(f"Not logged in to Hugging Face ({type(exc).__name__}). Run: hf auth login")

    repo_id = f"{user}/{REPO_NAME}"
    print(f"user: {user}\nspace: {repo_id}")

    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
            exist_ok=True,
        )
    except HfHubHTTPError as exc:
        if exc.response is not None and exc.response.status_code == 403:
            sys.exit(
                "\nYour Hugging Face token has no write access.\n"
                "Create a Write token at https://huggingface.co/settings/tokens\n"
                "then run:  hf auth login\n"
                "and retry: python scripts/push_space.py"
            )
        raise

    print("uploading dist/space ...")
    api.upload_folder(
        folder_path=str(SPACE_DIR),
        repo_id=repo_id,
        repo_type="space",
        commit_message="E5 inference API: FastAPI + EfficientNetV2-S + DINOv2",
    )
    print(f"done: https://huggingface.co/spaces/{repo_id}")
    print(f"API:  https://{user.lower()}-{REPO_NAME}.hf.space")


if __name__ == "__main__":
    main()
