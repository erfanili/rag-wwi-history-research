import os
import requests
from argparse import ArgumentParser
from dotenv import load_dotenv
from huggingface_hub import HfApi, upload_folder, list_repo_files

# --- CONFIG YOU DON'T TOUCH PER RUN ---
REPO_ID = "Erfanili/RAG-wwi-history"   # fixed
LOCAL_ROOT = "data"                    # everything lives under this
# --------------------------------------

load_dotenv()


def _token(t=None):
    return t or os.getenv("HF_API_KEY")


def _ensure_under_root(p: str) -> str:
    """Return absolute path and its repo-relative path under LOCAL_ROOT; error if not under root."""
    abs_p = os.path.abspath(p)
    abs_root = os.path.abspath(LOCAL_ROOT)
    if not abs_p.startswith(abs_root + os.sep) and abs_p != abs_root:
        raise ValueError(f"Path must be inside {LOCAL_ROOT}/. Got: {p}")
    rel = os.path.relpath(abs_p, abs_root).replace("\\", "/")
    rel = "" if rel == "." else rel
    return abs_p, rel


def upload_file_sync(local_path: str, token=None):
    abs_p, rel = _ensure_under_root(local_path)
    if not os.path.isfile(abs_p):
        raise FileNotFoundError(abs_p)
    api = HfApi(token=_token(token))
    api.upload_file(
        path_or_fileobj=abs_p,
        path_in_repo=rel,             # mirror under repo
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"Uploaded file: {rel or '/'}")


def upload_folder_sync(local_folder: str, token=None):
    abs_p, rel = _ensure_under_root(local_folder)
    if not os.path.isdir(abs_p):
        raise NotADirectoryError(abs_p)
    upload_folder(
        folder_path=abs_p,
        repo_id=REPO_ID,
        repo_type="dataset",
        path_in_repo=rel or "",       # mirror under repo
        token=_token(token),
    )
    print(f"Uploaded folder: {rel or '/'}")


def download_file_sync(repo_rel_path: str, token=None):
    # save to LOCAL_ROOT/repo_rel_path
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{repo_rel_path}"
    out_path = os.path.join(LOCAL_ROOT, repo_rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    headers = {"Authorization": f"Bearer {_token(token)}"} if _token(token) else {}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    print(f"Downloaded file: {repo_rel_path} -> {out_path}")


def download_folder_sync(repo_rel_folder: str, token=None):
    repo_rel_folder = repo_rel_folder.strip("/")

    files = list_repo_files(REPO_ID, repo_type="dataset", token=_token(token))
    wanted = [f for f in files if f.startswith(repo_rel_folder + "/")] if repo_rel_folder else files
    if not wanted:
        print(f"No files found under '{repo_rel_folder}'")
        return
    for rel in wanted:
        download_file_sync(rel, token=token)


if __name__ == "__main__":
    parser = ArgumentParser(description="Mirror files/folders between local data/ and HF dataset with identical paths.")
    parser.add_argument("--action",
                        choices=["upload-file", "upload-folder", "download-file", "download-folder"],
                        required=True)
    parser.add_argument("--path", required=True,
                        help="For uploads: local path inside data/. For downloads: repo-relative path (saved into data/).")
    parser.add_argument("--token", help="HF token (defaults to HF_API_KEY).")
    args = parser.parse_args()

    if args.action == "upload-file":
        upload_file_sync(args.path, token=args.token)
    elif args.action == "upload-folder":
        upload_folder_sync(args.path, token=args.token)
    elif args.action == "download-file":
        download_file_sync(args.path, token=args.token)
    elif args.action == "download-folder":
        download_folder_sync(args.path, token=args.token)
