"""
Minimal Tripo3D (v2 openapi) client used by the kitty asset pipeline.

Auth: reads the API key from the TRIPO_API_KEY environment variable ONLY.
Never hard-code or log the key value itself.
"""
import os
import time
import json
import urllib.request
import urllib.error

BASE = "https://api.tripo3d.ai/v2/openapi"


def _headers(extra=None):
    key = os.environ["TRIPO_API_KEY"]
    h = {"Authorization": f"Bearer {key}"}
    if extra:
        h.update(extra)
    return h


def _request(method, url, data=None, headers=None, is_json=True):
    if is_json and data is not None:
        body = json.dumps(data).encode("utf-8")
        hdrs = _headers({"Content-Type": "application/json"})
    else:
        body = data
        hdrs = _headers(headers)
    req = urllib.request.Request(url, data=body, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        payload = e.read().decode("utf-8")
        raise RuntimeError(f"HTTP {e.code} calling {url}: {payload}") from None


def upload_image(path):
    """Upload a raw image file, returns image_token (used as file_token)."""
    boundary = "----tripoUpload"
    with open(path, "rb") as f:
        content = f.read()
    filename = os.path.basename(path)
    ext = filename.rsplit(".", 1)[-1].lower()
    mime = "image/png" if ext == "png" else "image/jpeg"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode("utf-8") + content + f"\r\n--{boundary}--\r\n".encode("utf-8")
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    resp = _request("POST", f"{BASE}/upload", data=body, headers=headers, is_json=False)
    if resp.get("code") != 0:
        raise RuntimeError(f"upload failed: {resp}")
    return resp["data"]["image_token"]


def create_task(payload):
    resp = _request("POST", f"{BASE}/task", data=payload)
    if resp.get("code") != 0:
        raise RuntimeError(f"create_task failed: {resp}")
    return resp["data"]["task_id"]


def get_task(task_id):
    resp = _request("GET", f"{BASE}/task/{task_id}")
    if resp.get("code") != 0:
        raise RuntimeError(f"get_task failed: {resp}")
    return resp["data"]


def wait_for_task(task_id, poll_seconds=3, timeout=900, verbose=True):
    start = time.time()
    last_progress = None
    while True:
        data = get_task(task_id)
        status = data["status"]
        progress = data.get("progress")
        if verbose and progress != last_progress:
            print(f"  [{task_id}] status={status} progress={progress}")
            last_progress = progress
        if status == "success":
            return data
        if status in ("failed", "banned", "expired", "cancelled", "unknown"):
            raise RuntimeError(f"task {task_id} ended with status={status}: {data}")
        if time.time() - start > timeout:
            raise TimeoutError(f"task {task_id} timed out after {timeout}s (last status={status})")
        time.sleep(poll_seconds)


def download(url, dest_path):
    urllib.request.urlretrieve(url, dest_path)
    return dest_path
