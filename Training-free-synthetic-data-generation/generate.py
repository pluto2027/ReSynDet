"""Generate synthetic plant images through the ComfyUI HTTP API."""

import argparse
import copy
import hashlib
import io
import json
import time
import uuid
from pathlib import Path

import requests
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class ComfyUIClient:
    def __init__(self, server_address, request_timeout=60):
        server_address = server_address.rstrip("/")
        if server_address.startswith(("http://", "https://")):
            self.base_url = server_address
        else:
            self.base_url = "http://" + server_address
        self.client_id = str(uuid.uuid4())
        self.request_timeout = request_timeout
        self.session = requests.Session()

    def upload_image(self, image_path):
        with image_path.open("rb") as handle:
            response = self.session.post(
                self.base_url + "/upload/image",
                files={"image": (image_path.name, handle)},
                data={"overwrite": "true"},
                timeout=self.request_timeout,
            )
        response.raise_for_status()
        payload = response.json()
        if "name" not in payload:
            raise RuntimeError("ComfyUI upload response does not contain an image name.")
        return payload["name"]

    def queue_prompt(self, workflow):
        response = self.session.post(
            self.base_url + "/prompt",
            json={"prompt": workflow, "client_id": self.client_id},
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if "prompt_id" not in payload:
            raise RuntimeError("ComfyUI prompt response does not contain prompt_id.")
        return payload["prompt_id"]

    def get_history(self, prompt_id):
        response = self.session.get(
            self.base_url + "/history/" + prompt_id,
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        return response.json()

    def download_image(self, image_record):
        response = self.session.get(
            self.base_url + "/view",
            params={
                "filename": image_record["filename"],
                "subfolder": image_record.get("subfolder", ""),
                "type": image_record.get("type", "output"),
            },
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        return response.content

    def wait_for_output(self, prompt_id, timeout, poll_interval):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            history = self.get_history(prompt_id)
            task = history.get(prompt_id)
            if task:
                status = task.get("status", {})
                if isinstance(status, str):
                    status = {"status_str": status}
                if status.get("status_str") == "error":
                    raise RuntimeError(
                        "ComfyUI task failed: {}".format(task.get("messages", []))
                    )
                if task.get("outputs"):
                    return task
            time.sleep(poll_interval)
        raise TimeoutError("ComfyUI task {} timed out.".format(prompt_id))


def deterministic_noise_seed(base_seed, image_name, run_id):
    value = "{}:{}:{}".format(base_seed, image_name, run_id).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def configure_workflow(template, server_image_name, image_width, prompt, noise_seed):
    workflow = copy.deepcopy(template)
    try:
        workflow["17"]["inputs"]["image"] = server_image_name
        workflow["23"]["inputs"]["text"] = prompt
        workflow["44"]["inputs"].update(
            {"left": 0, "top": 0, "right": image_width, "bottom": 0}
        )
        workflow["46"]["inputs"]["noise_seed"] = noise_seed
    except KeyError as error:
        raise KeyError(
            "The workflow is missing a required node or input: {}".format(error)
        ) from error
    return workflow


def generate_one(
    client,
    template,
    image_path,
    output_dir,
    run_id,
    prompt,
    base_seed,
    output_node,
    task_timeout,
    poll_interval,
    overwrite,
):
    expected_output = output_dir / "{}_run{}_0_right.jpg".format(
        image_path.stem, run_id
    )
    if expected_output.exists() and not overwrite:
        print("Skip existing: {}".format(expected_output))
        return [expected_output]

    with Image.open(image_path) as source:
        source_width = source.width

    server_image_name = client.upload_image(image_path)
    noise_seed = deterministic_noise_seed(base_seed, image_path.name, run_id)
    workflow = configure_workflow(
        template, server_image_name, source_width, prompt, noise_seed
    )
    prompt_id = client.queue_prompt(workflow)
    task = client.wait_for_output(prompt_id, task_timeout, poll_interval)

    node_output = task.get("outputs", {}).get(output_node)
    if not node_output or not node_output.get("images"):
        raise RuntimeError(
            "Output node {} produced no images for prompt {}.".format(
                output_node, prompt_id
            )
        )

    saved = []
    for index, image_record in enumerate(node_output["images"]):
        image_bytes = client.download_image(image_record)
        with Image.open(io.BytesIO(image_bytes)) as result:
            result = result.convert("RGB")
            if result.width <= source_width:
                raise ValueError(
                    "Outpaint result width ({}) must exceed source width ({}).".format(
                        result.width, source_width
                    )
                )
            generated_region = result.crop(
                (source_width, 0, result.width, result.height)
            )
            output_path = output_dir / "{}_run{}_{}_right.jpg".format(
                image_path.stem, run_id, index
            )
            generated_region.save(output_path, quality=95)
            saved.append(output_path)
            print("Saved: {}".format(output_path))
    return saved


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--server", default="127.0.0.1:8188")
    parser.add_argument(
        "--workflow",
        default=str(script_dir / "flux_fill_outpaint_detail_final_plant.json"),
    )
    parser.add_argument(
        "--prompt", default="The top-view photograph of leafy greens"
    )
    parser.add_argument("--runs-per-image", type=int, default=20)
    parser.add_argument("--run-start", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-node", default="9")
    parser.add_argument("--task-timeout", type=float, default=1800.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.runs_per_image <= 0:
        raise ValueError("--runs-per-image must be positive.")
    if args.poll_interval <= 0 or args.task_timeout <= 0:
        raise ValueError("Timeout and polling interval must be positive.")

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    workflow_path = Path(args.workflow).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if not workflow_path.is_file():
        raise FileNotFoundError(workflow_path)

    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError("No input images were found in {}.".format(input_dir))

    with workflow_path.open("r", encoding="utf-8") as handle:
        template = json.load(handle)
    output_dir.mkdir(parents=True, exist_ok=True)
    client = ComfyUIClient(args.server)

    total_saved = 0
    for image_index, image_path in enumerate(image_paths, start=1):
        print("[{}/{}] {}".format(image_index, len(image_paths), image_path.name))
        for run_id in range(args.run_start, args.run_start + args.runs_per_image):
            total_saved += len(
                generate_one(
                    client=client,
                    template=template,
                    image_path=image_path,
                    output_dir=output_dir,
                    run_id=run_id,
                    prompt=args.prompt,
                    base_seed=args.seed,
                    output_node=args.output_node,
                    task_timeout=args.task_timeout,
                    poll_interval=args.poll_interval,
                    overwrite=args.overwrite,
                )
            )
    print("Completed. Generated image files: {}".format(total_saved))


if __name__ == "__main__":
    main()
