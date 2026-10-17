"""Generate Direct LLM answers with the paper's five-example prompt."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from typing import Any

BASE_MODEL = "kittn/mistral-7B-v0.1-hf"
ADAPTER = "xukehu/Mistral-7B-LoRA-Toponym-Resolution"
MAX_MENTION_LEN = 1200
GENERATION_PARAMETERS = {
    "temperature": 0.7, "top_p": 0.9, "top_k": 50,
    "num_beams": 4, "max_new_tokens": 128,
}
DEFAULT_PROMPT = Path(__file__).resolve().parents[1] / "docs/prompts/direct_llm.txt"


def record_id(row: dict[str, Any]) -> str:
    value = row.get("tweets_id", row.get("case_id"))
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError("Each input record requires a nonempty tweets_id or case_id")
    if "tweets_id" in row and "case_id" in row and str(row["tweets_id"]) != str(row["case_id"]):
        raise ValueError("Conflicting record identifiers")
    return str(value)


def compose_input(row: dict[str, Any]) -> str:
    """Preserve the original stripping, truncation, and concatenation behavior."""
    raw_text = (row.get("text") or "").strip()
    mentioned_text = (row.get("mentioned_text") or "").strip()
    if len(mentioned_text) > MAX_MENTION_LEN:
        removed = len(mentioned_text) - MAX_MENTION_LEN
        mentioned_text = mentioned_text[:MAX_MENTION_LEN] + f"... [TRUNCATED {removed} chars]"
    return raw_text + ("\n\n" + mentioned_text if mentioned_text else "")


def extract_answer(decoded: str) -> str | None:
    """Preserve last A: and first following line; no marker means no replacement."""
    return decoded.split("A:")[-1].split("\n")[0].strip() if "A:" in decoded else None


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name+".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


@contextmanager
def output_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Output lock exists: {path}; verify the earlier process has stopped before removing it") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


def load_checkpoint(path: Path, identity: dict[str, Any], ids: set[str], resume: bool) -> dict[str, Any]:
    if not path.exists():
        if resume:
            raise ValueError("--resume requires an existing checkpoint")
        return {"schema_version": 1, "identity": identity, "completed": {}}
    if not resume:
        raise ValueError("Checkpoint exists; use --resume or a different output/checkpoint path")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != 1 or state.get("identity") != identity:
        raise ValueError("Checkpoint input, prompt, model, device, or generation settings differ")
    completed = state.get("completed")
    if not isinstance(completed, dict) or not set(completed) <= ids:
        raise ValueError("Checkpoint contains invalid record IDs")
    for value in completed.values():
        if not isinstance(value, dict) or value.get("status") not in {"generated", "missing_answer_marker"}:
            raise ValueError("Invalid checkpoint record status")
        if value["status"] == "generated" and not isinstance(value.get("answer"), str):
            raise ValueError("Generated checkpoint answers must be strings")
    return state


def completed_output(rows: list[dict[str, Any]], completed: dict[str, Any]) -> list[dict[str, Any]]:
    if set(completed) != {record_id(row) for row in rows}:
        raise ValueError("Final output requires an attempted generation for every input record")
    result = []
    for row in rows:
        item = dict(row)
        saved = completed[record_id(row)]
        if saved["status"] == "generated":
            item["answer"] = saved["answer"]
        # Match the original script's no-marker behavior: retain the input answer,
        # and make this explicit in checkpoint status rather than calling it generated.
        result.append(item)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--adapter", default=ADAPTER)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or an explicit device such as cuda:0")
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--offload-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    checkpoint = args.checkpoint or args.output.with_suffix(args.output.suffix+".checkpoint.json")
    if len({args.input.resolve(), args.output.resolve(), checkpoint.resolve()}) != 3:
        raise ValueError("Input, final output, and checkpoint must be distinct files")
    rows = json.loads(args.input.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Input must be a nonempty list of record objects")
    ids = [record_id(row) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Input contains duplicate IDs")
    prompt = args.prompt_file.read_text(encoding="utf-8")
    for row in rows:
        prompt.format(text=compose_input(row))
    identity = {"input_records": rows, "prompt": prompt, "model": args.model,
                "adapter": args.adapter, "device": args.device, "model_dtype": "torch.float32",
                "generation_parameters": GENERATION_PARAMETERS}
    state = load_checkpoint(checkpoint, identity, set(ids), args.resume)
    if args.output.exists() and not args.resume:
        raise ValueError("Output exists; use a new output path or resume its checkpoint")
    with output_lock(args.output.with_suffix(args.output.suffix+".lock")):
        # Reload after acquiring the lock to prevent a stale read from replacing
        # a checkpoint advanced by another process.
        state = load_checkpoint(checkpoint, identity, set(ids), args.resume)
        pending = [row for row in rows if record_id(row) not in state["completed"]]
        if pending:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
            device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
            mapping = "auto" if args.device == "auto" else {"": args.device}
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            base = AutoModelForCausalLM.from_pretrained(args.model, device_map=mapping, torch_dtype=torch.float32)
            offload = args.offload_dir or args.output.parent / "model_offload"
            model = PeftModel.from_pretrained(base, args.adapter, device_map=mapping, offload_folder=str(offload))
            model.eval()
            generation_config = GenerationConfig(**GENERATION_PARAMETERS)
            for row in pending:
                inputs = tokenizer(prompt.format(text=compose_input(row)), return_tensors="pt").to(device)
                with torch.no_grad():
                    output = model.generate(**inputs, generation_config=generation_config,
                                            return_dict_in_generate=True, output_scores=False)
                decoded = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
                answer = extract_answer(decoded)
                state["completed"][record_id(row)] = ({"status": "generated", "answer": answer}
                    if answer is not None else {"status": "missing_answer_marker"})
                atomic_json(checkpoint, state)
                print(f"Completed {len(state['completed'])}/{len(rows)}", flush=True)
        atomic_json(args.output, completed_output(rows, state["completed"]))
        missing = sum(v["status"] == "missing_answer_marker" for v in state["completed"].values())
        print(json.dumps({"output_records": len(rows), "missing_answer_marker": missing}))


if __name__ == "__main__":
    main()
