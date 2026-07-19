from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Iterator

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers.utils import logging as hf_logging

from gomoku_ai.openvla_finetuning_prep import apply_special_tokens_to_tokenizer
from gomoku_ai.openvla_manifest_training import (
    collate_openvla_training_batch,
    write_training_summary,
)
from gomoku_ai.openvla_manifest_training import OpenVLAMoveOnlyDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune OpenVLA on a Gomoku OpenVLA-OFT manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--eval-manifest", help="Optional validation manifest evaluated during training.")
    parser.add_argument("--prep-dir", required=True, help="Directory containing special_tokens.json.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="openvla/openvla-7b")
    parser.add_argument("--stage", choices=("move_only",), default="move_only")
    parser.add_argument("--board-size", type=int, default=15)
    parser.add_argument("--image-key", default="board_top_before")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--max-eval-batches", type=int, default=16)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--verbose-third-party", action="store_true")
    parser.add_argument("--print-summary-json", action="store_true")
    parser.add_argument("--no-plot", action="store_true", help="Disable loss CSV/PNG artifact writing.")
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use bitsandbytes 4-bit loading. This can be version-sensitive; bf16 LoRA is the default smoke path.",
    )
    parser.add_argument("--no-4bit", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.eval_steps <= 0:
        parser.error("--eval-steps must be positive")
    if args.max_eval_batches <= 0:
        parser.error("--max-eval-batches must be positive")
    if args.log_steps <= 0:
        parser.error("--log-steps must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    if args.lora_rank <= 0:
        parser.error("--lora-rank must be positive")
    if not args.verbose_third_party:
        _quiet_third_party_logs()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    special_tokens = json.loads((Path(args.prep_dir) / "special_tokens.json").read_text(encoding="utf-8"))
    _log_run_start(args, output_dir)

    _log_stage("model", f"loading processor base={args.base_model}")
    processor = AutoProcessor.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    added_tokens = apply_special_tokens_to_tokenizer(processor.tokenizer, special_tokens)
    _log_stage(
        "tokens",
        f"added={added_tokens} tokenizer_vocab={len(processor.tokenizer)} "
        f"move_tokens={len(special_tokens.get('move_tokens', []))} "
        f"action_tokens={len(special_tokens.get('action_tokens', []))}",
    )

    quantization_config = None
    if args.use_4bit and not args.no_4bit:
        _require_cuda_bitsandbytes()
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    quantization_label = "4bit-nf4" if quantization_config is not None else "bf16"
    _log_stage("model", f"loading weights quantization={quantization_label}")
    with _force_accelerate_hooks_for_quantized_load(enabled=quantization_config is not None):
        model = AutoModelForVision2Seq.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map={"": 0} if quantization_config is not None else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
    model.resize_token_embeddings(len(processor.tokenizer))

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)
    else:
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    _log_stage("model", f"ready device={next(model.parameters()).device} gpu={_gpu_memory_summary()}")

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=min(args.lora_rank, 16),
        lora_dropout=0.0,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    model = get_peft_model(model, lora_config)
    model.get_input_embeddings().weight.requires_grad_(True)
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None:
        output_embeddings.weight.requires_grad_(True)
    trainable, total = _parameter_counts(model)
    _log_stage(
        "lora",
        f"rank={args.lora_rank} trainable={trainable:,}/{total:,} "
        f"({100.0 * trainable / total:.3f}%) target_modules=all-linear",
    )

    dataset = OpenVLAMoveOnlyDataset(
        args.manifest,
        processor=processor,
        stage=args.stage,
        board_size=args.board_size,
        image_key=args.image_key,
        max_length=args.max_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda samples: collate_openvla_training_batch(
            samples,
            pad_token_id=processor.tokenizer.pad_token_id,
            padding_side=processor.tokenizer.padding_side,
        ),
    )
    _log_stage(
        "data",
        f"train_samples={len(dataset)} batch_size={args.batch_size} "
        f"batches_per_epoch={len(dataloader)} manifest={_short_path(args.manifest)}",
    )
    eval_loader = None
    eval_dataset = None
    if args.eval_manifest:
        eval_dataset = OpenVLAMoveOnlyDataset(
            args.eval_manifest,
            processor=processor,
            stage=args.stage,
            board_size=args.board_size,
            image_key=args.image_key,
            max_length=args.max_length,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda samples: collate_openvla_training_batch(
                samples,
                pad_token_id=processor.tokenizer.pad_token_id,
                padding_side=processor.tokenizer.padding_side,
            ),
        )
        _log_stage(
            "data",
            f"eval_samples={len(eval_dataset)} max_eval_batches={args.max_eval_batches} "
            f"manifest={_short_path(args.eval_manifest)}",
        )
    optimizer = AdamW((param for param in model.parameters() if param.requires_grad), lr=args.learning_rate)
    device = next(model.parameters()).device
    losses: list[float] = []
    eval_losses: list[dict[str, float]] = []
    model.train()
    step = 0
    train_started_at = time.perf_counter()
    last_log_at = train_started_at
    last_log_step = 0
    _log_stage(
        "train",
        f"start max_steps={args.max_steps} eval_steps={args.eval_steps if eval_loader is not None else 'off'} "
        f"log_steps={args.log_steps} lr={args.learning_rate:.2e}",
    )
    while step < args.max_steps:
        for batch in dataloader:
            step_started_at = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                output = model(
                    input_ids=batch.input_ids.to(device),
                    attention_mask=batch.attention_mask.to(device),
                    pixel_values=batch.pixel_values.to(device=device, dtype=torch.bfloat16),
                    labels=batch.labels.to(device),
                )
            loss = output.loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            step += 1
            if step == 1 or step % args.log_steps == 0 or step >= args.max_steps:
                now = time.perf_counter()
                interval_steps = step - last_log_step
                interval_seconds = max(now - last_log_at, 1e-9)
                tokens = int(batch.attention_mask.sum().detach().cpu())
                _log_train_step(
                    step=step,
                    max_steps=args.max_steps,
                    loss=losses[-1],
                    mean_loss=_mean_tail(losses, 20),
                    lr=args.learning_rate,
                    step_seconds=now - step_started_at,
                    samples_per_second=(interval_steps * args.batch_size) / interval_seconds,
                    tokens_per_second=(interval_steps * tokens) / interval_seconds,
                )
                last_log_at = now
                last_log_step = step
            if eval_loader is not None and step % args.eval_steps == 0:
                eval_started_at = time.perf_counter()
                eval_loss = _evaluate_loss(model, eval_loader, device=device, max_batches=args.max_eval_batches)
                eval_losses.append({"step": float(step), "loss": eval_loss})
                _log_eval_step(
                    step=step,
                    max_steps=args.max_steps,
                    eval_loss=eval_loss,
                    elapsed_seconds=time.perf_counter() - eval_started_at,
                    batches=min(len(eval_loader), args.max_eval_batches),
                )
                model.train()
            if step >= args.max_steps:
                break
    if eval_loader is not None and (not eval_losses or eval_losses[-1]["step"] != float(step)):
        eval_started_at = time.perf_counter()
        eval_loss = _evaluate_loss(model, eval_loader, device=device, max_batches=args.max_eval_batches)
        eval_losses.append({"step": float(step), "loss": eval_loss})
        _log_eval_step(
            step=step,
            max_steps=args.max_steps,
            eval_loss=eval_loss,
            elapsed_seconds=time.perf_counter() - eval_started_at,
            batches=min(len(eval_loader), args.max_eval_batches),
        )
        model.train()

    _log_stage("save", f"writing adapter={output_dir / 'adapter'} processor={output_dir / 'processor'}")
    model.save_pretrained(output_dir / "adapter")
    processor.save_pretrained(output_dir / "processor")
    summary = {
        "base_model": args.base_model,
        "manifest": args.manifest,
        "eval_manifest": args.eval_manifest,
        "stage": args.stage,
        "samples": len(dataset),
        "steps": step,
        "losses": losses,
        "eval_losses": eval_losses,
        "added_tokens": added_tokens,
        "output_dir": str(output_dir),
        "adapter_dir": str(output_dir / "adapter"),
        "processor_dir": str(output_dir / "processor"),
        "quantization": "4bit" if quantization_config is not None else "none",
        "lora_rank": args.lora_rank,
        "log_steps": args.log_steps,
        "train_seconds": time.perf_counter() - train_started_at,
    }
    write_training_summary(output_dir / "training_summary.json", summary)
    history_csv = None
    plot_path = None
    if not args.no_plot:
        history_csv = output_dir / "loss_history.csv"
        plot_path = output_dir / "loss_curve.png"
        _write_loss_history(history_csv, losses=losses, eval_losses=eval_losses)
        plot_written = _write_loss_plot(
            plot_path,
            losses=losses,
            eval_losses=eval_losses,
            title=f"Gomoku-VLA {args.stage} ({'4bit' if quantization_config is not None else 'bf16'})",
        )
        summary["history_csv"] = str(history_csv)
        summary["loss_plot"] = str(plot_path) if plot_written else None
        write_training_summary(output_dir / "training_summary.json", summary)
        if plot_written:
            _log_stage("plot", f"loss_curve={plot_path} history={history_csv}")
        else:
            _log_stage("plot", f"history={history_csv} loss_curve=unavailable")
    _log_stage(
        "done",
        f"steps={step} final_loss={losses[-1]:.6f} "
        f"best_eval={min((item['loss'] for item in eval_losses), default=float('nan')):.6f} "
        f"summary={output_dir / 'training_summary.json'}",
    )
    if args.print_summary_json:
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _evaluate_loss(model, dataloader, *, device, max_batches: int) -> float:
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                output = model(
                    input_ids=batch.input_ids.to(device),
                    attention_mask=batch.attention_mask.to(device),
                    pixel_values=batch.pixel_values.to(device=device, dtype=torch.bfloat16),
                    labels=batch.labels.to(device),
                )
            losses.append(float(output.loss.detach().cpu()))
            if batch_index + 1 >= max_batches:
                break
    return sum(losses) / len(losses)


def _write_loss_history(path: Path, *, losses: list[float], eval_losses: list[dict[str, float]]) -> None:
    eval_by_step = {int(item["step"]): float(item["loss"]) for item in eval_losses}
    with path.open("w", encoding="utf-8") as handle:
        handle.write("step,train_loss,train_loss_20,eval_loss\n")
        for index, loss in enumerate(losses, start=1):
            eval_value = eval_by_step.get(index)
            eval_text = "" if eval_value is None else f"{eval_value:.8f}"
            handle.write(f"{index},{loss:.8f},{_mean_tail(losses[:index], 20):.8f},{eval_text}\n")


def _write_loss_plot(
    path: Path,
    *,
    losses: list[float],
    eval_losses: list[dict[str, float]],
    title: str,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    steps = list(range(1, len(losses) + 1))
    rolling = [_mean_tail(losses[:index], 20) for index in steps]
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    ax.plot(steps, losses, color="#8aa0ff", linewidth=1.0, alpha=0.35, label="train loss")
    ax.plot(steps, rolling, color="#1f4fff", linewidth=2.2, label="train loss (20-step mean)")
    if eval_losses:
        eval_steps = [int(item["step"]) for item in eval_losses]
        eval_values = [float(item["loss"]) for item in eval_losses]
        ax.plot(eval_steps, eval_values, color="#d62728", marker="o", linewidth=2.0, label="validation loss")
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.grid(True, color="#dddddd", linewidth=0.8, alpha=0.7)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def _require_cuda_bitsandbytes() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "--use-4bit requires CUDA, but torch.cuda.is_available() is false in this Python environment. "
            "Use a CUDA-enabled environment or rerun without --use-4bit."
        )
    try:
        import bitsandbytes.functional as bnb_functional
    except ImportError as exc:
        raise RuntimeError("--use-4bit requires bitsandbytes to be installed.") from exc
    library_name = str(getattr(getattr(bnb_functional, "lib", None), "_name", ""))
    if library_name.endswith("libbitsandbytes_cpu.so"):
        raise RuntimeError(
            "--use-4bit requires a CUDA-enabled bitsandbytes build, but this environment loaded "
            f"{library_name!r}. Install a CUDA bitsandbytes wheel or rerun without --use-4bit."
        )


def _quiet_third_party_logs() -> None:
    hf_logging.set_verbosity_error()
    warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.*")
    warnings.filterwarnings("ignore", message=".*torch.utils.checkpoint: please pass in use_reentrant.*")
    warnings.filterwarnings("ignore", message=".*Setting `save_embedding_layers` to `True`.*")


def _log_run_start(args: argparse.Namespace, output_dir: Path) -> None:
    print("", flush=True)
    print("=" * 88, flush=True)
    print("Gomoku-VLA OpenVLA LoRA fine-tuning", flush=True)
    print("-" * 88, flush=True)
    print(f"run.output_dir      : {output_dir}", flush=True)
    print(f"run.stage           : {args.stage}", flush=True)
    print(f"run.base_model      : {args.base_model}", flush=True)
    print(f"run.quantization    : {'4bit-nf4' if args.use_4bit and not args.no_4bit else 'bf16'}", flush=True)
    print(f"run.train_manifest  : {_short_path(args.manifest)}", flush=True)
    if args.eval_manifest:
        print(f"run.eval_manifest   : {_short_path(args.eval_manifest)}", flush=True)
    print("=" * 88, flush=True)


def _log_stage(stage: str, message: str) -> None:
    print(f"[{stage.upper():>5}] {message}", flush=True)


def _log_train_step(
    *,
    step: int,
    max_steps: int,
    loss: float,
    mean_loss: float,
    lr: float,
    step_seconds: float,
    samples_per_second: float,
    tokens_per_second: float,
) -> None:
    progress = 100.0 * step / max_steps
    eta_seconds = (max_steps - step) * max(step_seconds, 1e-9)
    print(
        "[TRAIN] "
        f"step={step:06d}/{max_steps:06d} "
        f"progress={progress:6.2f}% "
        f"loss={loss:9.5f} "
        f"loss_20={mean_loss:9.5f} "
        f"ppl={_safe_ppl(loss):>10} "
        f"lr={lr:.2e} "
        f"step_time={step_seconds:6.2f}s "
        f"samples/s={samples_per_second:6.2f} "
        f"tokens/s={tokens_per_second:7.1f} "
        f"eta={_format_duration(eta_seconds)} "
        f"gpu={_gpu_memory_summary()}",
        flush=True,
    )


def _log_eval_step(
    *,
    step: int,
    max_steps: int,
    eval_loss: float,
    elapsed_seconds: float,
    batches: int,
) -> None:
    print(
        "[ EVAL] "
        f"step={step:06d}/{max_steps:06d} "
        f"val_loss={eval_loss:9.5f} "
        f"val_ppl={_safe_ppl(eval_loss):>10} "
        f"batches={batches} "
        f"eval_time={_format_duration(elapsed_seconds)} "
        f"gpu={_gpu_memory_summary()}",
        flush=True,
    )


def _parameter_counts(model) -> tuple[int, int]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
    return trainable, total


def _mean_tail(values: list[float], window: int) -> float:
    tail = values[-window:]
    return sum(tail) / len(tail)


def _safe_ppl(loss: float) -> str:
    if loss > 20.0:
        return ">4.85e8"
    return f"{math.exp(loss):.3e}"


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _gpu_memory_summary() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    return f"{allocated:.1f}G/{reserved:.1f}G/{total:.1f}G"


def _short_path(path: str | Path, *, max_length: int = 92) -> str:
    text = str(path)
    if len(text) <= max_length:
        return text
    return "..." + text[-(max_length - 3) :]


@contextlib.contextmanager
def _force_accelerate_hooks_for_quantized_load(*, enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    import accelerate.big_modeling as accelerate_big_modeling
    import transformers.modeling_utils as transformers_modeling_utils

    original_accelerate_dispatch = accelerate_big_modeling.dispatch_model
    original_transformers_dispatch = transformers_modeling_utils.dispatch_model

    def dispatch_with_hooks(model, *args, **kwargs):
        kwargs["force_hooks"] = True
        return original_accelerate_dispatch(model, *args, **kwargs)

    accelerate_big_modeling.dispatch_model = dispatch_with_hooks
    transformers_modeling_utils.dispatch_model = dispatch_with_hooks
    try:
        yield
    finally:
        accelerate_big_modeling.dispatch_model = original_accelerate_dispatch
        transformers_modeling_utils.dispatch_model = original_transformers_dispatch


if __name__ == "__main__":
    main()
