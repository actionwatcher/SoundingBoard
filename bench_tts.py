import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# (name, language, text) — Chatterbox takes ISO codes, Qwen takes language names
PROMPTS = [
    ("en_cq", "en", "CQ contest, CQ contest, this is {call}, {call}, contest."),
    ("en_exchange", "en", "Thank you, five nine, zone fifteen."),
    ("ru_cq", "ru", "Всем, всем, всем! Здесь {call_ru}. Приём."),
]
QWEN_LANG = {"en": "english", "ru": "russian"}

parser = argparse.ArgumentParser(description="Compare generation speed of faster-qwen3-tts and Chatterbox Multilingual.")
parser.add_argument("--models", nargs="+", default=["qwen", "chatterbox"], choices=["qwen", "chatterbox"])
parser.add_argument("-n", "--runs", type=int, default=3, help="timed runs per prompt, after one warmup (default: 3)")
parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu (default: auto)")
parser.add_argument("--ref", help="reference wav for voice cloning (Qwen then uses the Base model)")
parser.add_argument("--ref-text", default="", help="transcript of --ref, improves Qwen cloning")
parser.add_argument("--qwen-model", help="HF id (default: 1.7B-CustomVoice, or 1.7B-Base with --ref)")
parser.add_argument("--qwen-speaker", help="built-in Qwen speaker (default: first available)")
parser.add_argument("--quant", default="Q8_0", help="GGML quant on Mac: F32, Q8_0, Q4_K_M; BF16 crashes on M1 Metal (default: Q8_0)")
parser.add_argument("--dtype", default="float16", help="torch dtype for Qwen on CUDA; 2080 Ti has no bf16 (default: float16)")
parser.add_argument("--call", default="Kilo Seven Alpha Bravo Charlie", help="callsign in English phonetics")
parser.add_argument("--call-ru", default="Кило Семь Альфа Браво Чарли", help="callsign as spoken in Russian")
parser.add_argument("-o", "--outdir", default="bench_out", help="where to write wavs and results (default: bench_out)")
parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
args = parser.parse_args()


def pick_device():
    import torch

    if args.device != "auto":
        return args.device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device):
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def peak_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10  # bytes on macOS, KiB on Linux


def gpu_peak_mb(device):
    import torch

    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    return None


def load_qwen(device):
    from faster_qwen3_tts import FasterQwen3TTS

    model_id = args.qwen_model or f"Qwen/Qwen3-TTS-12Hz-1.7B-{'Base' if args.ref else 'CustomVoice'}"
    if device == "cuda":
        backend = f"torch/{args.dtype}"
        model = FasterQwen3TTS.from_pretrained(model_id, device="cuda", dtype=args.dtype)
    else:
        # Torch backend is CUDA-graph only; on Mac/CPU use qwentts.cpp (Metal on Apple Silicon)
        backend = f"ggml/{args.quant}"
        model = FasterQwen3TTS.from_pretrained(model_id, backend="ggml", quant=args.quant)

    if args.ref:
        def generate(text, lang):
            wavs, sr = model.generate_voice_clone(
                text, QWEN_LANG[lang], ref_audio=args.ref, ref_text=args.ref_text, xvec_only=not args.ref_text
            )
            return np.asarray(wavs[0]), sr
        voice = f"clone:{Path(args.ref).name}"
    else:
        speaker = args.qwen_speaker or model.get_supported_speakers()[0]

        def generate(text, lang):
            wavs, sr = model.generate_custom_voice(text, speaker, QWEN_LANG[lang])
            return np.asarray(wavs[0]), sr
        voice = speaker
    return generate, f"{model_id.split('/')[-1]} [{backend}, {voice}]"


def load_chatterbox(device):
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    def generate(text, lang):
        wav = model.generate(text, language_id=lang, audio_prompt_path=args.ref)
        return wav.squeeze(0).cpu().numpy(), model.sr
    voice = f"clone:{Path(args.ref).name}" if args.ref else "builtin"
    return generate, f"Chatterbox Multilingual [torch/{device}, {voice}]"


def worker(name):
    """Benchmark one model in this process so memory peaks are not mixed between models."""
    device = pick_device()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    generate, label = (load_qwen if name == "qwen" else load_chatterbox)(device)
    load_s = time.perf_counter() - t0

    rows = []
    for key, lang, template in PROMPTS:
        text = template.format(call=args.call, call_ru=args.call_ru)
        times, durations = [], []
        for run in range(args.runs + 1):
            sync(device)
            t0 = time.perf_counter()
            audio, sr = generate(text, lang)
            sync(device)
            elapsed = time.perf_counter() - t0
            if run == 0:
                warmup_s = elapsed
                sf.write(outdir / f"{name}_{key}.wav", audio.astype(np.float32), sr)
                continue
            times.append(elapsed)
            durations.append(len(audio) / sr)
        rows.append({
            "prompt": key,
            "warmup_s": warmup_s,
            "gen_s": float(np.median(times)),
            "audio_s": float(np.median(durations)),
            "rtf": float(np.median(np.array(times) / np.array(durations))),
        })

    result = {
        "model": name,
        "label": label,
        "device": device,
        "load_s": load_s,
        "peak_rss_mb": peak_rss_mb(),
        # torch can't see qwentts.cpp's Metal allocations
        "gpu_peak_mb": None if "ggml" in label else gpu_peak_mb(device),
        "rows": rows,
    }
    (outdir / f"result_{name}.json").write_text(json.dumps(result, indent=2))


def report(results):
    print(f"\nHost: {platform.platform()}, Python {platform.python_version()}, {args.runs} timed runs/prompt")
    print("RTF = generation time / audio length (lower is better; < 1 is faster than real time)\n")
    for r in results:
        gpu = f", GPU peak {r['gpu_peak_mb']:.0f} MB" if r["gpu_peak_mb"] else ""
        print(f"{r['label']}")
        print(f"  load {r['load_s']:.1f} s, peak RSS {r['peak_rss_mb']:.0f} MB{gpu}")
        print(f"  {'prompt':<12} {'warmup s':>9} {'gen s':>7} {'audio s':>8} {'RTF':>6}")
        for row in r["rows"]:
            print(f"  {row['prompt']:<12} {row['warmup_s']:>9.2f} {row['gen_s']:>7.2f} {row['audio_s']:>8.2f} {row['rtf']:>6.2f}")
        print()
    print(f"Samples from each model's warmup run are in {args.outdir}/")


if args.worker:
    worker(args.models[0])
else:
    results = []
    for name in args.models:
        print(f"Running {name}...", flush=True)
        argv = [a for a in sys.argv[1:]]
        if "--models" in argv:
            i = argv.index("--models")
            j = i + 1
            while j < len(argv) and not argv[j].startswith("-"):
                j += 1
            del argv[i:j]
        proc = subprocess.run([sys.executable, __file__, *argv, "--models", name, "--worker"])
        if proc.returncode != 0:
            print(f"  {name} failed (exit {proc.returncode})")
            continue
        results.append(json.loads((Path(args.outdir) / f"result_{name}.json").read_text()))
    report(results)
