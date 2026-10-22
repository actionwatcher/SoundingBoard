# Replicating the `sound` conda environment

Instructions for an AI coding agent (or a human) setting up this project on a new machine.
Follow the steps in order: the order matters (see "Why it is built this way").

## Target

- Conda env name: `sound`
- Python 3.13
- TTS packages: `faster-qwen3-tts` (Qwen3-TTS) and `chatterbox-tts` (Chatterbox Multilingual)
- `ffmpeg` on PATH (used by `tempo.py`)

## Known-good versions (macOS, Apple M1 Pro, 2026-10-22)

| Package | Version | Source |
|---|---|---|
| python | 3.13.9 | conda-forge |
| numpy | 2.5.4 | conda-forge |
| librosa | 1.0.0 | conda-forge |
| pytorch | 2.13.0 (`cpu_generic` build, MPS available) | conda-forge |
| torchaudio | 2.11.0 | conda-forge |
| faster-qwen3-tts | 0.5.4 | pip |
| qwentts-cpp-python | 0.5.0 (macOS arm64 only) | pip |
| transformers | 5.20.2 | pip |
| chatterbox-tts | 0.1.7 (installed with `--no-deps`) | pip |
| diffusers | 0.41.0 | pip |

Pin to these versions if a newer release breaks something.

## Why it is built this way

1. **The two TTS packages have conflicting pins.** `chatterbox-tts` pins `transformers==5.2.0` (0.1.7)
   or `==4.46.3` (≤0.1.6). `faster-qwen3-tts` needs `transformers>=5.15.1` (≥0.4) or `==4.57.3` (≤0.3).
   No combination resolves. Install `faster-qwen3-tts` normally, then `chatterbox-tts --no-deps`
   and its unpinned dependencies separately. Chatterbox works with the newer stack; `pip check`
   will still report its pins. That is expected.
2. **Torch must come from conda-forge, not pip.** The conda-forge stack (OpenBLAS → `llvm-openmp`)
   and the pip `torch` wheel each ship a `libomp`. With both loaded, `import torch` aborts with
   `OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized`.
   Do **not** work around it with `KMP_DUPLICATE_LIB_OK=TRUE`.
3. **Install conda torch before the pip packages**, so pip sees `torch`/`torchaudio` as already
   satisfied and does not pull in its own wheels. After every pip step, check that no pip torch
   appeared: `pip list | grep -i torch` must show `torch`/`torchaudio` with the conda versions,
   and `conda list -n sound torch` must show channel `conda-forge`, not `pypi`.
4. **Pick the conda torch/torchaudio pair together.** Don't pin `pytorch` alone: conda-forge
   `torchaudio` builds require a specific `pytorch` minor (for example torchaudio 2.11.0 → pytorch 2.12/2.13).
   Let conda solve both at once.

## Steps: macOS (Apple Silicon)

```bash
conda create -y -n sound -c conda-forge --override-channels python=3.13 \
    numpy scipy librosa soundfile soxr scikit-learn numba matplotlib ffmpeg
conda install -y -n sound -c conda-forge --override-channels pytorch torchaudio

P=$(conda run -n sound which pip)
$P install faster-qwen3-tts          # pulls qwentts-cpp-python (GGML + Metal) on macOS arm64
$P install --no-deps chatterbox-tts==0.1.7
$P install s3tokenizer diffusers resemble-perth conformer==0.3.2 spacy-pkuseg \
    pykakasi==2.3.0 pyloudnorm omegaconf
```

On macOS, Qwen runs through `backend="ggml"` (qwentts.cpp, Metal). The faster-qwen3-tts
torch backend is CUDA-only. Chatterbox runs on PyTorch `mps`.

- **M1 (and likely M2): use `quant="Q8_0"`, not the default `BF16`.** M1 Metal has no bf16 kernels:
  BF16 segfaults with `kernel not found ... kernel_get_rows_bf16`. Q8_0 is verified on an M1 Pro.
- Chatterbox's Russian stress marking needs the optional `russian_text_stresser` module, which is not on PyPI.
  Without it Chatterbox logs a warning and may misplace stress in Russian.

## Steps: PC (Linux + NVIDIA GPU)

Same as macOS, but install a CUDA build of PyTorch from conda-forge:

```bash
conda create -y -n sound -c conda-forge --override-channels python=3.13 \
    numpy scipy librosa soundfile soxr scikit-learn numba matplotlib ffmpeg
conda install -y -n sound -c conda-forge --override-channels "pytorch=*=cuda*" torchaudio
# then the same three pip commands as macOS
```

- Check: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"`.
- `qwentts-cpp-python` is not installed on Linux (it is macOS arm64 only unless the `[ggml]` extra is used),
  so Qwen uses the torch + CUDA-graphs backend.
- **Turing GPUs (RTX 20xx, e.g. 2080 Ti):** no native bf16 and no FlashAttention 2. Load Qwen with
  `dtype="float16"` (the default in `bench_tts.py`) and `attn_implementation="sdpa"`. If output is
  silent or noisy, use `float32`. Do not install `flash-attn`.
- The pip torch wheel on Linux may not hit the libomp clash, but stick to conda-forge torch for consistency.

## Steps: PC (Windows)

Untested. Use the Linux steps in an Anaconda Prompt. If conda-forge has no CUDA `pytorch` build for
the current Python, use Python 3.12 or WSL2 + the Linux steps instead (preferred).

## Verify

Run from the project root:

```bash
conda run -n sound python -c "
import torch, torchaudio, librosa
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'mps', torch.backends.mps.is_available())
import faster_qwen3_tts
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
print('chatterbox ru:', 'ru' in SUPPORTED_LANGUAGES)
"
conda run -n sound pip check   # only chatterbox-tts pin complaints are acceptable
conda run -n sound python bench_tts.py -n 1   # end-to-end: downloads models, writes bench_out/
```

Expected: no `OMP: Error #15`, `mps True` on Mac or `cuda True` on Linux, `chatterbox ru: True`.
The first `bench_tts.py` run downloads several GB of models into `~/.cache/huggingface`.
Set `HF_TOKEN` for faster downloads.
