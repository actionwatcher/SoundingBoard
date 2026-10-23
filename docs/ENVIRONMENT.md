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
    numpy scipy librosa pysoundfile soxr scikit-learn numba matplotlib ffmpeg
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
    numpy scipy librosa pysoundfile soxr scikit-learn numba matplotlib ffmpeg
conda install -y -n sound -c conda-forge --override-channels \
    "pytorch=2.13.0=cuda130_generic*" "torchaudio=2.11.0=cuda130*" "libblas=*=*openblas"
# then the same three pip commands as macOS
```

- **Pin the CUDA build of both packages to the same CUDA version.** With just `"pytorch=*=cuda*" torchaudio`,
  conda picks pytorch built for CUDA 13.4 and torchaudio built for CUDA 13.0 (no cuda134 torchaudio exists yet).
  `import torchaudio` then fails with `PyTorch and TorchAudio were compiled with different CUDA versions`.
  Check with `conda list -n sound "^(pytorch|torchaudio)$"`: both build strings must start with the same `cudaNNN`.
- **Use the `generic` build, not `mkl`.** The `cuda130_mkl` pytorch build switches `libblas` to MKL, and numpy
  then aborts with `Intel oneMKL FATAL ERROR: Cannot load libmkl_gf_lp64.so.3`. The `generic` build plus
  `libblas=*=*openblas` keeps the OpenBLAS stack.

- Check: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"`.
- `qwentts-cpp-python` is not installed on Linux (it is macOS arm64 only unless the `[ggml]` extra is used),
  so Qwen uses the torch + CUDA-graphs backend.
- **Turing GPUs (RTX 20xx, e.g. 2080 Ti):** no native bf16 and no FlashAttention 2. Load Qwen with
  `dtype="float16"` (the default in `bench_tts.py`) and `attn_implementation="sdpa"`. If output is
  silent or noisy, use `float32`. Do not install `flash-attn`.
- The pip torch wheel on Linux may not hit the libomp clash, but stick to conda-forge torch for consistency.
- **Qwen fails to load on Linux** with `AttributeError: 'MimiConfig' object has no attribute 'rope_theta'`.
  This is a bug in `qwen-tts-hf` 0.1.1.post1 (the only release), pulled in by `faster-qwen3-tts`. Its
  `qwen_tts/_transformers_compat.py` registers a global `"default"` RoPE initializer that reads `config.rope_theta`.
  Transformers 5.x applies the global registry over Mimi's own initializer, and Mimi's config keeps the value in
  `config.rope_parameters["rope_theta"]`. macOS is not affected (Qwen runs on the ggml backend there).
  Workaround: in `_default_rope_parameters`, change `base = config.rope_theta` to
  `base = getattr(config, "rope_theta", None) or config.rope_parameters["rope_theta"]`.
  A pip reinstall of `qwen-tts-hf` reverts it.
- `sox` is not installed by these steps; `qwen_tts` prints `SoX could not be found!` on import. This is only a warning.

Known-good on Linux (kernel 6.8, glibc 2.35, RTX 2080 Ti, driver 595, 2026-10-22): python 3.13.17,
pytorch 2.13.0 `cuda130_generic`, torchaudio 2.11.0 `cuda130`, libblas `openblas`; pip packages as in the table above.
`bench_tts.py -n 1`: Qwen 1.7B CustomVoice (float16) RTF ~0.43, ~4.6 GB GPU; Chatterbox RTF ~0.85, ~3.3 GB GPU.

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
