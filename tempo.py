import argparse
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description="Change the tempo of an audio file without changing pitch (uses ffmpeg atempo).")
parser.add_argument("input", nargs="?", default="cq.wav", help="input audio file (default: cq.wav)")
parser.add_argument("-t", "--tempo", type=float, default=1.25, help="tempo rate, >1 is faster (default: 1.25)")
parser.add_argument("-o", "--output", help="output file (default: <input>_fast.wav)")
args = parser.parse_args()

if args.tempo <= 0:
    parser.error("tempo must be positive")

# atempo works best within 0.5–2.0, so chain several filters for larger changes
filters = []
rate = args.tempo
while rate > 2.0:
    filters.append("atempo=2.0")
    rate /= 2.0
while rate < 0.5:
    filters.append("atempo=0.5")
    rate /= 0.5
filters.append(f"atempo={rate}")

output = args.output or str(Path(args.input).with_name(Path(args.input).stem + "_fast.wav"))
subprocess.run(
    ["ffmpeg", "-loglevel", "error", "-y", "-i", args.input, "-filter:a", ",".join(filters), output],
    check=True,
)
print(f"Wrote {output} ({args.tempo}x)")
