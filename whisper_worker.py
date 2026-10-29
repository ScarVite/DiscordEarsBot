"""Persistent faster-whisper worker, spawned once by index.js.

Protocol on stdin: one JSON header line {"id": int, "lang": str, "bytes": int} followed by
`bytes` of raw 16 kHz 16-bit little-endian mono PCM.
Reply on stdout: one JSON line {"id": int, "text": str} or {"id": int, "error": str}.
"""
import json
import os
import sys

import numpy as np
from faster_whisper import WhisperModel

out = sys.stdout
sys.stdout = sys.stderr  # keep any library prints off the protocol channel

model = WhisperModel(os.environ.get("WHISPER_MODEL", "base"), device="cpu", compute_type="int8", cpu_threads=3)
print("whisper worker ready", flush=True)

stdin = sys.stdin.buffer
for header in iter(stdin.readline, b""):
    req = json.loads(header)
    pcm = stdin.read(req["bytes"])
    try:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = model.transcribe(
            audio,
            language=req.get("lang") or None,  # empty/None = auto-detect
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        res = {"id": req["id"], "text": " ".join(s.text.strip() for s in segments).strip()}
    except Exception as e:  # bad language code, odd byte count, ...: report and keep serving
        res = {"id": req["id"], "error": str(e)}
    out.write(json.dumps(res) + "\n")
    out.flush()
