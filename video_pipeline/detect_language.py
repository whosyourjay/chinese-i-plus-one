#!/usr/bin/env python3
"""Detect spoken language of audio clips.

Pluggable backends (cheaper than full transcription):
  - whisper-tiny       openai-whisper detect_language (~260 ms/clip on CPU)
  - faster-whisper     CTranslate2-backed whisper-tiny detect (typically faster)
  - sensevoice         FunASR SenseVoice (zh/en/ja/ko/yue only; misclassifies hi/bn)

CLI:
  python3 video_pipeline/detect_language.py <clip.mp3> [...]
  python3 video_pipeline/detect_language.py --backend faster-whisper <clip.mp3> [...]
  python3 video_pipeline/detect_language.py --benchmark audio_segments/017*.mp3
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

DEFAULT_BACKEND = "ecapa"
ECAPA_MAX_SECONDS = 3.0  # plenty for language ID; full clips are wasteful
ECAPA_SAMPLE_RATE = 16000
_MODELS: dict[str, object] = {}


def _shim_torch_amp() -> None:
    """speechbrain>=1.0 calls torch.amp.custom_fwd(device_type=...) added in
    torch 2.4. PyTorch dropped Intel-Mac wheels after 2.2, so we route to the
    pre-2.4 torch.cuda.amp.custom_fwd which takes no device_type kwarg."""
    import torch
    if hasattr(torch.amp, "custom_fwd"):
        return
    def _wrap(orig):
        def factory(*args, **kwargs):
            kwargs.pop("device_type", None)
            return orig(*args, **kwargs) if (args or kwargs) else orig
        return factory
    torch.amp.custom_fwd = _wrap(torch.cuda.amp.custom_fwd)  # type: ignore[attr-defined]
    torch.amp.custom_bwd = _wrap(torch.cuda.amp.custom_bwd)  # type: ignore[attr-defined]


def _ecapa_model():
    if "ecapa" not in _MODELS:
        _shim_torch_amp()
        from speechbrain.inference.classifiers import EncoderClassifier
        _MODELS["ecapa"] = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            savedir="/tmp/sb_voxlingua107",
        )
    return _MODELS["ecapa"]


def _ecapa_truncate(signal):
    max_samples = int(ECAPA_MAX_SECONDS * ECAPA_SAMPLE_RATE)
    if signal.shape[-1] > max_samples:
        signal = signal[..., :max_samples]
    return signal


def _detect_ecapa(path: Path) -> str:
    model = _ecapa_model()
    signal = _ecapa_truncate(model.load_audio(str(path)))
    _, _, _, label = model.classify_batch(signal.unsqueeze(0))
    return label[0].split(":", 1)[0].strip()


def _detect_ecapa_batch(paths: list[Path]) -> list[str]:
    import torch
    model = _ecapa_model()
    sigs = [_ecapa_truncate(model.load_audio(str(p))) for p in paths]
    lengths = torch.tensor([s.shape[-1] for s in sigs], dtype=torch.float)
    max_len = int(lengths.max().item())
    padded = torch.zeros(len(sigs), max_len)
    for i, s in enumerate(sigs):
        padded[i, : s.shape[-1]] = s
    wav_lens = lengths / max_len
    _, _, _, labels = model.classify_batch(padded, wav_lens)
    return [label.split(":", 1)[0].strip() for label in labels]


def _whisper_model():
    if "whisper" not in _MODELS:
        import whisper
        _MODELS["whisper"] = whisper.load_model("tiny")
    return _MODELS["whisper"]


def _detect_whisper(path: Path) -> str:
    import whisper
    model = _whisper_model()
    audio = whisper.load_audio(str(path))
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    _, probs = model.detect_language(mel)
    return max(probs, key=probs.get)


def _detect_whisper_batch(paths: list[Path]) -> list[str]:
    import torch
    import whisper
    model = _whisper_model()
    mels = []
    for p in paths:
        audio = whisper.load_audio(str(p))
        audio = whisper.pad_or_trim(audio)
        mels.append(whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels))
    batch = torch.stack(mels).to(model.device)
    _, probs_list = model.detect_language(batch)
    return [max(probs, key=probs.get) for probs in probs_list]


def _faster_whisper_model():
    if "faster-whisper" not in _MODELS:
        from faster_whisper import WhisperModel
        _MODELS["faster-whisper"] = WhisperModel("tiny", device="cpu", compute_type="int8")
    return _MODELS["faster-whisper"]


def _detect_faster_whisper(path: Path) -> str:
    from faster_whisper.audio import decode_audio
    model = _faster_whisper_model()
    audio = decode_audio(str(path), sampling_rate=16000)
    features = model.feature_extractor(audio)
    encoder_output = model.encode(features[:, : model.feature_extractor.nb_max_frames])
    results = model.model.detect_language(encoder_output)
    token, _ = results[0][0]
    return token.strip("<|>")


def _sensevoice_model():
    if "sensevoice" not in _MODELS:
        from transcribe_audio import get_model
        _MODELS["sensevoice"] = get_model()
    return _MODELS["sensevoice"]


def _detect_sensevoice(path: Path) -> str:
    from transcribe_audio import transcribe
    _sensevoice_model()
    return transcribe(path)[0]


def _detect_sensevoice_batch(paths: list[Path]) -> list[str]:
    from transcribe_audio import transcribe_many
    _sensevoice_model()
    return [lang for lang, _ in transcribe_many(paths, batch_size=len(paths))]


_BACKENDS = {
    "ecapa": _detect_ecapa,
    "whisper-tiny": _detect_whisper,
    "faster-whisper": _detect_faster_whisper,
    "sensevoice": _detect_sensevoice,
}


def is_available(backend: str = DEFAULT_BACKEND) -> bool:
    try:
        if backend == "ecapa":
            import speechbrain  # noqa: F401
        elif backend == "whisper-tiny":
            import whisper  # noqa: F401
        elif backend == "faster-whisper":
            import faster_whisper  # noqa: F401
        elif backend == "sensevoice":
            import funasr  # noqa: F401
        else:
            return False
        return True
    except ImportError:
        return False


def detect(path: str | Path, backend: str = DEFAULT_BACKEND) -> str:
    """Return ISO-639-1 language code for one audio file (e.g. 'zh')."""
    return _BACKENDS[backend](Path(path))


_BATCH_BACKENDS = {
    "ecapa": _detect_ecapa_batch,
    "whisper-tiny": _detect_whisper_batch,
    "sensevoice": _detect_sensevoice_batch,
}


def detect_many(
    paths: list[Path],
    backend: str = DEFAULT_BACKEND,
    batch_size: int = 32,
) -> list[str]:
    """Return one language code per path, in order. '' on failure."""
    if batch_size > 1 and backend in _BATCH_BACKENDS:
        batch_fn = _BATCH_BACKENDS[backend]
        out: list[str] = []
        for start in range(0, len(paths), batch_size):
            chunk = paths[start : start + batch_size]
            try:
                out.extend(batch_fn(chunk))
            except Exception as exc:
                print(f"  {backend} batch failed at {start}: {exc}")
                out.extend([""] * len(chunk))
            print(f"  detected {min(start + batch_size, len(paths))}/{len(paths)}")
        return out
    out = []
    for i, p in enumerate(paths, 1):
        try:
            out.append(detect(p, backend))
        except Exception as exc:
            print(f"  {backend} failed on {p.name}: {exc}")
            out.append("")
        if i % 20 == 0 or i == len(paths):
            print(f"  detected {i}/{len(paths)}")
    return out


def benchmark(paths: list[Path], backends: list[str]) -> None:
    print(f"clips: {len(paths)}")
    for backend in backends:
        if not is_available(backend):
            print(f"{backend}: not available")
            continue
        detect(paths[0], backend)  # warmup
        t0 = time.perf_counter()
        for p in paths:
            detect(p, backend)
        dt = time.perf_counter() - t0
        print(f"{backend:18s} {dt:6.2f}s total  {dt/len(paths)*1000:6.0f} ms/clip")


def detect_many_timed(paths: list[Path], backend: str, batch_size: int = 1) -> None:
    detect(paths[0], backend)  # warmup
    t0 = time.perf_counter()
    detect_many(paths, backend=backend, batch_size=batch_size)
    dt = time.perf_counter() - t0
    bs = f"batch={batch_size}" if batch_size > 1 else "serial"
    print(f"{backend:18s} {bs:>10s}  {dt:6.2f}s total  "
          f"{dt/len(paths)*1000:6.0f} ms/clip on {len(paths)} clips")


_SOUND_RE = re.compile(r"\[sound:([^\]]+)\]")


def sample_clips_from_csv(csv_path: Path, n: int, seed: int, audio_dir: Path) -> list[Path]:
    import csv
    import random
    files: list[str] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw = (row.get("audio") or "").strip()
            m = _SOUND_RE.search(raw)
            name = m.group(1) if m else raw
            if name and (audio_dir / name).exists():
                files.append(name)
    random.Random(seed).shuffle(files)
    return [audio_dir / name for name in files[:n]]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="*", type=Path)
    p.add_argument("--backend", default=DEFAULT_BACKEND, choices=list(_BACKENDS))
    p.add_argument(
        "--sample-from-csv",
        type=int,
        metavar="N",
        help="Pick N random existing audio files listed in --csv (cold-cache).",
    )
    p.add_argument("--csv", type=Path, default=Path("data_files/all_sentences.csv"))
    p.add_argument("--audio-dir", type=Path, default=Path("audio_segments"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Time the chosen --backend on the given clips.",
    )
    p.add_argument(
        "--benchmark-all",
        action="store_true",
        help="Time every backend (may hit OMP collisions when mixing torch/CT2).",
    )
    p.add_argument("--batch-size", type=int, default=1)
    args = p.parse_args()
    if args.sample_from_csv:
        paths = sample_clips_from_csv(args.csv, args.sample_from_csv, args.seed, args.audio_dir)
        print(f"sampled {len(paths)} clips from {args.csv} (seed={args.seed})")
    else:
        paths = args.paths
    if not paths:
        p.error("provide paths or --sample-from-csv N")
    if args.benchmark_all:
        benchmark(paths, list(_BACKENDS))
        return 0
    if args.benchmark:
        detect_many_timed(paths, args.backend, args.batch_size)
        return 0
    for path in paths:
        print(f"{path}\t{detect(path, args.backend)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
