"""Streaming scheduler: text segments -> per-segment GenerateRequests -> PCM chunks, with pauses between segments.

Backend calls run on a dedicated worker thread (`BackendWorker`), so MLX state stays on the thread that created it and
concurrent callers are serialized. Backends that set `concurrent = True` get one producer thread per stream instead.
Consumers read chunks from a bounded queue, so a slow HTTP client applies back-pressure instead of letting audio pile up
in memory. Closing the consumer cancels generation at the next chunk.
"""
from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Optional

import numpy as np

from .backends.base import GenerateRequest, SegmentInfo, Voice
from .defaults import DEFAULT_SAMPLING, FIRST_CHUNK_FRAMES, LANGUAGE, SAMPLE_RATE, STREAM_CHUNK_S, FRAME_RATE, Sampling, max_frames_for
from .prompt import build_prompt_ids
from .text import Segment

PAUSES = {"sentence": 0.25, "clause": 0.1, "paragraph": 0.45, "end": 0.0}


@dataclass
class StreamOptions:
    instruct: Optional[str] = None
    sampling: Sampling = field(default_factory=lambda: DEFAULT_SAMPLING)
    seed: Optional[int] = None  # segment i uses seed + i
    chunk_s: float = STREAM_CHUNK_S
    first_chunk_frames: int = FIRST_CHUNK_FRAMES
    language: str = LANGUAGE
    pauses: dict = field(default_factory=lambda: dict(PAUSES))
    stream: bool = True
    mode: str = "icl"  # icl | xvec | custom_voice | voice_design (resolved by Lure from the model and arguments)
    layout: str = "non_streaming"
    speaker: Optional[str] = None  # preset speaker name, custom_voice only
    speaker_id: Optional[int] = None


@dataclass
class Event:
    """What the scheduler yields: audio (pcm) or a finished segment (info)."""
    pcm: np.ndarray
    segment: Optional[Segment] = None
    info: Optional[SegmentInfo] = None


def make_request(tok, seg: Segment, voice: Optional[Voice], opt: StreamOptions) -> GenerateRequest:
    s = opt.sampling
    ref_text = voice.ref_text if opt.mode == "icl" else None
    return GenerateRequest(
        text=seg.text, voice=voice, prompt=build_prompt_ids(tok, seg.text, ref_text, opt.instruct),
        instruct=opt.instruct if opt.instruct and opt.instruct.strip() else None, sampling=s,
        seed=None if opt.seed is None else opt.seed + seg.index,
        max_new_tokens=s.max_new_tokens or max_frames_for(seg.text), stream=opt.stream,
        chunk_frames=max(1, int(round(opt.chunk_s * FRAME_RATE))), first_chunk_frames=opt.first_chunk_frames,
        language=opt.language, mode=opt.mode, layout=opt.layout, speaker=opt.speaker, speaker_id=opt.speaker_id,
    )


def run_segments(backend, tok, segments: Iterable[Segment], voice: Optional[Voice], opt: StreamOptions,
                 cancelled: Callable[[], bool] = lambda: False) -> Iterator[Event]:
    """Synchronous core loop (call it on the backend thread)."""
    for seg in segments:
        if cancelled():
            return
        req = make_request(tok, seg, voice, opt)
        gen = backend.generate(req)
        try:
            for ch in gen:
                if cancelled():
                    return
                if ch.pcm is not None and len(ch.pcm):
                    yield Event(np.asarray(ch.pcm, dtype=np.float32), seg)
                if ch.info is not None:
                    yield Event(np.zeros(0, np.float32), seg, ch.info)
        finally:
            close = getattr(gen, "close", None)
            if close:
                close()
        gap = opt.pauses.get(seg.kind, 0.0)
        if gap > 0:
            yield Event(np.zeros(int(gap * SAMPLE_RATE), np.float32), seg)


_END = object()


class BackendWorker:
    """Runs backend calls off the caller's thread.

    Default (concurrent=False): every call and every stream runs on ONE worker thread, serialized. Required by mlx (thread-local
    streams/state) and safe for any backend. concurrent=True (the backend class sets `concurrent = True`, e.g. cuda, which
    batches simultaneous requests in one decode loop): each stream gets its own producer thread so generate() calls overlap;
    load / encode_voice / close still go through the single worker.
    """

    def __init__(self, concurrent: bool = False):
        self.ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lure-backend")
        self.lock = threading.Lock()
        self.concurrent = concurrent
        self._n = 0

    def call(self, fn, *a, **kw):
        return self.ex.submit(fn, *a, **kw).result()

    def stream(self, make_iter: Callable[[Callable[[], bool]], Iterator], maxsize: int = 256) -> Iterator:
        """Run make_iter(cancelled) on the worker thread; yield its items here. Closing this generator cancels it."""
        q: queue.Queue = queue.Queue(maxsize=maxsize)
        cancel = threading.Event()

        def produce():
            try:
                for item in make_iter(cancel.is_set):
                    while not cancel.is_set():
                        try:
                            q.put(item, timeout=0.1)
                            break
                        except queue.Full:
                            continue
                    if cancel.is_set():
                        break
            except BaseException as e:  # forward errors to the consumer
                q.put(e)
            finally:
                q.put(_END)

        if self.concurrent:
            with self.lock:
                self._n += 1
                name = f"lure-stream-{self._n}"
            th = threading.Thread(target=produce, name=name, daemon=True)
            th.start()
            done = lambda: not th.is_alive()  # noqa: E731
        else:
            fut = self.ex.submit(produce)
            done = fut.done
        try:
            while True:
                item = q.get()
                if item is _END:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            cancel.set()
            while not done():  # drain so the producer can see the cancel and exit
                try:
                    q.get(timeout=0.05)
                except queue.Empty:
                    pass

    def shutdown(self):
        self.ex.shutdown(wait=True)
