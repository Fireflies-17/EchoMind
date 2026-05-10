from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .utils import slugify


@dataclass(frozen=True)
class RunPaths:
    data_dir: Path
    run_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "run_id", slugify(str(self.run_id)))

    @classmethod
    def from_input(
        cls,
        input_path: str | Path,
        data_dir: str | Path = "data",
        run_id: str | None = None,
    ) -> "RunPaths":
        source = Path(input_path)
        if run_id is None:
            run_id = source.stem
        return cls(data_dir=Path(data_dir), run_id=slugify(run_id))

    @property
    def root(self) -> Path:
        return self.data_dir / "runs" / self.run_id

    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

    @property
    def chunks_dir(self) -> Path:
        return self.audio_dir / "chunks"

    @property
    def transcript_dir(self) -> Path:
        return self.root / "transcript"

    @property
    def diarization_dir(self) -> Path:
        return self.root / "diarization"

    @property
    def summary_dir(self) -> Path:
        return self.root / "summary"

    @property
    def vector_db_dir(self) -> Path:
        return self.root / "vector_db"

    @property
    def input_meta(self) -> Path:
        return self.root / "input_source.json"

    @property
    def audio_wav(self) -> Path:
        return self.audio_dir / f"{self.run_id}.wav"

    @property
    def vad_json(self) -> Path:
        return self.transcript_dir / f"{self.run_id}_vad.json"

    @property
    def asr_json(self) -> Path:
        return self.transcript_dir / f"{self.run_id}_asr.json"

    @property
    def speakers_json(self) -> Path:
        return self.diarization_dir / f"{self.run_id}_speakers.json"

    @property
    def speakers_raw_json(self) -> Path:
        return self.diarization_dir / f"{self.run_id}_speakers_raw.json"

    @property
    def timeline_json(self) -> Path:
        return self.transcript_dir / f"{self.run_id}_timeline.json"

    @property
    def summary_json(self) -> Path:
        return self.summary_dir / f"{self.run_id}_summary.json"

    @property
    def kb_db(self) -> Path:
        return self.vector_db_dir / "qdrant"

    @property
    def kb_meta(self) -> Path:
        return self.vector_db_dir / f"{self.run_id}_meta.json"

    def ensure(self) -> None:
        for path in [
            self.audio_dir,
            self.chunks_dir,
            self.transcript_dir,
            self.diarization_dir,
            self.summary_dir,
            self.vector_db_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
