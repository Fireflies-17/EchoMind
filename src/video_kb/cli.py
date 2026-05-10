from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .utils import DependencyError, save_json


def cmd_extract_audio(args: argparse.Namespace) -> None:
    from .audio import extract_audio

    output = extract_audio(args.input, args.output, ffmpeg_bin=args.ffmpeg, sample_rate=args.sample_rate)
    print(output)


def cmd_asr(args: argparse.Namespace) -> None:
    from .asr import run_asr

    payload = run_asr(
        args.audio,
        args.output,
        vad_output_path=args.vad_output,
        chunks_dir=args.chunks_dir,
        device=args.device,
        language=args.language,
        vad_model=args.vad_model,
        asr_model=args.asr_model,
        max_segment_ms=args.max_segment_ms,
        batch_size_s=args.batch_size_s,
    )
    print(json.dumps({"segments": len(payload["segments"]), "output": args.output}, ensure_ascii=False))


def cmd_diarize(args: argparse.Namespace) -> None:
    from .diarization import diarize

    payload = diarize(
        args.audio,
        args.output,
        token=args.token,
        model_name=args.model,
        device=args.device,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        backend=args.backend,
        speaker_num=args.speaker_num,
        include_overlap=args.include_overlap,
        repo_path=args.threed_speaker_repo,
        model_cache_dir=args.model_cache_dir,
        python_exe=args.threed_speaker_python,
    )
    print(json.dumps({"segments": len(payload["segments"]), "output": args.output}, ensure_ascii=False))


def cmd_clean_diarization(args: argparse.Namespace) -> None:
    from .diarization import clean_diarization

    payload = clean_diarization(
        args.input,
        args.output,
        min_segment_ms=args.min_segment_ms,
        merge_gap_ms=args.merge_gap_ms,
        min_total_ms=args.min_total_ms,
        max_speakers=args.max_speakers,
        reassign_gap_ms=args.reassign_gap_ms,
    )
    cleaning = payload.get("cleaning", {})
    print(
        json.dumps(
            {
                "segments": len(payload.get("segments", [])),
                "speakers": cleaning.get("speakers_after", []),
                "dropped_short_segments": cleaning.get("dropped_short_segments", 0),
                "reassigned_segments": cleaning.get("reassigned_segments", 0),
                "output": args.output,
            },
            ensure_ascii=False,
        )
    )


def cmd_merge(args: argparse.Namespace) -> None:
    from .timeline import merge_timeline

    timeline = merge_timeline(args.asr, args.speakers, args.output, merge_adjacent=not args.no_merge_adjacent)
    print(json.dumps({"segments": len(timeline), "output": args.output}, ensure_ascii=False))


def cmd_summarize(args: argparse.Namespace) -> None:
    from .summarize import summarize_timeline

    summary = summarize_timeline(
        args.timeline,
        args.output,
        engine=args.engine,
        prompt_output_path=args.prompt_output,
    )
    print(
        json.dumps(
            {
                "knowledge_points": len(summary.get("knowledge_points", [])),
                "action_items": len(summary.get("action_items", [])),
                "output": args.output,
            },
            ensure_ascii=False,
        )
    )


def cmd_build_kb(args: argparse.Namespace) -> None:
    from .kb import build_kb

    meta = build_kb(
        args.summary,
        args.db,
        collection_name=args.collection,
        backend=args.backend,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        fallback_hash=not args.no_fallback,
        source_file=args.source_file,
        meta_output_path=args.meta_output,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def cmd_query(args: argparse.Namespace) -> None:
    from .kb import query_kb

    results = query_kb(args.db, args.query, limit=args.limit, meta_path=args.meta)
    print(json.dumps(results, ensure_ascii=False, indent=2))


def cmd_report(args: argparse.Namespace) -> None:
    from .reports import write_report

    report = write_report(
        args.timeline,
        args.output,
        mode=args.mode,
        speaker=args.speaker,
        engine=args.engine,
        output_format=args.format,
        topic_gap_ms=args.topic_gap_ms,
        topic_window_ms=args.topic_window_ms,
        min_topic_ms=args.min_topic_ms,
        max_llm_chars=args.max_llm_chars,
    )
    count_key = "topics" if args.mode == "topics" else "segments"
    print(
        json.dumps(
            {
                "mode": args.mode,
                "items": len(report.get(count_key, [])),
                "output": args.output,
            },
            ensure_ascii=False,
        )
    )


def cmd_run(args: argparse.Namespace) -> None:
    from .asr import run_asr
    from .audio import extract_audio
    from .diarization import clean_diarization, diarize, write_skipped_diarization
    from .kb import build_kb
    from .paths import RunPaths
    from .summarize import summarize_timeline
    from .timeline import merge_timeline

    input_path = Path(args.input)
    paths = RunPaths.from_input(input_path, data_dir=args.data_dir, run_id=args.run_id)
    paths.ensure()
    save_json({"input": str(input_path.resolve()), "run_id": paths.run_id}, paths.input_meta)

    extract_audio(input_path, paths.audio_wav, ffmpeg_bin=args.ffmpeg, sample_rate=args.sample_rate)
    asr_payload = run_asr(
        paths.audio_wav,
        paths.asr_json,
        vad_output_path=paths.vad_json,
        chunks_dir=paths.chunks_dir,
        device=args.device,
        language=args.language,
        vad_model=args.vad_model,
        asr_model=args.asr_model,
        max_segment_ms=args.max_segment_ms,
        batch_size_s=args.batch_size_s,
    )

    if args.skip_diarization:
        write_skipped_diarization(
            paths.speakers_json,
            "Skipped by --skip-diarization.",
            audio_path=paths.audio_wav,
        )
    else:
        try:
            diarization_output = paths.speakers_json if args.no_clean_diarization else paths.speakers_raw_json
            diarize(
                paths.audio_wav,
                diarization_output,
                token=args.hf_token,
                model_name=args.diarization_model,
                device=args.device,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                backend=args.diarization_backend,
                speaker_num=args.speaker_num,
                include_overlap=args.include_overlap,
                repo_path=args.threed_speaker_repo,
                model_cache_dir=args.diarization_model_cache_dir,
                python_exe=args.threed_speaker_python,
            )
            if not args.no_clean_diarization:
                clean_diarization(
                    diarization_output,
                    paths.speakers_json,
                    min_segment_ms=args.clean_min_segment_ms,
                    merge_gap_ms=args.clean_merge_gap_ms,
                    min_total_ms=args.clean_min_total_ms,
                    max_speakers=args.clean_max_speakers
                    if args.clean_max_speakers is not None
                    else args.max_speakers,
                    reassign_gap_ms=args.clean_reassign_gap_ms,
                )
        except DependencyError as exc:
            if args.require_diarization:
                raise
            write_skipped_diarization(paths.speakers_json, str(exc), audio_path=paths.audio_wav)

    timeline = merge_timeline(paths.asr_json, paths.speakers_json, paths.timeline_json)
    summary = summarize_timeline(
        paths.timeline_json,
        paths.summary_json,
        engine=args.summary_engine,
        prompt_output_path=paths.summary_json.with_name(f"{paths.run_id}_summary_prompt.txt"),
    )
    kb_meta = None
    if not args.skip_kb:
        kb_meta = build_kb(
            paths.summary_json,
            paths.kb_db,
            collection_name=args.collection,
            backend=args.kb_backend,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
            fallback_hash=not args.no_kb_fallback,
            source_file=str(input_path),
            meta_output_path=paths.kb_meta,
        )

    result = {
        "run_id": paths.run_id,
        "root": str(paths.root),
        "audio": str(paths.audio_wav),
        "vad": str(paths.vad_json),
        "asr": str(paths.asr_json),
        "asr_segments": len(asr_payload.get("segments", [])),
        "speakers": str(paths.speakers_json),
        "raw_speakers": str(paths.speakers_raw_json)
        if not args.skip_diarization and not args.no_clean_diarization
        else None,
        "timeline": str(paths.timeline_json),
        "timeline_segments": len(timeline),
        "summary": str(paths.summary_json),
        "knowledge_points": len(summary.get("knowledge_points", [])),
        "kb": str(paths.kb_db) if kb_meta else None,
        "kb_meta": str(paths.kb_meta) if kb_meta else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="video-kb", description="Local video knowledge base pipeline.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract-audio", help="Extract 16kHz mono WAV from local media.")
    extract.add_argument("input")
    extract.add_argument("output")
    extract.add_argument("--ffmpeg", default="ffmpeg")
    extract.add_argument("--sample-rate", type=int, default=16000)
    extract.set_defaults(func=cmd_extract_audio)

    asr = subparsers.add_parser("asr", help="Run fsmn-vad plus SenseVoiceSmall ASR.")
    asr.add_argument("audio")
    asr.add_argument("output")
    asr.add_argument("--vad-output")
    asr.add_argument("--chunks-dir")
    asr.add_argument("--device", default="auto")
    asr.add_argument("--language", default="zh")
    asr.add_argument("--vad-model", default="fsmn-vad")
    asr.add_argument("--asr-model", default="iic/SenseVoiceSmall")
    asr.add_argument("--max-segment-ms", type=int, default=30000)
    asr.add_argument("--batch-size-s", type=int, default=60)
    asr.set_defaults(func=cmd_asr)

    diar = subparsers.add_parser("diarize", help="Run speaker diarization.")
    diar.add_argument("audio")
    diar.add_argument("output")
    diar.add_argument("--backend", choices=["pyannote", "3dspeaker"], default="pyannote")
    diar.add_argument("--token")
    diar.add_argument("--model", default="pyannote/speaker-diarization-community-1")
    diar.add_argument("--device", default="auto")
    diar.add_argument("--min-speakers", type=int)
    diar.add_argument("--max-speakers", type=int)
    diar.add_argument("--speaker-num", type=int, help="Exact speaker count for 3D-Speaker.")
    diar.add_argument("--include-overlap", action="store_true", help="Enable 3D-Speaker overlap detection.")
    diar.add_argument("--threed-speaker-repo", help="Path to a cloned modelscope/3D-Speaker repository.")
    diar.add_argument("--threed-speaker-python", help="Python executable from an isolated 3D-Speaker environment.")
    diar.add_argument("--model-cache-dir", help="Model cache directory for 3D-Speaker.")
    diar.set_defaults(func=cmd_diarize)

    clean_diar = subparsers.add_parser("clean-diarization", help="Clean diarization fragments before timeline merge.")
    clean_diar.add_argument("input")
    clean_diar.add_argument("output")
    clean_diar.add_argument("--min-segment-ms", type=int, default=800)
    clean_diar.add_argument("--merge-gap-ms", type=int, default=500)
    clean_diar.add_argument("--min-total-ms", type=int, default=0)
    clean_diar.add_argument("--max-speakers", type=int)
    clean_diar.add_argument("--reassign-gap-ms", type=int, default=3000)
    clean_diar.set_defaults(func=cmd_clean_diarization)

    merge = subparsers.add_parser("merge", help="Merge ASR and speaker timelines.")
    merge.add_argument("asr")
    merge.add_argument("speakers")
    merge.add_argument("output")
    merge.add_argument("--no-merge-adjacent", action="store_true")
    merge.set_defaults(func=cmd_merge)

    summarize = subparsers.add_parser("summarize", help="Create structured summary JSON.")
    summarize.add_argument("timeline")
    summarize.add_argument("output")
    summarize.add_argument("--engine", choices=["auto", "llm", "heuristic"], default="auto")
    summarize.add_argument("--prompt-output")
    summarize.set_defaults(func=cmd_summarize)

    kb = subparsers.add_parser("build-kb", help="Build local vector knowledge base from summary JSON.")
    kb.add_argument("summary")
    kb.add_argument("db")
    kb.add_argument("--collection", default="video_knowledge")
    kb.add_argument("--backend", choices=["qdrant", "json", "milvus"], default="qdrant")
    kb.add_argument("--embedding-provider", choices=["sentence-transformers", "hash"], default="sentence-transformers")
    kb.add_argument("--embedding-model")
    kb.add_argument("--source-file")
    kb.add_argument("--meta-output")
    kb.add_argument("--no-fallback", action="store_true")
    kb.set_defaults(func=cmd_build_kb)

    query = subparsers.add_parser("query", help="Query the knowledge base.")
    query.add_argument("--db", required=True)
    query.add_argument("--query", required=True)
    query.add_argument("--limit", type=int, default=5)
    query.add_argument("--meta")
    query.set_defaults(func=cmd_query)

    report = subparsers.add_parser("report", help="Generate full, speaker, or topic reports from a timeline JSON.")
    report.add_argument("timeline")
    report.add_argument("output")
    report.add_argument("--mode", choices=["full", "speaker", "topics"], required=True)
    report.add_argument("--speaker", help="Speaker label for --mode speaker, for example SPEAKER_01 or 1.")
    report.add_argument("--engine", choices=["auto", "llm", "heuristic"], default="auto")
    report.add_argument("--format", choices=["json", "md"], default="json")
    report.add_argument("--topic-gap-ms", type=int, default=45000)
    report.add_argument("--topic-window-ms", type=int, default=5 * 60 * 1000)
    report.add_argument("--min-topic-ms", type=int, default=45000)
    report.add_argument("--max-llm-chars", type=int, default=24000)
    report.set_defaults(func=cmd_report)

    run = subparsers.add_parser("run", help="Run the full pipeline.")
    run.add_argument("--input", required=True)
    run.add_argument("--run-id", help="Output run id. Defaults to the input file name without extension.")
    run.add_argument("--data-dir", default="data")
    run.add_argument("--ffmpeg", default="ffmpeg")
    run.add_argument("--sample-rate", type=int, default=16000)
    run.add_argument("--device", default="auto")
    run.add_argument("--language", default="zh")
    run.add_argument("--vad-model", default="fsmn-vad")
    run.add_argument("--asr-model", default="iic/SenseVoiceSmall")
    run.add_argument("--max-segment-ms", type=int, default=30000)
    run.add_argument("--batch-size-s", type=int, default=60)
    run.add_argument("--skip-diarization", action="store_true")
    run.add_argument("--require-diarization", action="store_true")
    run.add_argument("--hf-token")
    run.add_argument("--diarization-backend", choices=["pyannote", "3dspeaker"], default="pyannote")
    run.add_argument("--diarization-model", default="pyannote/speaker-diarization-community-1")
    run.add_argument("--min-speakers", type=int)
    run.add_argument("--max-speakers", type=int)
    run.add_argument("--speaker-num", type=int, help="Exact speaker count for 3D-Speaker.")
    run.add_argument("--include-overlap", action="store_true", help="Enable 3D-Speaker overlap detection.")
    run.add_argument("--threed-speaker-repo", help="Path to a cloned modelscope/3D-Speaker repository.")
    run.add_argument("--threed-speaker-python", help="Python executable from an isolated 3D-Speaker environment.")
    run.add_argument("--diarization-model-cache-dir", help="Model cache directory for 3D-Speaker.")
    run.add_argument("--no-clean-diarization", action="store_true")
    run.add_argument("--clean-min-segment-ms", type=int, default=800)
    run.add_argument("--clean-merge-gap-ms", type=int, default=500)
    run.add_argument("--clean-min-total-ms", type=int, default=0)
    run.add_argument("--clean-max-speakers", type=int)
    run.add_argument("--clean-reassign-gap-ms", type=int, default=3000)
    run.add_argument("--summary-engine", choices=["auto", "llm", "heuristic"], default="auto")
    run.add_argument("--skip-kb", action="store_true")
    run.add_argument("--collection", default="video_knowledge")
    run.add_argument("--kb-backend", choices=["qdrant", "json", "milvus"], default="qdrant")
    run.add_argument("--embedding-provider", choices=["sentence-transformers", "hash"], default="sentence-transformers")
    run.add_argument("--embedding-model")
    run.add_argument("--no-kb-fallback", action="store_true")
    run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
