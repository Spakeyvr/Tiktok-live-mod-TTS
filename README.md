# TikTok Live TTS moderation pipeline

Local pipeline that reads TikTok Live chat, runs each message through TTS, transcribes the synthesized audio, asks a local LLM whether the *spoken* form is safe, and either plays the audio or punishes the offender (mute / block) and drops the audio. Everything runs locally; the only network dependency is your local LLM endpoint and the TikTok WebSocket itself.

## Pipeline

```
TikTok chat (Node bridge)
   └─> Python (asyncio)
        ├─ HomoglyphNormalizer  (NFKC + zero-width strip + confusables)
        ├─ RealtimeTTS          (Kokoro or Piper) -> float32 PCM buffer
        ├─ faster-whisper       (Distil-Whisper)  -> transcript
        ├─ OpenAI-compat LLM    -> {"safe": bool, "punishment": "mute"|"block"}
        └─ Moderator + AudioPlayer + SQLite event log
```

Each stage is a thin class behind a `Protocol`, so swap any of them by editing `src/main.py`.

## Setup

Requires Python 3.11+ and Node 18+.

```bash
pip install -r requirements.txt
npm install
```

The first TTS run downloads the Kokoro voice. The first STT run downloads the Distil-Whisper weights.

Start a local LLM that exposes an OpenAI-compatible endpoint. Any of these work — only the URL and model name change:

| Backend  | `base_url`                          | `model` example      |
|----------|-------------------------------------|----------------------|
| Ollama   | `http://localhost:11434/v1`         | `qwen3.5:4b`         |
| LM Studio| `http://localhost:1234/v1`          | `qwen3.5-4b-instruct`|
| oMLX     | `http://localhost:8080/v1`          | `qwen3.5-4b`         |

## Configuration

Edit `config.yaml`. The important knobs:

- `tiktok.username` — host you want to monitor (no `@`).
- `tiktok.session_id` / `tiktok.tt_target_idc` — required only if you want the moderator to actually mute/block on TikTok. Without them the bridge logs the intent but the action will fail (TikTok-Live-Connector does not expose unauthenticated moderation).
- `tts.engine` — `kokoro` or `piper`. For Piper, set `piper_model` to a `.onnx` voice file.
- `llm.base_url` / `llm.model` — point at your local OpenAI-compatible server.
- `moderation.blocklist` — hard local backstop; substring-matched (case-insensitive) against the STT transcript.

## Run

```bash
python -m src.main --config config.yaml
```

Events go to `moderation.db` (SQLite) — schema in `src/db.py`. Inspect with any sqlite browser:

```sql
SELECT ts, unique_id, raw_comment, transcript, safe, punishment FROM events ORDER BY ts DESC;
```

## Notes on moderation actions

TikTok-Live-Connector is primarily a read client. The bridge attempts the unofficial webcast HTTP endpoints (`/webcast/room/mute_user/`, `/webcast/room/block_user/`) when an authenticated session is provided. If your version of the connector doesn't expose `webcastHttpClient` / `postFormDataToWebcastApi`, the bridge will emit `{"ok": false, "reason": "moderation api not exposed..."}` and the pipeline still logs the event to SQLite. Replace `bridge/tiktok_bridge.js` with a TikTok Studio API client if you need guaranteed mutations.

## Failure modes (by design)

- **LLM unreachable** — fail closed: `safe=false`, `punishment=mute`, audio dropped, reason recorded as `llm_error: …`.
- **LLM returns malformed JSON** — same: fail closed with mute.
- **TTS produces empty audio** — message skipped, nothing logged (treated as no-op).
- **Moderator dispatch fails** — event still logged with `action_ok=0`.

## Layout

```
config.yaml
package.json
requirements.txt
bridge/tiktok_bridge.js          Node TikTok-Live-Connector wrapper (NDJSON over stdio)
src/
  main.py                        entry point
  pipeline.py                    per-message orchestration
  config.py                      typed config loader
  bridge.py                      Python side of the Node bridge
  normalizer.py                  unicode + confusables
  tts.py                         RealtimeTTS -> PCM buffer
  stt.py                         faster-whisper / Distil-Whisper
  llm_filter.py                  OpenAI-compatible safety filter
  moderator.py                   mute / block dispatch
  audio.py                       sounddevice playback
  db.py                          SQLite event log
```
