# TikTok Live TTS + Gemma 4 Direct-Audio Moderation

The pipeline synthesizes each comment locally with Kokoro 82M, sends that exact
waveform to `google/gemma-4-E4B-it`, and either plays it or blocks the viewer.
There is no speech-to-text stage or OpenAI-compatible text-model server.

## Pipeline

```text
TikTok chat -> normalize -> Kokoro 82M PCM -> Gemma 4 E4B audio decision
                                             | safe  -> play same PCM
                                             ` block -> drop + block viewer
```

The classifier blocks only two categories:

1. Clear CSAM possession/access/request/offer claims.
2. Racism, including deliberate spoken racist near-slurs such as “Meggar.”

Everything else is allowed. The complete source-of-truth policy is in the
sibling fine-tuning repo's `requirements.md`. Unexpected output and model errors
fail open, so they can never create an unapproved mute or block.

## Setup

Requires an Apple Silicon Mac, Python 3.11 or 3.12, Node 18+, and access to the
gated Gemma 4 model on Hugging Face. Kokoro runs locally through MLX-Audio.

```bash
python3 -m venv .venv
./.venv/bin/python3 -m pip install -r requirements.txt
npm install
hf auth login
```

Create `config.yaml` (it is intentionally gitignored):

```yaml
tiktok:
  username: streamer_name
  session_id: ""
  tt_target_idc: ""

tts:
  model: mlx-community/Kokoro-82M-bf16
  voice: af_heart
  language_code: a
  speed: 1.0
  followers_only: false

gemma:
  model: google/gemma-4-E4B-it
  adapter_path: ../tiktok-lm-mod-finetune/trained_model/gemma4_audio_adapter
  revision: null
  device: auto
  max_new_tokens: 32

moderation:
  enabled: true
  mute_duration_seconds: 300

audio:
  output_device: null

db:
  path: moderation.db

logging:
  level: INFO
```

`mute_duration_seconds` remains only for compatibility with the TikTok moderator;
the Gemma policy never emits `mute`.

## Run

```bash
python3 -m src.main --config config.yaml
```

The first run downloads Gemma 4 and the Kokoro 82M BF16 checkpoint. Startup
loads and silently warms Kokoro before connecting to TikTok, so a missing or
incompatible model cannot fail for the first live comment. `af_heart` is the
recommended default American English voice. Every American English voice shipped
with Kokoro 1.0 is accepted by `tts.voice`; use `python3 -m test_tts --voice`
to compare them. `tts.speed` accepts `0.5` through `2.0`.

Events go to SQLite:

```sql
SELECT ts, unique_id, raw_comment, safe, punishment, reason
FROM events ORDER BY ts DESC;
```

## Failure behavior

- Unexpected pipeline runner exception: log the failure, drain pending messages,
  close TTS, shut down the bridge, and close the event log; exit with status `1`.
  Normal runner completion and handled SIGINT/SIGTERM shutdown exit with `0`.
  A `KeyboardInterrupt` caught by `main()` exits with `130`.
- Kokoro unavailable or its warmup fails: abort startup before connecting to
  TikTok; there is no fallback speech engine.
- Kokoro fails on one comment, returns invalid samples, or produces more than 30
  seconds of audio: drop that comment without playing it.
- Gemma unavailable, timeout-equivalent failure, or inference exception: allow
  and play the already-synthesized audio; record `model_error`.
- Empty or malformed JSON: allow and play the audio.
- Any schema other than exact `{"safe":false,"punishment":"block"}`: allow.
- TTS produces no audio: no-op.
- TikTok block dispatch fails: drop unsafe audio and log `action_ok=0`.
- Unknown punishment passed to the moderator: refuse it; never default to mute.

TikTok mutations use unofficial webcast endpoints exposed by the Node bridge.
Without an authenticated session, the block attempt can fail even though the
unsafe audio is still dropped locally.

## Tests

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q src tests
```

The unit tests do not download model weights. Real model checks are explicit:

```bash
python3 -m test_tts --config config.yaml --text "Hello from Kokoro."
python3 -m test_tts --config config.yaml --no-play
python3 -m test_pipeline --config config.yaml
```

`test_tts` reports load time, synthesis time, audio duration, and real-time
factor before playing the clip. `test_pipeline.py` requires a configured Gemma
adapter and exercises the complete Kokoro PCM to Gemma moderation path.
