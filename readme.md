# Convert Audio to Text with Speakers

Transcribes audio files to text with speaker diarization — each line is labeled with a speaker ID and timestamp.

## How it works

1. Reads audio files from `input/`
2. Transcribes speech using [Whisper](https://github.com/openai/whisper)
3. Extracts speaker embeddings using [SpeechBrain](https://speechbrain.github.io/) (ECAPA-TDNN)
4. Clusters speakers via Agglomerative Clustering (auto-detects 2–8 speakers, or fixed count)
5. Merges consecutive segments from the same speaker into one line
6. Saves transcript to `output/<filename>.txt`
7. Moves processed files to `done/`, failed files to `failed/`

## Output format

```
[0.0s - 5.2s] Speaker 1: Hello, welcome to the meeting.
[5.3s - 10.1s] Speaker 2: Thanks for having me.
```

## Requirements

- Python 3.8+
- [ffmpeg](https://ffmpeg.org/) (must be in PATH)
- Python packages:

```bash
pip install openai-whisper speechbrain scikit-learn soundfile numpy torch
```

## Usage

1. Put audio files in `input/`
2. Run:

```bash
python convert.py
```

3. Find transcripts in `output/`

## Supported formats

`.mp3`, `.mp4`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.webm`

## Configuration

Edit the constants at the top of [convert.py](convert.py):

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `base` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) |
| `NUM_SPEAKERS` | `None` | Fix speaker count, or `None` to auto-detect (2–8) |

## Directory structure

```
input/      # Put audio files here
output/     # Transcripts saved here
done/       # Successfully processed audio files
failed/     # Audio files that failed to process
pretrained_models/  # Auto-downloaded speaker model
```
