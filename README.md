# Mic Madness ReelKit

Turns an artist's raw performance video and their song MP3 into a finished Instagram reel:
- The MP3 is lined up to the video (Shazam-style matching, checked to within 45 ms).
- The video's own sound is muted.
- A warm color grade goes on last, under the text.
- A white pill text hook, in one of 3 fonts.
- Lyric captions, 1 word at a time, timed to the song.

Artists use it through the prompt in `PROMPT.md`, pasted into Claude (Opus 5.5) on claude.ai or in Claude Code.

## Files
- `reelkit.py`: the editor (subcommands `check`, `sync`, `captions`, `preview`, `render`).
- `model/w2v_base_int8.onnx`: the lyric timing model (95 MB).

## Licenses
- The lyric timing model is wav2vec 2.0 Base 960h by Facebook AI Research, from fairseq / torchaudio (MIT License), exported to ONNX and quantized to int8.
- Fonts are pulled at run time from Fontsource (Montserrat, Anton, Nunito; SIL Open Font License).
- Emoji: Apple Color Emoji on a Mac, otherwise Twemoji (CC-BY 4.0).
