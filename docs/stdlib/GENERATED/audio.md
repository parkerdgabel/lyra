# AUDIO

| Function | Usage | Summary |
|---|---|---|
| `AudioChannelMix` | `AudioChannelMix[input, opts]` | Convert channel count (mono/stereo) |
| `AudioConcat` | `AudioConcat[inputs, opts]` | Concatenate multiple inputs |
| `AudioConvert` | `AudioConvert[input, format, opts]` | Convert audio to WAV |
| `AudioDecode` | `AudioDecode[input, opts]` | Decode audio to raw (s16le) or WAV |
| `AudioEncode` | `AudioEncode[raw, opts]` | Encode raw PCM to WAV |
| `AudioFade` | `AudioFade[input, opts]` | Fade in/out |
| `AudioGain` | `AudioGain[input, opts]` | Apply gain in dB or linear |
| `AudioInfo` | `AudioInfo[input]` | Probe audio metadata |
| `AudioResample` | `AudioResample[input, opts]` | Resample to new sample rate |
| `AudioTrim` | `AudioTrim[input, opts]` | Trim audio by time range |
| `MediaExtractAudio` | `MediaExtractAudio[input, opts]` | Extract audio track to format |

## `AudioConvert`

- Usage: `AudioConvert[input, format, opts]`
- Summary: Convert audio to WAV
- Examples:
  - `wav := AudioConvert[<|"Path"->"ding.mp3"|>, "wav"]  ==> base64url`

## `AudioInfo`

- Usage: `AudioInfo[input]`
- Summary: Probe audio metadata
- Examples:
  - `AudioInfo[wav]  ==> <|"sampleRate"->..., "channels"->...|>`
