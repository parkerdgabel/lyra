# IMAGE

| Function | Usage | Summary |
|---|---|---|
| `ImageCanvas` | `ImageCanvas[opts]` | Create a blank canvas (PNG) |
| `ImageConvert` | `ImageConvert[input, format, opts]` | Convert image format |
| `ImageCrop` | `ImageCrop[input, opts]` | Crop image by rect or gravity |
| `ImageDecode` | `ImageDecode[input, opts]` | Decode image to raw or reencoded bytes |
| `ImageEncode` | `ImageEncode[input, encoding]` | Encode raw pixels or reencode bytes |
| `ImageInfo` | `ImageInfo[input, opts]` | Read basic image info |
| `ImagePad` | `ImagePad[input, opts]` | Pad image to target size |
| `ImageResize` | `ImageResize[input, opts]` | Resize image (contain/cover) |
| `ImageThumbnail` | `ImageThumbnail[input, opts]` | Create thumbnail (cover) |
| `ImageTransform` | `ImageTransform[input, pipeline]` | Apply pipeline of operations |
| `MediaThumbnail` | `MediaThumbnail[input, opts]` | Extract video frame as image |

## `ImageCanvas`

- Usage: `ImageCanvas[opts]`
- Summary: Create a blank canvas (PNG)
- Tags: image
- Examples:
  - `ImageCanvas[<|Width->64, Height->64, Bg->"#ffffff"|>]`

## `ImageInfo`

- Usage: `ImageInfo[input, opts]`
- Summary: Read basic image info
- Tags: image
- Examples:
  - `ImageInfo[<|Path->"in.png"|>]`

## `ImageResize`

- Usage: `ImageResize[input, opts]`
- Summary: Resize image (contain/cover)
- Tags: image, transform
- Examples:
  - `ImageResize[<|Path->"in.png"|>, <|Width->128|>]`
