# AnimeSR native host

The host runs AnimeSR v2 as a fixed FP16 TensorRT pipeline:

`1280x720 RGBA -> AnimeSR x4 -> 5120x2880 AV1/fMP4`

`1920x1080 RGBA -> AnimeSR x4 -> 7680x4320 AV1/fMP4`

It does not resize the AnimeSR output. Native Messaging is used only to start
the host and obtain a token-authenticated `127.0.0.1` WebSocket endpoint.

## Local installation

The current Windows beta reuses an existing compatible Python/TensorRT runtime
and locally built 720p and 1080p engines. It does not import application code
from that runtime.

```powershell
powershell -ExecutionPolicy Bypass -File .\native\install.ps1
```

Pass `-ExtensionId`, `-PythonPath`, or `-ModelDirectory` when your unpacked
extension ID or runtime location differs. Restart Edge and reload
`dist-chrome` after installation.

Run the protocol check without loading CUDA:

```powershell
python .\native\animesr_host.py --self-test
```

Run the CUDA/TensorRT/AV1 end-to-end smoke tests after copying both engines into
`native/models` or setting `ANIMESR_MODEL_DIR`:

```powershell
python .\native\smoke_test.py
python .\native\smoke_test.py --width 1920 --height 1080
```
