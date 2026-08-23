# AnimeSR native host

The host runs AnimeSR v2 as a fixed FP16 TensorRT pipeline:

`1280x720 RGBA -> AnimeSR x4 -> 5120x2880 AV1/fMP4`

It does not resize the AnimeSR output. Native Messaging is used only to start
the host and obtain a token-authenticated `127.0.0.1` WebSocket endpoint.

## Local installation

The current Windows beta reuses an existing compatible Python/TensorRT runtime
and a locally built 720p engine. It does not import application code from that
runtime.

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

Run the CUDA/TensorRT/AV1 end-to-end smoke test after copying the engine into
`native/models` or setting `ANIMESR_MODEL_DIR`:

```powershell
python .\native\smoke_test.py
```
