# YOLO-World Video + JSON Timeline (Replicate / Cog)

Menerima satu video, mendeteksi objek open-vocabulary (YOLO-World v2),
memberi track ID (ByteTrack/BoT-SORT), lalu mengeluarkan **dua file** dalam
satu prediction: `output.mp4` beranotasi dan `timeline.json` per-frame.

Lihat `PLAN.md` untuk desain lengkap.

## Struktur

- `run.py` — Cog `BaseRunner` + skema input/output
- `pipeline/` — video_io (decode/encode/audio), detector, tracker, annotate, timeline, process
- `tests/` — smoke tests tanpa GPU
- `weights/` — `yolov8s-worldv2.pt` (unduh sebelum build, Ă§ek checksum)

## Menjalankan lokal

```bash
cog build
cog run -i video=@tests/fixtures/short.mp4 -i classes="person,car" \
  -i confidence=0.25 -i tracker=bytetrack
```

## API Replicate setelah push

```bash
curl -X POST \
  "https://api.replicate.com/v1/models/YOUR_USERNAME/yolo-world-video-timeline/predictions" \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input": {"video": "https://example.com/sample.mp4", "classes": "person,car"}}'
```

Output: `{"video": "<url output.mp4>", "timeline": "<url timeline.json>"}`

Output prediction Kadaluarsa ~1 jam (Replicate retention) — salin ke storage
sendiri jika perlu permanen.

## Batas MVP

- Maks 60 detik, 1080p; fps input >30 dinormalisasi ke 30 (CFR).
- `classes` = string dipisah koma, open-vocabulary (tanpa fine-tuning).
- Track ID scoped per prediction, tidak dijamin stabil saat oklusi panjang.
