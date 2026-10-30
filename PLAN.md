# PLAN — YOLO-World Video + JSON Timeline di Replicate (Cog)

> **Status:** Rencana implementasi (belum dibangun atau dideploy)  
> **Tanggal:** 29 Oktober 2026  
> **Target:** Model Replicate baru yang memakai *pretrained weights* YOLO-World, menghasilkan **video MP4 beranotasi** dan **JSON deteksi/tracking per frame** dalam **satu prediction**.  
> **Referensi inspirasi:** https://replicate.com/zsxkib/yolo-world

## 0. Ringkasan keputusan

**Ini adalah membuat model/endpoint Replicate baru secara produk dan deployment, bukan melatih arsitektur atau bobot model YOLO baru dari nol.** Model deteksi yang sudah dilatih digunakan sebagai *base*, lalu kita menambahkan pengolah video, tracker, annotator, dan serializer timeline. Nantinya dipaketkan dengan **Cog** dan dipublikasikan sebagai model Replicate milik sendiri.

**Pilihan MVP:**

| Area | Keputusan |
| --- | --- |
| Hosting/inference | Custom Replicate model dengan Cog dan GPU |
| Core detector | Ultralytics **YOLO-World v2 small** (`yolov8s-worldv2.pt`) sebagai baseline |
| Custom label | Kelas objek berupa daftar teks; tidak memerlukan fine-tuning |
| Tracking | ByteTrack default; BoT-SORT sebagai opsi lanjutan |
| Input | Satu file video per prediction, plus kelas dan opsi deteksi |
| Output utama | `output.mp4` dan `timeline.json` sebagai dua **file output** terpisah |
| Output Cog | `cog.BaseModel` berisi `video: cog.Path` dan `timeline: cog.Path` |
| Decoder/timestamp | PyAV/FFmpeg; timestamp sumber dari PTS, lalu diselaraskan dengan video output |
| Encoder | FFmpeg, H.264 + MP4 (opsional salin/encode audio) |
| Pola eksekusi | Batch asynchronous; **bukan** live CCTV pada MVP |
| Antarmuka Cog terkini | `run.py:Runner` menggunakan `BaseRunner.run()`; `predict.py:Predictor` adalah antarmuka legacy |

**Deliverable MVP:** setelah 1 video diproses, API Replicate mengembalikan objek berisi URL untuk MP4 beranotasi dan file JSON timeline yang menggambarkan setiap frame video hasil.

## 1. Goal & non-goal

### Goal

1. Menerima file MP4/MOV (dan format lain yang lolos uji decoder).
2. Mendeteksi objek menggunakan *open-vocabulary prompt*, misalnya `person, car, motorcycle`.
3. Memberi **track ID** yang relatif stabil antar-frame untuk objek yang sama.
4. Menyimpan record **semua frame output**, termasuk frame tanpa objek (`objects: []`).
5. Mengeluarkan video dengan bbox + label + confidence + track ID yang sesuai JSON.
6. Menghasilkan file `timeline.json` yang *machine-readable* dan mempunyai versi skema.
7. Dapat dijalankan lewat Cog secara lokal (dengan GPU NVIDIA) lalu dipublikasi ke Replicate.

### Non-goal MVP

- Bukan training/fine-tuning model dari nol.
- Bukan streaming RTSP, WebRTC, HLS, CCTV real-time, atau API video live.
- Bukan re-identification lintas kamera/video.
- Bukan analisis aktivitas semantik otomatis (misalnya `person is stealing`).
- Bukan jaminan track ID tidak berubah saat oklusi atau objek keluar-masuk frame.

## 2. Kontrak input/output model

### 2.1 Input prediction (rancangan)

| Parameter | Tipe Cog | Default | Keterangan |
| --- | --- | --- | --- |
| `video` | `Path` | wajib | Video yang diunggah/diakses Replicate |
| `classes` | `str` | `person,car,motorcycle,bicycle` | Daftar kelas dipisahkan koma; trim, deduplicate, dan validasi |
| `confidence` | `float` | `0.25` | Ambang deteksi, dibatasi 0.05–0.95 |
| `iou_threshold` | `float` | `0.7` | NMS IoU sesuai dukungan model |
| `tracker` | `str` | `bytetrack` | Pilihan `bytetrack` atau `botsort` |
| `show_confidence` | `bool` | `true` | Tampilan label di video |
| `show_track_id` | `bool` | `true` | Tampilan ID di video |
| `preserve_audio` | `bool` | `true` | Pertahankan audio input bila memungkinkan |
| `max_duration_seconds` | — | kebijakan server | **Bukan** parameter bebas pengguna pada MVP; ditetapkan di sisi server untuk menjaga biaya |

Catatan: `classes` dalam string lebih sederhana untuk playground Replicate. Bisa diganti `list[str]` setelah kompatibilitas SDK dan kebutuhan klien diuji.

### 2.2 Output Cog/Replicate (dua file terpisah)

```json
{
  "video": "https://replicate.delivery/.../output.mp4",
  "timeline": "https://replicate.delivery/.../timeline.json"
}
```

Nilai di atas **contoh bentuk response**, bukan URL yang valid. Kedua properti berasal dari objek output Cog (`Path`), bukan dari JSON timeline yang di-*inline* ke response.

### 2.3 Contoh isi `timeline.json`

```json
{
  "schema_version": "1.0.0",
  "video": {
    "filename": "output.mp4",
    "width": 1280,
    "height": 720,
    "fps_num": 30,
    "fps_den": 1,
    "total_frames": 300,
    "duration_ms": 10000,
    "timestamp_basis": "output_presentation_time"
  },
  "model": {
    "detector": "yolov8s-worldv2",
    "tracker": "bytetrack",
    "classes": ["person", "car"],
    "confidence_threshold": 0.25
  },
  "frames": [
    {
      "frame_index": 0,
      "timestamp_ms": 0,
      "source_timestamp_ms": 0,
      "objects": [
        {
          "track_id": 1,
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.9671,
          "bbox_xyxy": [120.0, 110.0, 360.0, 700.0]
        }
      ]
    },
    {
      "frame_index": 1,
      "timestamp_ms": 33.333,
      "source_timestamp_ms": 33.333,
      "objects": []
    }
  ]
}
```

Aturan format:

- `frame_index` dimulai dari 0 dan berurutan tanpa hilang hingga `total_frames - 1`.
- `timestamp_ms` adalah posisi presentasi pada **video hasil**; monoton naik. `source_timestamp_ms` menyimpan PTS frame input jika tersedia.
- `bbox_xyxy = [x1,y1,x2,y2]` dalam **koordinat piksel video output**; divalidasi ke dimensi video.
- `track_id` adalah integer unik **di dalam satu prediction/video**, atau `null` jika deteksi belum dapat diasosiasikan tracker.
- `class_id` menunjuk indeks `model.classes`; `class_name` berisi nama yang dapat dipakai langsung oleh aplikasi.
- `confidence` berada dalam rentang 0 sampai 1; **bukan** kepastian absolut.
- Jangan membuat deteksi palsu agar terlihat kontinu ketika objek hilang. `objects` boleh kosong.
- `total_frames`, `fps`, dan durasi **harus berasal dari output nyata**, bukan sekadar metadata container input.

### 2.4 Pilihan timeline besar

MVP mengembalikan JSON utuh. Untuk video panjang, JSON per frame bisa sangat besar; tahap berikutnya dapat menyediakan `timeline.jsonl` dan/atau output terkompresi, tanpa mengubah kontrak output utama sebelum versi API baru.

## 3. Alur arsitektur

```text
Client / Replicate Playground
        |
        | video + classes + confidence + tracker
        v
Replicate Prediction API
        |
        v
Cog Runner (run.py:Runner)
        |
        +--> Validate input: format, durasi, resolusi, file size
        |
        +--> Decode frames (PyAV/FFmpeg) + timestamps (PTS)
        |         |
        |         v
        |    YOLO-World v2: detections
        |         |
        |         v
        |    ByteTrack / BoT-SORT: track_id
        |         |
        |         +---------------------+
        |         |                     |
        |         v                     v
        |    Annotation renderer     Timeline serializer
        |         |                     |
        |         v                     v
        |    H.264 MP4 encoder       timeline.json
        |         |
        |      output.mp4
        |         |                     |
        +---------+---------------------+
                  |
                  v
         Cog Output(BaseModel)
            video: Path
            timeline: Path
                  |
                  v
         Replicate hosted file URLs
```

**Pemisahan tanggung jawab:** model hanya melakukan inference/tracking/penulisan berkas. Jika nantinya dipakai SaaS, backend aplikasi mengurus autentikasi, rate limit, job database, webhook, dan menyalin kedua hasil ke object storage milik sendiri.

## 4. Struktur repository

```text
yolo-world-video-timeline/
├── PLAN.md                    # Salinan dokumen ini
├── README.md                  # Cara pakai model dan contoh API
├── cog.yaml                   # Image, CUDA/GPU, dependencies, Cog runner
├── requirements.txt           # Python packages, versi dikunci setelah smoke test
├── run.py                     # BaseRunner + input/output schema
├── pipeline/
│   ├── __init__.py
│   ├── video_io.py            # Decoder, frame PTS, probe input, encoder, audio
│   ├── detector.py            # YOLO-World, custom classes, bbox/confidence
│   ├── tracker.py             # ByteTrack/BoT-SORT; state per prediction
│   ├── annotate.py            # Drawing bounding boxes dan labels
│   ├── timeline.py            # Streaming JSON serializer + schema version
│   └── process.py             # Orkestrasi keseluruhan pipeline
├── weights/
│   └── .gitkeep               # Weights ditambahkan saat build atau setup
├── tests/
│   ├── fixtures/              # Video pendek CFR/VFR/no-audio/empty-video
│   ├── test_schema.py
│   ├── test_timestamps.py
│   ├── test_tracking.py
│   └── test_end_to_end.py
└── .dockerignore
```

## 5. Implementasi Cog / Replicate

> **Penting (versi Cog terbaru):** panduan Cog kini menganjurkan `BaseRunner`, metode `run()`, dan key `run:` di `cog.yaml`. Banyak tutorial lama masih menggunakan `BasePredictor`, `predict()` dan `predict:`; cara lama masih didukung untuk kompatibilitas, tetapi **bukan** pilihan proyek baru ini.

### 5.1 `cog.yaml` — contoh awal

```yaml
build:
  gpu: true
  python_version: "3.11"
  system_packages:
    - "ffmpeg"
  python_requirements: requirements.txt

run: "run.py:Runner"
```

Catatan:

- GPU NVIDIA diperlukan untuk target performa; pastikan Docker NVIDIA runtime tersedia pada host pengujian lokal.
- Cog mengelola CUDA secara default; **verifikasi** kompatibilitas CUDA, PyTorch, driver, dan base image saat build, lalu *pin* versi yang sukses diuji.
- `system_packages` memberi FFmpeg; Python packages dikelola terpisah.
- Tetapkan `image:` atau `model:` sesuai format push yang dipilih jika ingin menyimpan target di YAML; contoh di sini memakai target eksplisit saat `cog push`.

### 5.2 `requirements.txt` — daftar awal, belum lock

```text
ultralytics
opencv-python-headless
av
numpy
```

`torch`/`torchvision` umumnya ikut diresolusikan oleh Ultralytics. Namun image GPU produksi **wajib** memakai kombinasi paket yang sudah diuji dan di-*pin* secara eksplisit (termasuk PyTorch) setelah uji kompatibilitas CUDA. Jangan menganggap contoh daftar ini sebagai lockfile production.

### 5.3 `run.py` — kontrak Cog (kerangka integrasi)

```python
import tempfile
from cog import BaseRunner, BaseModel, Input, Path
from ultralytics import YOLOWorld
from pipeline.process import process_video  # Akan diimplementasikan pada fase pipeline


class Output(BaseModel):
    video: Path
    timeline: Path


class Runner(BaseRunner):
    def setup(self):
        # Gunakan bobot lokal yang disiapkan pada image / bootstrap setup.
        self.model = YOLOWorld("weights/yolov8s-worldv2.pt")

    def run(
        self,
        video: Path = Input(description="Video input"),
        classes: str = Input(
            default="person,car,motorcycle,bicycle",
            description="Object classes, separated by commas",
        ),
        confidence: float = Input(default=0.25, ge=0.05, le=0.95),
        tracker: str = Input(default="bytetrack", choices=["bytetrack", "botsort"]),
        preserve_audio: bool = Input(default=True),
    ) -> Output:
        labels = list(dict.fromkeys(x.strip() for x in classes.split(",") if x.strip()))
        if not labels:
            raise ValueError("classes tidak boleh kosong")

        # Jangan gunakan `with TemporaryDirectory()` di sini: output harus
        # tetap ada hingga Cog selesai mengambil dan mengunggah berkas.
        workdir = Path(tempfile.mkdtemp(prefix="video-timeline-"))

        # process_video harus melakukan:
        # 1. Input validation + probing + decode
        # 2. model.set_classes(labels), inference, per-request tracker reset
        # 3. Annotated MP4 + timeline.json yang sinkron
        # 4. Mengembalikan path file yang benar-benar sudah ada
        mp4_path, json_path = process_video(
            input_video=video,
            workdir=workdir,
            model=self.model,
            classes=labels,
            confidence=confidence,
            tracker=tracker,
            preserve_audio=preserve_audio,
        )
        return Output(video=Path(mp4_path), timeline=Path(json_path))
```

**Kode di atas adalah *skeleton*, belum aplikasi yang dapat dijalankan.** Khususnya `pipeline.process.process_video` perlu diimplementasikan, bobot harus sudah tersedia di `weights/`, dan state tracker perlu **direset untuk setiap prediction**.

Syarat teknis tambahan:

- **Jangan** berbagi instance tracker antar-prediction. Model dapat di-*warm load* di `setup()`, tetapi setting kelas dan tracker state harus aman per request. Pada MVP serialkan inference atau gunakan instance model per request bila implementasi reset belum terbukti aman.
- Jangan menghapus direktori output sebelum Cog selesai mengambil dua file `Path`.
- Pastikan file video sudah ditutup/di-*flush* dan JSON valid sebelum `run()` mengembalikan `Output`.
- Set batas durasi/file size di pipeline, karena validasi tidak boleh mengandalkan ekstensi file saja.

### 5.4 Manajemen model weights

Dua opsi:

**Opsi A (direkomendasikan untuk reproducibility):** unduh weights sekali saat proses persiapan/build, masukkan ke direktori `weights/`, lalu push sebagai bagian image/model; rekam checksum dan versi asal. Konsekuensinya image bisa lebih besar. Gunakan `cog push --separate-weights` bila ukuran weights besar dan sesuai alur publikasi.

**Opsi B:** download weights pada `setup()` saat container pertama berjalan. Image lebih kecil, tetapi cold start bergantung jaringan/host model. Jika memilih ini, gunakan URL sumber terverifikasi dan checksum; hindari unduhan berulang per frame/prediction.

## 6. Detail pipeline video

### 6.1 Validasi & probing

- Probe video dengan `ffprobe`/PyAV: codec, durasi, resolusi, jumlah video stream, tipe audio stream, frame rate dan time base.
- Tolak file tanpa video stream, durasi nol, resolusi ekstrem, format berbahaya, atau melebihi kuota.
- Batas MVP yang **diusulkan**: maksimum 60 detik, hingga 1080p, target 30 FPS; sesuaikan setelah benchmark GPU/cost.
- Input video yang gagal didekode tidak boleh menghasilkan output parsial berstatus sukses.

### 6.2 Decode & timestamp

- Decode frame satu per satu untuk menghindari menampung seluruh video di RAM.
- Ambil `source_timestamp_ms` dari presentation timestamp (PTS) serta `time_base` bila tersedia.
- Untuk video **CFR** normal, keluarkan video di FPS tetap yang sama dan buat `timestamp_ms` dari PTS output.
- Untuk input **VFR**, pilih salah satu kebijakan eksplisit:
  1. **Rekomendasi MVP:** normalisasi ke CFR (misalnya FPS target 30); deteksi semua frame output dan simpan `source_timestamp_ms` untuk melacak asalnya.
  2. Fase lanjut: pertahankan PTS VFR output; perlu muxing ber-timestamp eksplisit dan tes sinkronisasi khusus.
- Jika normalisasi CFR menambah/menggandakan frame, jalankan deteksi pada frame **yang benar-benar ditulis** ke output; `total_frames` harus cocok dengan jumlah frame hasil.

### 6.3 Detection & tracker

- Panggil `model.set_classes(labels)` sekali per prediction, **bukan** setiap frame.
- Jalankan deteksi setiap frame untuk mode `per_frame_full` (tidak ada frame skipping).
- Terapkan ByteTrack melalui tracking API atau adaptor tracker yang dapat diverifikasi perilaku reset dan ID-nya.
- Beda `detection` vs `tracking`: objek dapat terdeteksi tanpa ID; gunakan `track_id: null` bila perlu.
- Simpan koordinat pada resolusi output dan batasi nilai dalam dimensi frame.
- Jika objek sementara tertutup/keluar frame, ID boleh berubah; jangan mengklaim ID global yang permanen.

### 6.4 Renderer & encoder

- Render video dari hasil deteksi/tracking **yang sama** dengan objek dalam JSON (single source of truth).
- Format visual default: `class_name #track_id confidence` + bounding box.
- Output MP4 kompatibel pemutar umum: H.264, `yuv420p`, `faststart` bila perlu untuk web playback.
- Pertahankan audio bila didukung: remux atau re-encode audio input dengan FFmpeg secara terpisah; jika video dinormalisasi, pastikan sinkronisasi durasi audio dan kebijakan padding/trimming.
- Video input tanpa audio tetap menghasilkan MP4 valid.

### 6.5 Serializer timeline

- Hindari menyimpan seluruh objek JSON di RAM untuk video panjang; tulis objek frame secara incremental ke file JSON sementara, kemudian finalisasi array dan metadata.
- Urutkan frame ascending; pastikan indeks berurutan, timestamp monoton, data numeric valid.
- Tambahkan `schema_version`; API consumer dapat menolak perubahan breaking tanpa versi baru.
- File JSON **jangan** berisi seluruh image/frame dalam base64; hanya metadata.

## 7. Deploy Cog ke Replicate — langkah operasional

### Prasyarat

- Akun Replicate dengan izin membuat model dan konfigurasi billing.
- Docker dan Cog terpasang; CUDA-capable NVIDIA GPU untuk tes inference lokal.
- Jika development di macOS tanpa GPU NVIDIA: editing/build dapat dilakukan di Mac, tetapi tes GPU CUDA sebaiknya pada Linux GPU/host cloud atau lewat Replicate.
- Bobot model sudah disiapkan dan dicek lisensinya.

### Tahapan

```bash
# 1. Buat folder proyek
mkdir yolo-world-video-timeline
cd yolo-world-video-timeline

# 2. Inisialisasi bila memulai dari kosong; Cog terkini membuat cog.yaml + run.py
cog init

# 3. Isi cog.yaml, requirements.txt, run.py dan pipeline/
#    Serta siapkan weights dan sample video tes.

# 4. Validasi lingkungan / build
cog doctor
cog build

# 5. Tes prediction pada mesin Linux dengan GPU NVIDIA
cog run -i video=@tests/fixtures/short.mp4 \
  -i classes="person,car" \
  -i confidence=0.25 \
  -i tracker=bytetrack

# 6. Login dan push (buat model melalui Replicate UI terlebih dahulu)
cog login
cog push r8.im/YOUR_USERNAME/yolo-world-video-timeline
```

Perintah di atas **baru dapat dieksekusi setelah source code dan fixtures dibuat**. `YOUR_USERNAME` diganti identitas owner Replicate. Buat model bernama `yolo-world-video-timeline` di https://replicate.com/create dan pilih public/private sesuai kebutuhan/lisensi.

### Uji API setelah publish

Contoh request asynchronous API (gunakan versi/endpoint yang cocok dengan model yang diterbitkan):

```bash
curl -X POST \
  "https://api.replicate.com/v1/models/YOUR_USERNAME/yolo-world-video-timeline/predictions" \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "video": "https://example.com/sample.mp4",
      "classes": "person,car",
      "confidence": 0.25,
      "tracker": "bytetrack"
    }
  }'
```

Respons awal mungkin `starting`/`processing`; lakukan polling URL prediction atau gunakan webhook sampai `succeeded`. Setelah sukses, baca `output.video` dan `output.timeline`, lalu unduh kedua file.

**Retensi Replicate:** menurut dokumentasi, output prediction yang dibuat melalui API dibersihkan otomatis sekitar **1 jam** secara default; aplikasi yang memerlukan penyimpanan permanen harus menyalin MP4 dan JSON ke object storage sendiri sebelum kadaluarsa.

## 8. Roadmap pengerjaan dan definition of done

### Fase A — Spike / proof-of-concept

- [ ] Siapkan repo + `cog.yaml` + `run.py:Runner`.
- [ ] Build Cog GPU berhasil, bobot YOLO-World ter-load.
- [ ] YOLO-World mampu mendeteksi kelas custom pada beberapa frame contoh.
- [ ] Tracking ByteTrack menghasilkan ID pada dua frame berurutan.
- [ ] Output Cog berbentuk `BaseModel(video: Path, timeline: Path)` berhasil dikenali.

**DoD A:** 1 sampel 5–10 detik menghasilkan **dua file** yang dapat dibaca, walau UI/anotasi belum final.

### Fase B — MVP video + timeline penuh

- [ ] Probe input + validasi durasi/resolusi/codec.
- [ ] Decode seluruh frame output secara sequential.
- [ ] Deteksi + tracking per frame; reset state tiap prediction.
- [ ] Renderer bbox/label/ID dengan warna stabil menurut track ID.
- [ ] MP4 H.264 ter-*encode* dan dapat diputar di browser.
- [ ] `timeline.json` valid dan punya skema berversi.
- [ ] `source_timestamp_ms` dan `timestamp_ms` direkam sesuai kebijakan CFR/VFR.
- [ ] Dukungan input dengan/tanpa audio; tidak menyebabkan error.

**DoD B:** jumlah `frames` di JSON **sama persis** dengan frame video hasil berdasarkan dekode output; label/bbox sinkron dengan gambar.

### Fase C — Publish dan uji production readiness

- [ ] Versions package/weights/checksum dikunci dan dicatat.
- [ ] `cog run` berhasil pada GPU NVIDIA.
- [ ] `cog push` berhasil dan prediction Replicate sukses.
- [ ] API menghasilkan output object dengan dua URL file.
- [ ] Uji video tanpa objek, objek banyak, VFR, audio, resolusi besar, video rusak.
- [ ] Monitor waktu proses, GPU memory, ukuran JSON, failure rate.
- [ ] Dokumentasi endpoint dan contoh penggunaan Python/JavaScript dibuat.
- [ ] Review lisensi model, paket YOLO, dan opsi public/private selesai.

**DoD C:** dua file dapat diunduh melalui API Replicate dan lulus tes sinkronisasi otomatis.

### Fase D — Integrasi aplikasi (opsional)

- [ ] Backend aplikasi membuat prediction Replicate dan menyimpan `prediction_id`.
- [ ] Status job diambil via polling atau webhook tervalidasi.
- [ ] MP4 + JSON disalin ke penyimpanan milik aplikasi sebelum retensi Replicate habis.
- [ ] UI timeline mengizinkan klik timestamp → seek video; filter berdasar class/track ID.
- [ ] Event turunan (`first_seen`, `last_seen`, interval objek muncul) dihasilkan oleh service metadata terpisah.

### Fase E — Optimasi / versi selanjutnya

- [ ] Benchmark `s` vs `m` YOLO-World pada GPU target.
- [ ] Opsi BoT-SORT untuk scene kompleks/bergerak.
- [ ] Batch/FP16/export ONNX/TensorRT jika kompatibel dan hasil setara.
- [ ] JSONL/kompresi untuk video panjang.
- [ ] Streaming RTSP/WebRTC dan deteksi near-real-time **di pipeline berbeda**, bukan sekadar mengubah input Cog batch.

## 9. Test matrix dan acceptance criteria

| Test | Input | Kriteria lulus |
| --- | --- | --- |
| Normal | MP4 30 FPS 10 detik | MP4 dan JSON valid; urutan frame dan timestamp benar |
| Empty detection | Video tanpa label yang cocok | Semua frame tetap tercatat `objects: []` |
| Track continuity | Orang bergerak konsisten | ID biasanya sama selama terlihat; mismatch dihitung |
| Multiple objects | Beberapa kelas saling mendekat | Bbox/label/ID sesuai gambar; tidak crash |
| Video tanpa audio | MP4 silent | Output valid dan tidak memerlukan audio stream |
| Video dengan audio | MP4 + AAC | Audio dan video tetap sinkron dalam toleransi yang disepakati |
| VFR | Video ponsel variable FPS | Kebijakan normalisasi ditetapkan; timestamp hasil konsisten |
| Invalid input | File rusak / bukan video | Prediction gagal dengan pesan validasi jelas |
| Over limit | Durasi/resolusi > batas | Ditolak cepat tanpa inference mahal |
| Repeat prediction | Dua video berturut | State ID tracker tidak bocor antar-request |

**Acceptance utama (wajib):**

1. `output.video` dan `output.timeline` keduanya URL output Replicate yang dapat diunduh.
2. `output.mp4` dapat di-decode sampai selesai, tidak hanya memiliki header valid.
3. `len(timeline.frames) == actual_decoded_frame_count(output.mp4)`.
4. Indeks frame tidak ada gap/duplikat; timestamp output monoton dan wajar terhadap FPS/PTS.
5. Semua `bbox_xyxy` valid (`0 <= x1 < x2 <= width`, `0 <= y1 < y2 <= height`) dan confidence valid.
6. ID objek *scoped per prediction*; tidak ada state tracker yang bocor.
7. JSON valid sesuai `schema_version`, termasuk ketika objek nihil.
8. Output terverifikasi pada prediction Replicate, bukan hanya uji proses Python lokal.

## 10. Risiko, mitigasi dan kebijakan produk

| Risiko | Dampak | Mitigasi |
| --- | --- | --- |
| Lisensi Ultralytics | Proprietary SaaS dapat membutuhkan lisensi enterprise / kepatuhan AGPL | Review komersial **sebelum publish**, lihat tautan resmi di referensi |
| Lisensi weights/repo YOLO-World asli | Redistribusi weights/code dapat mempunyai syarat terpisah | Audit LICENSE repo asli, lisensi setiap weights dan atribusi |
| JSON meledak pada video panjang | File output dan memori besar | Limit durasi, serializer incremental, kompresi/JSONL versi lanjut |
| Video VFR | Drift timestamp/frame | Normalisasi CFR yang eksplisit + regression tests PTS |
| Tracker ID switch | Data analitik keliru | Dokumentasikan sifat probabilistik; evaluasi BoT-SORT |
| OOM / durasi prediction tinggi | Prediction gagal/mahal | Batas input, process streaming, benchmark GPU dan skala weights |
| Cold start download weights | Startup lama atau gagal | Build-in weights/checksum; pertimbangkan separate-weights |
| File output Replicate kadaluarsa | Data produk hilang | Salin hasil ke object storage sendiri segera setelah sukses |
| Konten input sensitif | Risiko privasi | Batasi akses, minimalkan penyimpanan, jangan log frame mentah |
| GPU shared state | Prediksi berbeda saling mengubah kelas/tracker | Serialize request atau isolasi model/tracker per job |

### Catatan lisensi yang tidak boleh dilewati

- Implementasi **Ultralytics** didistribusikan menggunakan **AGPL-3.0** dengan opsi **Enterprise License** untuk kebutuhan proprietary tertentu, termasuk SaaS; periksa kewajiban tepatnya dengan pihak berwenang/lisensi sebelum launch komersial.
- Repo YOLO-World asli, bobot, dan implementasi pihak ketiga dapat memiliki lisensi yang berbeda. Jangan menganggap semua code/weights otomatis bebas untuk redistribusi komersial hanya karena bisa dipakai di Replicate.
- Jika perlu source tertutup, lakukan review opsi lisensi atau pilih implementasi detector alternatif yang sesuai sebelum membangun ketergantungan produksi.

## 11. Estimasi dan indikator performa (tanpa janji angka)

**Tidak menetapkan FPS inference atau biaya prediction pasti sebelum benchmark.** Waktu proses merupakan fungsi dari resolusi, durasi/FPS video, model weight, GPU, jumlah prompt classes, decoding, tracking, dan encoding.

Ukur baseline pada video 10s/30s/60s (720p dan 1080p):

- `setup_seconds` / cold start.
- `decode_fps`, `infer_fps`, `encode_fps`, `end_to_end_seconds`.
- Peak GPU memory dan CPU/RAM.
- Ukuran MP4 dan timeline JSON per menit.
- `cost_per_video_minute = rate_GPU * runtime_seconds / 3600` (tambahkan overhead platform/storage jika relevan).
- Akurasi deteksi pada label khusus, stabilitas track ID, dan latency seek UI.

Mulai dengan YOLO-World `small` demi biaya rendah, kemudian naikkan bobot bila akurasi butuh peningkatan dan anggarannya memungkinkan.

## 12. Daftar keputusan yang perlu dikunci sebelum coding production

1. **License strategy:** seluruh source layanan boleh AGPL atau butuh enterprise/alternatif?
2. **Batas video:** 60 detik / 1080p / 30 FPS hanya target awal; benchmark untuk batas final.
3. **CFR/VFR:** rekomendasi normalize output ke CFR agar UI timeline dan video sinkron.
4. **Storage:** apakah cukup link Replicate sesaat atau selalu disimpan ke R2/S3?
5. **Model checkpoint:** `yolov8s-worldv2.pt` dan versi paket yang telah diuji.
6. **Model privacy:** public atau private saat dibuat di Replicate.

## 13. Referensi teknis resmi

- Cog Python / BaseRunner / structured output: https://cog.run/python/
- Cog YAML `build`, `gpu`, `run`: https://cog.run/yaml/
- Cog CLI `cog init`, `cog run`, `cog push`: https://cog.run/cli/
- Replicate — Push your own model: https://replicate.com/docs/guides/build/push-a-model
- Replicate — Output files: https://replicate.com/docs/topics/predictions/output-files
- Replicate — Data retention: https://replicate.com/docs/topics/predictions/data-retention
- Ultralytics — YOLO-World and custom classes: https://docs.ultralytics.com/models/yolo-world/
- Ultralytics — Multi-Object Tracking: https://docs.ultralytics.com/modes/track/
- Ultralytics licensing: https://www.ultralytics.com/license
- Base model inspirasi: https://replicate.com/zsxkib/yolo-world

---

**Rekomendasi pelaksanaan:** kerjakan **Fase A → B → C** terlebih dahulu. Keberhasilan awal bukan pada jumlah fitur, melainkan memastikan **satu prediction Replicate benar-benar mengeluarkan MP4 + JSON dengan timestamp dan objek yang sinkron untuk setiap frame**.
