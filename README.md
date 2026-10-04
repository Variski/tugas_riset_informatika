# Riset Kecil: Ketahanan Detektor Deepfake Audio Berbahasa Indonesia terhadap Kompresi Codec Aplikasi Pesan

**Mata Kuliah:** Riset Informatika
**Nama:** Muchammad Basroil Billah
**Bidang:** Audio Deepfake Detection (ADD) / Speech Anti-Spoofing

---

## 1. Ringkasan Riset

Penipuan berbasis suara kloning (voice cloning) di Indonesia umumnya terjadi lewat voice note atau panggilan di aplikasi pesan seperti WhatsApp. Audio yang lewat jalur ini sudah terkompresi oleh codec (misalnya Opus) dan sering berkualitas rendah. Di sisi lain, sebagian besar model deteksi deepfake audio dilatih dan diuji pada audio bersih berbahasa Inggris.

Riset kecil ini menguji seberapa jauh performa detektor deepfake audio turun ketika (1) diterapkan pada ucapan berbahasa Indonesia dan (2) audionya sudah melewati kompresi codec seperti di aplikasi pesan. Setelah itu, riset menguji apakah **codec augmentation** saat fine-tuning bisa memulihkan performa tersebut.

**Judul kerja:**
> Evaluasi Ketahanan Detektor Deepfake Audio Berbahasa Indonesia terhadap Degradasi Codec Aplikasi Pesan dan Pengaruh Codec Augmentation

---

## 2. Research Gap dari Penelitian Terdahulu

| No | Penelitian | Temuan Utama | Keterbatasan (Gap) |
|----|-----------|--------------|--------------------|
| 1 | Müller dkk. (2022), *Does Audio Deepfake Detection Generalize?* | Model yang bagus di ASVspoof 2019 turun drastis di data "in-the-wild" (EER naik hingga sekitar 1000%). | Data uji hanya berbahasa Inggris. |
| 2 | Mawalim, Arief & Lestari (2025), *InaSAS* | Membuat dataset InaSpoof-v1 (bahasa Indonesia); AASIST efektif untuk serangan sintesis. | Penulis sendiri menyarankan perlunya menangani tantangan skenario dunia nyata; fokus pada kondisi rekaman, bukan kanal aplikasi pesan. |
| 3 | SEA-Spoof (2025) | Model SoTA yang dilatih pada data bahasa tinggi sumber daya (high-resource) gagal di bahasa Asia Tenggara termasuk Indonesia; fine-tuning memulihkan performa. | Seluruh audio disimpan bersih (16 kHz FLAC), sehingga belum diuji pada audio terkompresi. Fine-tuning juga memunculkan catastrophic forgetting yang dibiarkan sebagai future work. |
| 4 | Shi dkk. (2025), *ADD-C* | Codec dan packet loss di skenario komunikasi menurunkan performa ADD; data augmentation membantu. | Tidak menyasar bahasa Indonesia. |
| 5 | *Measuring the Robustness of Audio Deepfake Detection under Real-World Corruptions* (2025) | Kompresi (termasuk Opus dan neural codec) merusak performa deteksi meski kualitas perseptual tetap bagus. | Evaluasi pada data berbahasa Inggris. |
| 6 | Adila, Mawalim & Unoki (2024), *Indonesian & Thai non-native speech* | Detektor yang dilatih pada penutur native kesulitan pada penutur non-native Indonesia/Thai. | Fokus pada aksen non-native dalam bahasa Inggris, bukan bahasa Indonesia, dan belum menyentuh codec. |

**Kesimpulan gap:**
Ada dua garis penelitian yang berjalan terpisah. Garis pertama membuktikan bahwa detektor gagal pada bahasa Indonesia (InaSAS, SEA-Spoof). Garis kedua membuktikan bahwa detektor gagal pada audio terkompresi codec (ADD-C, studi robustness). Dari literatur yang ditelusuri, belum ditemukan penelitian yang menggabungkan keduanya, yaitu menguji detektor deepfake **berbahasa Indonesia** pada kondisi **codec aplikasi pesan**, padahal kombinasi inilah yang paling dekat dengan modus penipuan suara di Indonesia.

---

## 3. Peluang Pengembangan

1. **Benchmark kecil bahasa Indonesia + kondisi kanal.** Menambahkan versi terkompresi (Opus, AMR-NB, MP3) pada subset audio berbahasa Indonesia sebagai test set baru yang realistis.
2. **Codec augmentation untuk bahasa Indonesia.** Menguji apakah strategi augmentasi yang terbukti di bahasa Inggris (ADD-C) juga berlaku saat fine-tuning ke bahasa Indonesia.
3. **Generalisasi ke codec yang tidak terlihat (unseen codec).** Melatih dengan sebagian codec, lalu menguji pada codec lain, untuk melihat apakah model belajar ciri deepfake atau sekadar menghafal artefak codec.
4. **Mengurangi catastrophic forgetting.** Membandingkan fine-tuning murni bahasa Indonesia dengan mixed-source training (Indonesia + ASVspoof) yang disebut sebagai future work oleh SEA-Spoof.
5. **Pengembangan lanjut (di luar cakupan riset kecil):** model ringan untuk on-device detection di smartphone, deteksi partial spoof (hanya sebagian kalimat yang palsu), dan perluasan ke bahasa daerah (Jawa, Sunda) serta campuran bahasa (code-switching).

---

## 4. Rencana Topik

### 4.1 Rumusan Masalah
1. Seberapa besar penurunan performa (EER) detektor deepfake audio yang dilatih pada data bahasa Inggris ketika diuji pada audio berbahasa Indonesia?
2. Seberapa besar penurunan tambahan ketika audio berbahasa Indonesia tersebut dikompresi dengan codec aplikasi pesan (Opus bitrate rendah) dan codec telepon (AMR-NB)?
3. Apakah fine-tuning dengan codec augmentation menghasilkan EER yang lebih rendah dibanding fine-tuning pada audio bersih, termasuk pada codec yang tidak dipakai saat training?

### 4.2 Tujuan
1. Mengukur performa zero-shot detektor berbasis ASVspoof pada audio berbahasa Indonesia, bersih dan terkompresi.
2. Membandingkan tiga skenario fine-tuning: tanpa augmentasi, codec augmentation, dan mixed-source + codec augmentation.
3. Mempublikasikan kode, protokol split, dan skrip degradasi codec di repo ini agar dapat direproduksi.

### 4.3 Batasan
- Bahasa: Indonesia baku (bukan bahasa daerah).
- Jenis serangan: TTS dan voice conversion (logical access), tanpa replay attack.
- Skala: subset kecil (target sekitar 2.000 utterance bona fide + 2.000 utterance spoof) agar dapat dilatih di GPU gratis (Google Colab / Kaggle Notebook).

### 4.4 Metodologi

```
[Data bahasa Indonesia]  ->  [Degradasi codec]  ->  [Model]  ->  [Evaluasi]
 bona fide + spoof           clean, Opus 16/24k,     LFCC-LCNN     EER per kondisi
                             AMR-NB 12.2k, MP3 64k   AASIST        seen vs unseen codec
```

**a. Data**
- Opsi utama: subset bahasa Indonesia dari SEA-Spoof (jika sudah dirilis) atau InaSpoof-v1 (jika aksesnya diberikan penulis).
- Opsi cadangan (paling realistis untuk tugas): membangun mini-dataset sendiri.
  - Bona fide: Mozilla Common Voice bahasa Indonesia.
  - Spoof: teks transkrip yang sama disintesis dengan MMS-TTS bahasa Indonesia (`facebook/mms-tts-ind`) dan satu sistem TTS lain (misalnya Edge-TTS atau XTTS-v2), sehingga setiap kalimat punya pasangan asli dan palsu.
- Split berdasarkan speaker (speaker-disjoint) untuk train/dev/test agar tidak terjadi kebocoran data.

**b. Degradasi codec (memakai `ffmpeg`)**

| Kondisi | Pengaturan | Mewakili |
|---------|-----------|----------|
| Clean | 16 kHz WAV | Kondisi laboratorium |
| Opus-24k | libopus 24 kbps | Voice note aplikasi pesan |
| Opus-16k | libopus 16 kbps | Voice note dengan sinyal buruk |
| AMR-NB | 12.2 kbps, 8 kHz | Panggilan telepon seluler |
| MP3-64k | libmp3lame 64 kbps | Audio yang diunggah ulang ke media sosial |

**c. Model**
- Baseline ringan: LFCC + LCNN.
- Baseline kuat: AASIST (pretrained ASVspoof 2019 LA dari repo resmi).
- Opsional bila sumber daya cukup: wav2vec 2.0 XLS-R + AASIST (SSL front-end).

**d. Skenario eksperimen**

| Kode | Training | Testing | Tujuan |
|------|----------|---------|--------|
| E1 | ASVspoof 2019 LA (pretrained) | ID clean | Efek beda bahasa |
| E2 | ASVspoof 2019 LA (pretrained) | ID semua codec | Efek bahasa + codec |
| E3 | Fine-tune ID clean | ID semua codec | Fine-tuning tanpa augmentasi |
| E4 | Fine-tune ID + augmentasi Opus & MP3 | ID semua codec (AMR-NB = unseen) | Efek codec augmentation dan generalisasi unseen codec |
| E5 | Fine-tune ID + ASVspoof + augmentasi | ID semua codec dan ASVspoof eval | Mengukur catastrophic forgetting |

**e. Metrik**
- Equal Error Rate (EER), metrik standar ASVspoof.
- Pelaporan per kondisi codec dan per sistem TTS penghasil spoof.

### 4.5 Hipotesis
- H1: EER pada E1 jauh lebih tinggi dibanding EER model pada ASVspoof (konsisten dengan SEA-Spoof).
- H2: Kompresi Opus bitrate rendah dan AMR-NB menaikkan EER lebih jauh lagi (E2 > E1).
- H3: E4 menghasilkan EER lebih rendah dari E3 pada kondisi terkompresi, termasuk pada AMR-NB yang tidak terlihat saat training.

### 4.6 Rencana Struktur Repository

```
deepfake-audio-id-codec/
├── README.md                 # dokumen ini
├── data/
│   ├── protocols/            # daftar file train/dev/test (speaker-disjoint)
│   └── README.md             # cara mengunduh dataset (audio tidak di-upload)
├── scripts/
│   ├── build_spoof_tts.py    # generate spoof dari transkrip Common Voice
│   ├── apply_codec.sh        # degradasi codec dengan ffmpeg
│   └── compute_eer.py        # evaluasi EER
├── models/                   # konfigurasi LFCC-LCNN dan AASIST
├── notebooks/                # eksperimen di Colab / Kaggle
├── results/                  # tabel EER dan grafik
└── references/
    └── references.bib        # daftar pustaka format BibTeX
```

> Catatan: file audio dataset tidak diunggah ke GitHub karena ukuran dan lisensi. Cukup sertakan skrip, protokol split, dan tautan sumber.

---

## 5. Sumber

### a. Nama Repository Dataset

| Dataset | Bahasa | Peran dalam riset |
|---------|--------|-------------------|
| ASVspoof 2019 Logical Access (LA) | Inggris | Data pretraining model baseline, uji forgetting |
| In-the-Wild (Müller dkk., 2022) | Inggris | Pembanding kondisi dunia nyata |
| SEA-Spoof (subset Indonesia) | Indonesia dan 5 bahasa SEA lain | Data utama (jika tersedia) |
| InaSpoof-v1 | Indonesia | Data utama alternatif (akses via penulis) |
| Mozilla Common Voice (Indonesian) | Indonesia | Sumber audio bona fide untuk mini-dataset |
| MLAAD | Multibahasa | Pembanding multibahasa (opsional) |
| The Fake-or-Real (FoR) Dataset | Inggris | Data tambahan (opsional) |

### b. Alamat Repository (GitHub / Kaggle / Hugging Face / lainnya)

**Dataset**
- ASVspoof 2019 (Edinburgh DataShare): https://datashare.ed.ac.uk/handle/10283/3336
- In-the-Wild: https://deepfake-demo.aisec.fraunhofer.de/in_the_wild
- In-the-Wild (mirror Hugging Face): https://huggingface.co/datasets/mueller91/In-The-Wild
- SEA-Spoof (Hugging Face, sesuai yang dicantumkan di paper): https://huggingface.co/datasets/Jack-ppkdczgx/SEA-Spoof
- InaSAS / InaSpoof-v1 (halaman paper): https://www.nowpublishers.com/article/Details/SIP-20240080
- Mozilla Common Voice: https://commonvoice.mozilla.org/datasets
- MLAAD (Hugging Face): https://huggingface.co/datasets/mueller91/MLAAD
- Fake-or-Real (Kaggle): https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset

**Kode dan model**
- AASIST (resmi): https://github.com/clovaai/aasist
- SSL wav2vec 2.0 + AASIST: https://github.com/TakHemlata/SSL_Anti-spoofing
- MMS-TTS bahasa Indonesia: https://huggingface.co/facebook/mms-tts-ind

**Paper (arXiv)**
- SEA-Spoof: https://arxiv.org/abs/2509.19865
- Does Audio Deepfake Detection Generalize?: https://arxiv.org/abs/2203.16263
- ADD-C: https://arxiv.org/abs/2504.12423
- Robustness under Real-World Corruptions: https://arxiv.org/abs/2503.17577
- Indonesian & Thai non-native speech: https://arxiv.org/abs/2412.01040

### c. Referensi Jurnal

> Keterangan DOI: DOI berawalan `10.48550/arXiv` adalah DOI versi preprint arXiv. Dipakai jika DOI penerbit tidak tersedia (misalnya prosiding LREC dan JMLR memang tidak menerbitkan DOI) atau belum berhasil ditelusuri.

1. Wang, X., Yamagishi, J., Todisco, M., dkk. (2020). ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech. *Computer Speech & Language*, 64, 101114.
   DOI: https://doi.org/10.1016/j.csl.2020.101114
2. Liu, X., Wang, X., Sahidullah, M., dkk. (2023). ASVspoof 2021: Towards spoofed and deepfake speech detection in the wild. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 31, 2507–2522.
   DOI: https://doi.org/10.1109/TASLP.2023.3285283
3. Müller, N., Czempin, P., Diekmann, F., Froghyar, A., & Böttinger, K. (2022). Does audio deepfake detection generalize? *Proc. Interspeech 2022*, 2783–2787.
   DOI: https://doi.org/10.21437/Interspeech.2022-108
4. Mawalim, C. O., Arief, S. A., & Lestari, D. P. (2025). InaSAS: Benchmarking Indonesian speech antispoofing systems. *APSIPA Transactions on Signal and Information Processing*, 14(3), e203.
   DOI: https://doi.org/10.1561/116.20240080
5. SEA-Spoof: Bridging the gap in multilingual audio deepfake detection for South-East Asia. (2025). arXiv:2509.19865.
   DOI (arXiv): https://doi.org/10.48550/arXiv.2509.19865
6. Shi, H., Shi, X., Dogan, S., Alzubi, S., Huang, T., & Zhang, Y. (2025). Benchmarking audio deepfake detection robustness in real-world communication scenarios. arXiv:2504.12423.
   DOI (arXiv): https://doi.org/10.48550/arXiv.2504.12423
7. Measuring the robustness of audio deepfake detection under real-world corruptions. (2025). arXiv:2503.17577.
   DOI (arXiv): https://doi.org/10.48550/arXiv.2503.17577
8. Adila, A., Mawalim, C. O., & Unoki, M. (2024). Detecting spoof voices in Asian non-native speech: An Indonesian and Thai case study. *Proc. APSIPA ASC 2024*.
   DOI (arXiv): https://doi.org/10.48550/arXiv.2412.01040
9. Jung, J., Heo, H.-S., Tak, H., Shim, H., Chung, J. S., Lee, B.-J., Yu, H.-J., & Evans, N. (2022). AASIST: Audio anti-spoofing using integrated spectro-temporal graph attention networks. *Proc. IEEE ICASSP 2022*.
   DOI: https://doi.org/10.1109/ICASSP43922.2022.9747766
10. Müller, N. M., Kawa, P., Choong, W. H., Casanova, E., Gölge, E., Müller, T., Syga, P., Sperl, P., & Böttinger, K. (2024). MLAAD: The multi-language audio anti-spoofing dataset. *Proc. IJCNN 2024*, 1–7.
    DOI (arXiv): https://doi.org/10.48550/arXiv.2401.09512
11. Wu, H., Tseng, Y., & Lee, H. (2024). CodecFake: Enhancing anti-spoofing models against deepfake audios from codec-based speech synthesis systems. *Proc. Interspeech 2024*, 1770–1774.
    DOI (arXiv): https://doi.org/10.48550/arXiv.2406.07237
12. Pratap, V., Tjandra, A., Shi, B., dkk. (2024). Scaling speech technology to 1,000+ languages. *Journal of Machine Learning Research*, 25(97), 1–52.
    DOI (arXiv): https://doi.org/10.48550/arXiv.2305.13516
13. Ardila, R., Branson, M., Davis, K., dkk. (2020). Common Voice: A massively-multilingual speech corpus. *Proc. LREC 2020*, 4218–4222.
    DOI (arXiv): https://doi.org/10.48550/arXiv.1912.06670
14. Prihasto, B., Nur Farid, M., & Al Khairy, R. (2024). Advancing voice anti-spoofing systems: Self-supervised learning and Indonesian dataset integration for enhanced generalization. *Brilliance: Research of Artificial Intelligence*, 4(2).
    DOI: belum ditemukan, cek di halaman artikel pada situs jurnal Brilliance.
