# mAIware - AI Malware Detector 🛡️

Phân tích file PE (`.exe`, `.dll`) bằng AI, phân loại: **benign** (an toàn), **suspicious** (nghi ngờ), **malware** (độc hại).

## 🚀 Cài đặt

```bash
# 1. Clone repo
git clone https://github.com/vuquangtien/mAIware---AI.git
cd mAIware---AI

# 2. Tạo môi trường ảo Python (BẮT BUỘC)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# 3. Cài thư viện
pip install -r requirements.txt

# 4. Tải models từ Releases
# Truy cập: https://github.com/vuquangtien/mAIware---AI/releases
# Tải file ensemble_models.zip và giải nén
unzip ensemble_models.zip -d ensemble_models/
# Nếu models nằm trong ensemble_models/ensemble_models/, di chuyển lên:
mv ensemble_models/ensemble_models/*.joblib ensemble_models/ 2>/dev/null || true
```

## 💻 Sử dụng

```bash
# Kích hoạt môi trường ảo (nếu chưa)
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Tạo thư mục chứa file .exe cần quét
mkdir temp_scan
cp your_file.exe temp_scan/

# Chạy phân tích
python3 ensemble_predict_dir.py temp_scan/
```

**Kết quả:** File `<folder>_voting_result.csv` chứa kết quả phân loại.

```csv
sample_name,Entropy_Total,ensemble_class,ensemble_score
your_file.exe,6.85,malware,0.85
```

## 🗺️ Trích xuất call graph (tùy chọn)

```bash
python3 extract_callgraph.py your_file.exe -o callgraph --render
```

Kết quả: tạo `callgraph.callgraph.dot` (và `callgraph.callgraph.png` nếu có Graphviz).

---