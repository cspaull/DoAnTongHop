# 📊 Table Detection using TabRec Dataset (YOLO)

Dự án này thực hiện bài toán **Table Detection** sử dụng dataset
**TabRec**, với pipeline chính:

1.  Convert annotation TabRec → YOLO format\
2.  Train model YOLO (Ultralytics)\
3.  Train với custom backbone (tùy chỉnh trong `fix.yaml`)

------------------------------------------------------------------------

## 📁 Cấu trúc thư mục

    .
    ├── checkgpu.py                  # Kiểm tra GPU
    ├── convertyolo.py              # Convert TabRec → YOLO annotation
    ├── fix.yaml                    # Config model (custom backbone)
    ├── test.py                     # Test model
    ├── train.py                    # Train YOLO (Ultralytics gốc)
    ├── train_backbonecustome.py    # Train với custom backbone
    ├── validate.py                 # Evaluate model

------------------------------------------------------------------------

## ⚙️ Cài đặt

### 1. Clone repo

``` bash
git clone <your-repo-url>
cd <repo-name>
```

### 2. Cài dependencies

``` bash
pip install -r requirements.txt
```

Hoặc tối thiểu:

``` bash
pip install ultralytics opencv-python
```

------------------------------------------------------------------------

## 📦 Dataset: TabRec

-   Dataset gốc: **TabRec**
-   Cần convert về format YOLO trước khi train

------------------------------------------------------------------------

## 🔄 Convert annotation (TabRec → YOLO)

Chạy script:

``` bash
python convertyolo.py
```

### Output:

    dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/

### YOLO annotation format:

    <class_id> <x_center> <y_center> <width> <height>

------------------------------------------------------------------------

## 🚀 Train model (Ultralytics YOLO)

Dùng script có sẵn:

``` bash
python train.py
```

Hoặc trực tiếp:

``` bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640
```

------------------------------------------------------------------------

## 🧠 Train với Custom Backbone

Sử dụng file:

``` bash
python train_backbonecustome.py
```

### Điểm đặc biệt:

-   Backbone được tùy chỉnh trong `fix.yaml`
-   Có thể thay thế:
    -   CNN backbone
    -   hoặc tích hợp model khác (VD: DINO, Transformer...)

------------------------------------------------------------------------

## 🧪 Validate model

``` bash
python validate.py
```

------------------------------------------------------------------------

## 🔍 Test model

``` bash
python test.py
```

------------------------------------------------------------------------

## 🖥️ Kiểm tra GPU

``` bash
python checkgpu.py
```

------------------------------------------------------------------------

## 📌 Ghi chú

-   `train.py` → dùng pipeline chuẩn của **Ultralytics**
-   `train_backbonecustome.py` → dùng khi muốn:
    -   thay backbone
    -   thử nghiệm research
-   `fix.yaml` → nơi cấu hình model (quan trọng)

------------------------------------------------------------------------

## 📈 TODO / Future Work

-   [ ] Improve backbone (DINOv2 / ViT)
-   [ ] Data augmentation nâng cao
-   [ ] Multi-scale training
-   [ ] Fine-tune trên dataset khác
