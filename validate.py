from ultralytics import YOLO

def main():
    # Đường dẫn tới mô hình đã huấn luyện
    model_path = "runs/detect/train/weights/best.pt"  # đổi nếu khác
    # Đường dẫn tới dataset validation (file YAML)
    data_path = "D:/tabrec_yolo/data.yaml"

    # Load mô hình
    model = YOLO(model_path)

    # Chạy validate/evaluate
    # metrics: mAP, precision, recall
    results = model.val(
        data=data_path,
        imgsz=640,     # Kích thước ảnh (tùy chỉnh)
        batch=8,
        device=0       # GPU index hoặc 'cpu'
    )

    # Hiển thị kết quả
    
if __name__ == "__main__":
    main()
