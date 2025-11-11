from ultralytics import YOLO

def main():
    # Load mô hình
    model = YOLO("E:/DoAnTongHop/runs/detect/train/weights/best.pt")

    # Chạy detect giống lệnh CLI
    results = model.predict(
        source="E:/DoAnTongHop/tabrec_yolo/images/predict",  # folder hoặc ảnh
        imgsz=640,
        conf=0.25,
        iou=0.45,
        save=True,      # lưu ảnh detect
        save_txt=False, # CLI mặc định không lưu txt nếu không thêm
        device=0        # GPU hoặc 'cpu'
    )

    print("Detection completed! Results saved to runs/detect/predict/")

if __name__ == "__main__":
    main()
