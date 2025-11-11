from ultralytics import YOLO

def main():
    # Load mô hình từ backbone YAML bạn tạo (chưa có weight)
    model = YOLO("E:/DoAnTongHop/fix.yaml")

    # Train model
    results = model.train(
        data="E:/DoAnTongHop/tabrec_yolo/data.yaml",  # file data.yaml mô tả đường dẫn train/val
        epochs=100,           # số epoch muốn train
        imgsz=640,            # kích thước ảnh
        batch=8,              # batch size (tùy GPU)
        workers=4,            # số luồng đọc dữ liệu
        device=0,             # dùng GPU đầu tiên, hoặc 'cpu'
        lr0=0.01,             # learning rate khởi đầu
        optimizer="SGD",      # hoặc "AdamW" nếu bạn muốn fine-tune
        pretrained=False,     # vì bạn train từ backbone YAML
        project="E:/DoAnTongHop/runs",   # thư mục lưu kết quả
        name="train_custom_backbone",    # tên subfolder trong project
        verbose=True
    )

    print("Training completed! Check results in runs/train_custom_backbone/")

if __name__ == "__main__":
    main()
