# """
# train_yolo_full.py
# ====================
# Script huấn luyện YOLOv11 với đầy đủ tham số thường dùng.

# Cách chạy:
#     python train_yolo_full.py

# Yêu cầu:
#     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#     pip install ultralytics pillow
# """

# from ultralytics import YOLO

# def main():
    
#     # THAM SỐ CẤU HÌNH
#     # --- Dataset & Model ---
#     data_path = "D:/tabrec_yolo/data.yaml"   # Dataset YAML
#     model_name = "yolo11s.pt"               # Chọn backbone YOLOv11: n, s, m, l, x
#     epochs = 50                              # Số vòng huấn luyện
#     imgsz = 640                              # Kích thước ảnh đầu vào
#     batch = 16                               # Số ảnh mỗi batch
#     device = 0                               # GPU index hoặc 'cpu'
#     workers = 3                              # Số luồng đọc dữ liệu
#     pretrained = True                        # Dùng pretrained weights?

#     # --- Data Augmentation ---
#     degrees = 10.0       # Góc xoay ngẫu nhiên (-10° đến +10°)
#     scale = 0.9          # Phóng to/thu nhỏ ngẫu nhiên (tỉ lệ 0.9 ~ 1.1)
#     shear = 2.0          # Biến dạng shear (nghiêng ảnh)
#     flipud = 0.0         # Lật ảnh theo chiều dọc (tỉ lệ xác suất)
#     fliplr = 0.5         # Lật ảnh theo chiều ngang (50%)
#     mosaic = 1.0         # Xác suất sử dụng mosaic augmentation (1.0 = luôn dùng)
#     mixup = 0.1          # Trộn ảnh mixup (tăng cường học tổng quát)
#     hsv_h = 0.015        # Biến thiên Hue (mặc định)
#     hsv_s = 0.7          # Biến thiên Saturation (mặc định)
#     hsv_v = 0.4          # Biến thiên Value (độ sáng) (mặc định)

#     # --- Training Behavior ---
#     lr0 = 0.01           # Learning rate ban đầu
#     lrf = 0.01           # Learning rate cuối (tỉ lệ giảm)
#     momentum = 0.937     # Momentum (tốc độ cập nhật)
#     weight_decay = 0.0005# Regularization
#     warmup_epochs = 3.0  # Epoch khởi động (tăng dần LR)
#     patience = 30        # Số epoch chờ nếu val không giảm

#     # KHỞI TẠO MODEL
#     model = YOLO(model_name)

#     # TIẾN HÀNH HUẤN LUYỆN
#     results = model.train(
#         data=data_path,
#         epochs=epochs,
#         imgsz=imgsz,
#         batch=batch,
#         device=device,
#         workers=workers,
#         pretrained=pretrained,

#         # Augmentation params
#         degrees=degrees,
#         scale=scale,
#         shear=shear,
#         flipud=flipud,
#         fliplr=fliplr,
#         mosaic=mosaic,
#         mixup=mixup,
#         hsv_h=hsv_h,
#         hsv_s=hsv_s,
#         hsv_v=hsv_v,

#         # Optimizer & scheduler
#         lr0=lr0,
#         lrf=lrf,
#         momentum=momentum,
#         weight_decay=weight_decay,
#         warmup_epochs=warmup_epochs,
#         patience=patience,
#     )

#     # HIỂN THỊ KẾT QUẢ
#     print("\nTraining completed successfully!")
#     print(f"Results saved to: {results.save_dir}")
#     print(f"Best model: {results.save_dir}/weights/best.pt")

# if __name__ == "__main__":
#     main()
# from ultralytics import YOLO

# def main():
    
#     # ============================
#     # CẤU HÌNH CƠ BẢN
#     # ============================
#     data_path = "D:/tabrec_yolo/data.yaml"   # Đường dẫn file dataset YAML
#     model_name = "yolov11s.pt"               # Mô hình backbone (small)
#     epochs = 50                              # Số vòng huấn luyện
#     imgsz = 800                              # Kích thước ảnh (cao hơn 640 để thấy chi tiết bảng)
#     batch = 8                                # Số ảnh mỗi batch (giữ nhỏ để tránh thiếu VRAM)
#     device = 0                               # GPU index hoặc 'cpu'
#     workers = 3                              # Số luồng đọc dữ liệu
#     pretrained = True                        # Sử dụng trọng số pretrained (fine-tune)

#     # ============================
#     # THAM SỐ AUGMENTATION
#     # Dành riêng cho "table detection in the wild"
#     # ============================
#     degrees = 8.0          # Góc xoay ngẫu nhiên (ảnh chụp bảng thường bị nghiêng)
#     scale = 0.7            # Phóng to/thu nhỏ vừa phải
#     shear = 3.0            # Biến dạng do góc chụp (shear)
#     flipud = 0.0           # Không lật dọc (vô lý với văn bản)
#     fliplr = 0.3           # Lật ngang nhẹ (đôi khi bảng chụp ngược chiều)
#     mosaic = 0.8           # Kết hợp 4 ảnh ngẫu nhiên (giúp mô hình học bố cục đa dạng)
#     mixup = 0.1            # Trộn nhẹ hai ảnh (giảm overfitting)
#     hsv_h = 0.03           # Biến thiên Hue (màu sắc ảnh)
#     hsv_s = 0.6            # Biến thiên Saturation (độ bão hòa)
#     hsv_v = 0.5            # Biến thiên Value (độ sáng/tối)

#     # ============================
#     # THAM SỐ HUẤN LUYỆN (OPTIMIZER)
#     # ============================
#     lr0 = 0.003            # Learning rate ban đầu (nhỏ hơn mặc định để fine-tune mượt hơn)
#     lrf = 0.01             # Learning rate cuối cùng
#     momentum = 0.937       # Momentum của optimizer
#     weight_decay = 0.0005  # Regularization
#     warmup_epochs = 2.0    # Số epoch khởi động (tăng dần learning rate)
#     patience = 50          # Số epoch chờ nếu không cải thiện

#     # ============================
#     # KHỞI TẠO MODEL
#     # ============================
#     model = YOLO(model_name)

#     # ============================
#     # TIẾN HÀNH HUẤN LUYỆN
#     # ============================
#     results = model.train(
#         data=data_path,
#         epochs=epochs,
#         imgsz=imgsz,
#         batch=batch,
#         device=device,
#         workers=workers,
#         pretrained=pretrained,

#         # Augmentation parameters
#         degrees=degrees,
#         scale=scale,
#         shear=shear,
#         flipud=flipud,
#         fliplr=fliplr,
#         mosaic=mosaic,
#         mixup=mixup,
#         hsv_h=hsv_h,
#         hsv_s=hsv_s,
#         hsv_v=hsv_v,

#         # Optimizer & scheduler parameters
#         lr0=lr0,
#         lrf=lrf,
#         momentum=momentum,
#         weight_decay=weight_decay,
#         warmup_epochs=warmup_epochs,
#         patience=patience,
#     )

#     
#     # KẾT QUẢ
#     print("\nTraining completed successfully!")
#     print(f"Results saved to: {results.save_dir}")
#     print(f"Best model: {results.save_dir}/weights/best.pt")


# if __name__ == "__main__":
#     main()

from ultralytics import YOLO

def main():
    
    # CẤU HÌNH CƠ BẢN
    data_path = "D:/tabrec_yolo/data.yaml"   # Đường dẫn file dataset YAML
    model_name = "yolo11s.pt"               # Mô hình backbone (small)
    epochs = 10                              # Số vòng huấn luyện
    imgsz = 640                              # Kích thước ảnh 
    batch = 8                                # Số ảnh mỗi batch (giữ nhỏ để tránh thiếu VRAM)
    device = 0                               # GPU index hoặc 'cpu'
    workers = 3                              # Số luồng đọc dữ liệu
    pretrained = True                        # Sử dụng trọng số pretrained (fine-tune)

    # THAM SỐ AUGMENTATION
    # Dành riêng cho "table detection in the wild"
    degrees = 8.0          # Góc xoay ngẫu nhiên (ảnh chụp bảng thường bị nghiêng)
    scale = 0.7            # Phóng to/thu nhỏ vừa phải
    shear = 3.0            # Biến dạng do góc chụp (shear)
    flipud = 0.0           # Không lật dọc (vô lý với văn bản)
    fliplr = 0.3           # Lật ngang nhẹ (đôi khi bảng chụp ngược chiều)
    mosaic = 0.8            # Kết hợp 4 ảnh ngẫu nhiên (giúp mô hình học bố cục đa dạng)
    mixup = 0.1            # Trộn nhẹ hai ảnh (giảm overfitting)
    hsv_h = 0.03           # Biến thiên Hue (màu sắc ảnh)
    hsv_s = 0.6            # Biến thiên Saturation (độ bão hòa)
    hsv_v = 0.5            # Biến thiên Value (độ sáng/tối)

    # THAM SỐ HUẤN LUYỆN (OPTIMIZER)
    lr0 = 0.003            # Learning rate ban đầu (nhỏ hơn mặc định để fine-tune mượt hơn)
    lrf = 0.01             # Learning rate cuối cùng
    momentum = 0.937       # Momentum của optimizer
    weight_decay = 0.0005  # Regularization
    warmup_epochs = 2.0    # Số epoch khởi động (tăng dần learning rate)
    patience = 50          # Số epoch chờ nếu không cải thiện

    # KHỞI TẠO MODEL
    model = YOLO(model_name)

    # TIẾN HÀNH HUẤN LUYỆN
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        pretrained=pretrained,
        degrees=degrees,
        scale=scale,
        shear=shear,
        flipud=flipud,
        fliplr=fliplr,
        mosaic=mosaic,
        mixup=mixup,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        patience=patience,
    )

    # KẾT QUẢ
    print("\nTraining completed successfully!")
    print(f"Results saved to: {results.save_dir}")
    print(f"Best model: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
