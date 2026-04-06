from ultralytics import YOLO

def main():
    model = YOLO("E:/DoAnTongHop/fix.yaml")
    results = model.train(
        data="E:/DoAnTongHop/tabrec_yolo/data.yaml", 
        epochs=100,           
        imgsz=640,            
        batch=8,              
        workers=4,            
        device=0,             
        lr0=0.01,             
        optimizer="SGD",      # hoặc "AdamW" nếu muốn fine-tune
        pretrained=False,     # vì train từ backbone YAML
        project="E:/DoAnTongHop/runs",   # thư mục lưu kết quả
        name="train_custom_backbone",    # tên subfolder trong project
        verbose=True
    )

    print("Training completed! Check results in runs/train_custom_backbone/")

if __name__ == "__main__":
    main()
