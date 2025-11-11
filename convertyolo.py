#!/usr/bin/env python3
# cách chạy: python convertyolo.py --json_dir path/to/jsons --images_dir path/to/images --output_dir path/to/output
# vd:python convertyolo.py --json_dir "D:/TabRecSet (CurveTabSet)/TabRecSet (CurveTabSet)/TD annotation/english" --images_dir "D:/TabRecSet (CurveTabSet)/TabRecSet (CurveTabSet)/image/english_no-line" --output_dir "C:/Users/ploc/Downloads/tabrec_yolo"
#python convertyolo.py --json_dir "D:/TabRecSet (CurveTabSet)/TabRecSet (CurveTabSet)/TD annotation/chinese" --images_dir "D:/TabRecSet (CurveTabSet)/TabRecSet (CurveTabSet)/image/chinese_three-line" --output_dir "D:/tabrec_chinese"
import os, sys, argparse, shutil, random, json, glob
from PIL import Image

def parse_json(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        ann = json.load(f)
    fname = ann["imagePath"]
    h = ann["imageHeight"]
    w = ann["imageWidth"]

    bboxes = []
    for shape in ann.get("shapes", []):
        if shape["shape_type"] != "polygon":
            continue
        pts = shape["points"]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        bboxes.append((xmin, ymin, xmax, ymax))
    return fname, w, h, bboxes

def clamp01(x):
    return max(0.0, min(1.0, x))

def find_image(images_dir, imagePath):
    base = os.path.splitext(imagePath)[0]
    candidates = glob.glob(os.path.join(images_dir, base + ".*"))
    if candidates:
        return candidates[0]
    return None

def main(args):
    random.seed(42)

    if not os.path.isdir(args.json_dir):
        print("Annotation directory not found:", args.json_dir)
        sys.exit(1)

    json_files = [os.path.join(args.json_dir, f) for f in os.listdir(args.json_dir) if f.endswith(".json")]
    if not json_files:
        print("No JSON files found in", args.json_dir)
        sys.exit(1)

    print(f"Found {len(json_files)} JSON annotation files")

    # Split 70% train, 20% val, 10% predict
    random.shuffle(json_files)
    n = len(json_files)
    n_train = int(n * 0.7)
    n_val = int(n * 0.2)
    train_jsons = json_files[:n_train]
    val_jsons = json_files[n_train:n_train + n_val]
    predict_jsons = json_files[n_train + n_val:]

    out = args.output_dir
    for split in ["train", "val", "predict"]:
        os.makedirs(os.path.join(out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out, "labels", split), exist_ok=True)

    def process_list(file_list, split):
        copied = 0
        for json_file in file_list:
            fname, w, h, bboxes = parse_json(json_file)
            img_path = find_image(args.images_dir, fname)

            if not img_path or not os.path.exists(img_path):
                print("Missing image for:", fname)
                continue

            dst_img = os.path.join(out, "images", split, os.path.basename(img_path))
            dst_lbl = os.path.join(out, "labels", split, os.path.splitext(os.path.basename(fname))[0] + ".txt")

            shutil.copy2(img_path, dst_img)

            lines = []
            for (xmin, ymin, xmax, ymax) in bboxes:
                xc = (xmin + xmax) / 2.0 / w
                yc = (ymin + ymax) / 2.0 / h
                bw = (xmax - xmin) / w
                bh = (ymax - ymin) / h
                lines.append(f"0 {clamp01(xc):.6f} {clamp01(yc):.6f} {clamp01(bw):.6f} {clamp01(bh):.6f}")

            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            copied += 1
        print(f"{split}: {copied} images processed")

    process_list(train_jsons, "train")
    process_list(val_jsons, "val")
    process_list(predict_jsons, "predict")

    yaml_path = os.path.join(out, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"train: {os.path.abspath(os.path.join(out, 'images/train'))}\n")
        f.write(f"val: {os.path.abspath(os.path.join(out, 'images/val'))}\n")
        f.write(f"predict: {os.path.abspath(os.path.join(out, 'images/predict'))}\n\n")
        f.write("nc: 1\nnames: ['table']\n")

    print("Done. Dataset ready at:", os.path.abspath(out))
    print("Train YOLO with:")
    print(f"   yolo detect train data={yaml_path} model=yolov11s.pt epochs=100 imgsz=640")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", required=True, help="Path to JSON annotation folder")
    parser.add_argument("--images_dir", required=True, help="Path to images folder")
    parser.add_argument("--output_dir", default="dataset_yolo", help="Output YOLO dataset folder")
    args = parser.parse_args()
    main(args)
