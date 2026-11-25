from pathlib import Path
import argparse, csv, sys
from PIL import Image

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="dataset/DIV2K/DIV2K_train_HR")
    ap.add_argument("--output",default="dataset/div2k_rgb_u8.bin")
    ap.add_argument("--manifest",default="dataset/div2k_manifest.csv")
    args=ap.parse_args()
    src=Path(args.input); out=Path(args.output); manifest=Path(args.manifest)
    if not src.is_dir(): raise SystemExit(f"Missing image directory: {src}")
    out.parent.mkdir(parents=True,exist_ok=True)
    rows=[]; total_pixels=total_bytes=0
    with out.open("wb") as raw:
        for path in sorted(src.glob("*.png")):
            try:
                with Image.open(path) as im:
                    rgb=im.convert("RGB")
                    w,h=rgb.size; data=rgb.tobytes()
                raw.write(data); pixels=w*h; nbytes=len(data)
                rows.append((path.name,w,h,pixels,nbytes))
                total_pixels+=pixels; total_bytes+=nbytes
                print(f"{path.name}: {w}x{h}, {nbytes:,} bytes")
            except Exception as exc:
                print(f"SKIP {path}: {exc}",file=sys.stderr)
    with manifest.open("w",newline="") as f:
        wr=csv.writer(f); wr.writerow(["filename","width","height","pixels","bytes"]);wr.writerows(rows)
    print(f"Images={len(rows)} pixels={total_pixels:,} RGB_values={total_bytes:,}")
    print(f"Binary: {out} ({out.stat().st_size:,} bytes)")
if __name__=="__main__": main()
