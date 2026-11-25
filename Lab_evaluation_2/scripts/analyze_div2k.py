from pathlib import Path
import argparse, csv, math
import numpy as np

def stats(h):
    p=h.astype(np.float64)/h.sum()
    nz=p[p>0]
    mean=float(np.dot(np.arange(256),p))
    var=float(np.dot((np.arange(256)-mean)**2,p))
    return {"values":int(h.sum()),"non_empty_bins":int(np.count_nonzero(h)),
            "most_frequent_intensity":int(np.argmax(h)),"max_bin_population":int(h.max()),
            "min_bin_population":int(h[h>0].min()),"mean":mean,"std":math.sqrt(var),
            "entropy_bits":float(-(nz*np.log2(nz)).sum()),
            "coefficient_of_variation":math.sqrt(var)/mean if mean else 0.0}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",default="dataset/div2k_rgb_u8.bin")
    ap.add_argument("--output",default="dataset/div2k_histogram_analysis.csv")
    ap.add_argument("--chunk-mib",type=int,default=48);a=ap.parse_args()
    p=Path(a.input)
    if not p.exists(): raise SystemExit(f"Missing {p}")
    mm=np.memmap(p,dtype=np.uint8,mode="r"); combined=np.zeros(256,dtype=np.uint64)
    channels=[np.zeros(256,dtype=np.uint64) for _ in range(3)]
    chunk=a.chunk_mib*1024*1024
    # RGB bytes are interleaved; preserve channel alignment across chunk boundaries.
    step=chunk-(chunk%3)
    for start in range(0,len(mm),step):
        x=np.asarray(mm[start:min(start+step,len(mm))])
        combined+=np.bincount(x,minlength=256).astype(np.uint64)
        rgb=x[:len(x)//3*3].reshape(-1,3)
        for c in range(3): channels[c]+=np.bincount(rgb[:,c],minlength=256).astype(np.uint64)
        print(f"Processed {min(start+len(x),len(mm)):,}/{len(mm):,}")
    rows=[("combined_rgb",stats(combined))]
    rows += [(name,stats(h)) for name,h in zip(("red","green","blue"),channels)]
    with open(a.output,"w",newline="") as f:
        w=csv.writer(f);w.writerow(["channel","metric","value"])
        for name,s in rows:
            for k,v in s.items():w.writerow([name,k,v])
    print(f"Wrote {a.output}")
if __name__=="__main__":main()
