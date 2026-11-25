from pathlib import Path
import argparse,csv,statistics
def main():
    ap=argparse.ArgumentParser();ap.add_argument("input");ap.add_argument("--output",default="results/summary.csv")
    ap.add_argument("--group",default="experiment");ap.add_argument("--metric",default="time_ms");a=ap.parse_args()
    groups={}
    with open(a.input,newline="") as f:
        for row in csv.DictReader(f):
            try: groups.setdefault(row[a.group],[]).append(float(row[a.metric]))
            except (KeyError,ValueError): continue
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with open(a.output,"w",newline="") as f:
        w=csv.writer(f);w.writerow(["group","count","mean","stddev","min","max"])
        for k,v in sorted(groups.items()):
            w.writerow([k,len(v),statistics.mean(v),statistics.stdev(v) if len(v)>1 else "",min(v),max(v)])
    print("Wrote",a.output)
if __name__=="__main__":main()
