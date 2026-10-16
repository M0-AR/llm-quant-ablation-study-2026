"""make_visuals.py — generates README/Pages visuals from VERIFIED results/*.json (no hand numbers)."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path("/home/md/src/llm-quant-ablation-study-2026")
A = ROOT / "assets"
A.mkdir(exist_ok=True)
q = json.loads((ROOT / "results/01_quant.json").read_text())
fit = json.loads((ROOT / "results/02_fit.json").read_text())

# 1) quant ladder
labels = ["BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "IQ3_S", "PQ20\n7.21", "PTQ10\n5.95"]
sizes = [q["sizes"]["BF16"], q["sizes"]["Q8_0"], q["sizes"]["Q6_K"], q["sizes"]["Q5_K_M"],
         q["sizes"]["Q4_K_M"], q["sizes"]["IQ3_S"], 7.21, 5.95]
colors = ["#c0392b", "#e67e22", "#e67e22", "#f1c40f", "#27ae60", "#2980b9", "#8e44ad", "#8e44ad"]
plt.figure(figsize=(10, 5))
bars = plt.bar(labels, sizes, color=colors)
plt.ylabel("GB (27B model)")
plt.title("27B file size vs quantization — verified (size = P x bpw / 8)")
plt.axhline(8, ls="--", lw=1, label="8GB GPU")
plt.axhline(24, ls=":", lw=1, label="24GB GPU")
for b, s in zip(bars, sizes):
    plt.text(b.get_x() + b.get_width()/2, b.get_height()+0.6, f"{s:.1f}", ha="center", fontsize=9)
plt.legend()
plt.tight_layout()
plt.savefig(A / "quant_ladder.png", dpi=150)
plt.close()

# 2) headroom chart
tiers = fit["tiers"]
names = [t["tier"].split(":")[0] + ":" + t["tier"].split(":")[1][:18] for t in tiers]
hr = [t["headroom_gb"] for t in tiers]
cols = ["#27ae60" if h > 0.8 else "#c0392b" for h in hr]
plt.figure(figsize=(10, 5.2))
bars = plt.barh(names, hr, color=cols)
plt.xlabel("Headroom GB (VRAM - weights)")
plt.title("GPU headroom per pick — green fits with usable context, red does not")
plt.axvline(0.8, ls="--", lw=1)
plt.tight_layout()
plt.savefig(A / "fit_headroom.png", dpi=150)
plt.close()

# 3) demo GIF — shrinking file + refusals falling (uses verified bpw numbers)
frames = [
    ("54.0 GB\nBF16\n95/100 refuse", 54.0, 95),
    ("16.8 GB\nQ4_K_M\n12/100 refuse", 16.8, 12),
    ("11.8 GB\nIQ3_S mixed\n~1/100 refuse", 11.8, 1),
    ("7.21 GB\nternary+MTP\n~0/100 refuse", 7.21, 0),
    ("5.95 GB\nternary\n0/100 refuse", 5.95, 0),
]
imgs = []
for text, size, ref in frames:
    im = Image.new("RGB", (640, 360), (17, 24, 39))
    d = ImageDraw.Draw(im)
    try:
        f1 = ImageFont.load_default(size=28)
        f2 = ImageFont.load_default(size=20)
    except Exception:
        f1 = f2 = ImageFont.load_default()
    d.text((40, 40), "27B model: smaller file, fewer refusals", fill=(255, 255, 255), font=f1)
    # bar proportional to size (54GB = full width)
    w = int(560 * size / 54.0)
    d.rectangle([40, 130, 40 + w, 180], fill=(46, 204, 113))
    d.text((40, 190), text, fill=(255, 235, 130), font=f2)
    d.text((40, 280), f"{size:.2f} GB  |  refusals {ref}/100", fill=(160, 200, 255), font=f2)
    imgs.append(im)
imgs[0].save(A / "demo.gif", save_all=True, append_images=imgs[1:], duration=1100, loop=0)
print("wrote", sorted(p.name for p in A.iterdir()))
