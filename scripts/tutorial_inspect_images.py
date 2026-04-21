import fitz
import sys

pdf_in = sys.argv[1]
doc = fitz.open(pdf_in)
page = doc[0]

# 在原来的基础上，我们看看如何获取图片的坐标
blocks = page.get_text("dict")["blocks"]

print("正在查找此 PDF 中的嵌入图像(如条形码、LOGO)...")
found = 0
for b in blocks:
    if b["type"] == 1: # type 0 是文字，type 1 是图像
        rect = fitz.Rect(b["bbox"])
        w, h = b["width"], b["height"]
        print(f"[图像 {found+1}] 坐标: {rect}, 尺寸: {w}x{h} px, 格式: {b['ext']}")
        found += 1
if found == 0:
    print("未发现嵌入图像。")
