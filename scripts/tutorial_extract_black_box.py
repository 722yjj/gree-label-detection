import fitz
import sys
import os

def extract_from_black_box(pdf_path):
    """
    提取黑框中的内容：
    1. 找到我们认为是“黑框”的矢量图形（Rect）。
    2. 找到所有与这个“黑框”坐标相交的文字，提取出来。
    """
    doc = fitz.open(pdf_path)
    page = doc[0] # 取第一页
    
    # 获取所有的矢量形状
    drawings = page.get_drawings()
    
    # 比如：寻找最大的黑色矩形框，或者特定尺寸的框
    # 这里的简单策略：过滤出颜色是黑色的线条/填充，并找出一个面积比较可观的。
    # 具体怎么找取决于实际 PDF (可能需要通过 visualize_pdf 观察后，直接写死坐标，或者通过逻辑判断)。
    black_boxes = []
    
    for d in drawings:
        rect = fitz.Rect(d["rect"])
        
        # 判断颜色是不是黑色 (0, 0, 0)
        # 有的可能没有颜色，或者有填充色
        color = d.get("color") 
        if color and all(c < 0.2 for c in color):
            # 将矩形存起来
            black_boxes.append(rect)
            
    if not black_boxes:
        print("未自动检测到明显的黑色矩形框。")
        print("你也可以直接使用刚刚通过 tutorial_visualize_pdf.py 确定的坐标：")
        """
        示例：假如你通过肉眼或者其他代码确认黑框坐标是 (x0=50, y0=100, x1=200, y1=150)
        你可以直接写:
        target_rect = fitz.Rect(50, 100, 200, 150)
        """
        return
        
    print(f"找到 {len(black_boxes)} 个黑色线条组成的边界框。")
    # 我们先以包含面积最大的“黑框”作为演示
    target_rect = max(black_boxes, key=lambda r: r.width * r.height)
    print(f"\n选定目标「黑框」坐标：")
    print(f"[{target_rect.x0:.1f}, {target_rect.y0:.1f}, {target_rect.x1:.1f}, {target_rect.y1:.1f}]")
    
    print("\n[开始提取黑框内部的文本...]")
    # 获取文本块
    blocks = page.get_text("blocks")
    extracted_texts = []
    for b in blocks:
        block_rect = fitz.Rect(b[:4])
        text = b[4].strip()
        
        if not text:
            continue
            
        # 如果文本框与我们的黑框相交面积大于0，或者包含在黑框内
        if target_rect.intersects(block_rect):
            # 相交了，我们就认为它是黑框里面的字
            extracted_texts.append(text)
            
    print("\n=============== 提取结果 ===============")
    for t in extracted_texts:
        print(t)
    print("========================================")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python tutorial_extract_black_box.py <输入PDF路径>")
        sys.exit(1)
        
    pdf_in = sys.argv[1]
    
    if not os.path.exists(pdf_in):
        print(f"错误: 找不到文件 {pdf_in}")
        sys.exit(1)
        
    extract_from_black_box(pdf_in)
