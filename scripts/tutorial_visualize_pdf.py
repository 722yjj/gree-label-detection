import fitz  # PyMuPDF
import sys
import os

def visualize_pdf_elements(pdf_path, output_image_path):
    """
    可视化 PDF 中的核心元素：
    1. 文本块 (Text Blocks) - 蓝色框
    2. 矢量图形 (Drawings/Paths) - 红色框
    """
    print(f"正在打开 PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    page = doc[0]  # 通常我们先看第一页

    # 1. 获取所有矢量图形（如黑框、线条、表格边框等）
    # get_drawings() 返回一个字典列表，每个字典包含坐标 'rect' 和颜色信息
    drawings = page.get_drawings()
    print(f"-> 找到 {len(drawings)} 个矢量图形对象 (线条/框等)。")
    for d in drawings:
        rect = d["rect"]  # 获取该对象的边界框坐标 [x0, y0, x1, y1]
        # 在该坐标画一个红色边框，便于我们在图片上看到它
        page.draw_rect(rect, color=(1, 0, 0), width=1)

    # 2. 获取所有的文本块
    # get_text("dict") 或 ("blocks") 返回文本段落坐标
    blocks = page.get_text("blocks")
    print(f"-> 找到 {len(blocks)} 个文本段落块。")
    for b in blocks:
        # b[0] 到 b[3] 是坐标 (x0, y0, x1, y1)
        # b[4] 是文本内容
        rect = fitz.Rect(b[:4])
        # 用蓝色框标出所有文本块
        page.draw_rect(rect, color=(0, 0, 1), width=1)

    # 3. 将带有标记的页面保存为图片（分辨率为 150 DPI）
    print("正在渲染并保存带标记的图片...")
    pix = page.get_pixmap(dpi=150)
    pix.save(output_image_path)
    print(f"=== 完成！请打开 {output_image_path} 查看 ===\n")
    print("【教学提示】")
    print("在输出图片中：")
    print("- 你自己看到的所谓『黑框』，现在被红色线条（PDF的矢量图形）覆盖了。你可以根据这算出坐标。")
    print("- 所有的文字现在都被蓝色框包裹了。")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python tutorial_visualize_pdf.py <输入PDF路径> <输出图片路径>")
        sys.exit(1)
    
    pdf_in = sys.argv[1]
    img_out = sys.argv[2]
    
    if not os.path.exists(pdf_in):
        print(f"错误: 找不到文件 {pdf_in}")
        sys.exit(1)
        
    visualize_pdf_elements(pdf_in, img_out)
