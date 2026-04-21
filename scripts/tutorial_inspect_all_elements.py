import fitz  # PyMuPDF
import sys
import os

def inspect_all_pdf_elements(pdf_path, output_image_path):
    print(f"==================================================")
    print(f"正在全面解剖 PDF: {pdf_path}")
    print(f"==================================================\n")
    
    doc = fitz.open(pdf_path)
    page = doc[0]  # 我们以第一页为例分析

    # ==========================================
    # 1. 页面基础属性 (Page Properties)
    # ==========================================
    print("【1. 页面基础属性】")
    print(f"- page.rect (页面物理尺寸坐标): {page.rect}")
    print(f"  注: 这通常是类似 Rect(0.0, 0.0, 595.27, 841.89) 的值，代表 A4 纸大小的磅值(points)。")
    print(f"- page.rotation (页面旋转角度): {page.rotation} 度")
    print("\n----------------------------------------------------")

    # ==========================================
    # 2. 注释与超链接 (Annots & Links)
    # ==========================================
    print("【2. 浮层元素: 注释与链接 (可能为空)】")
    
    # 获取所有的链接
    links = page.get_links()
    print(f"- page.get_links() -> 找到 {len(links)} 个超链接。")
    for i, link in enumerate(links[:3]): # 只打印前3个避免刷屏
        print(f"  [{i}] 链接坐标: {link.get('from')}, 跳转目标: {link.get('uri') or link.get('page')}")
        # 在图上用紫色画出链接的包围盒
        page.draw_rect(link['from'], color=(0.5, 0, 0.5), width=2, dashes="[5]")
        
    # 获取所有的注释 (比如高亮、手写笔记、图章)
    annots = page.annots()
    annot_count = 0
    if annots:
        for annot in annots:
            annot_count += 1
            print(f"  [注释 {annot_count}] 类型: {annot.type[1]}, 坐标: {annot.rect}")
            # 在图上用绿色画出注释的包围盒
            page.draw_rect(annot.rect, color=(0, 1, 0), width=2, dashes="[5]")
    print(f"- 共有 {annot_count} 个额外注释 (Annots)。")
    print("\n----------------------------------------------------")

    # ==========================================
    # 3. 矢量图形 (Drawings/Paths)
    # ==========================================
    print("【3. 底层骨架: 矢量图形 (Drawings)】")
    drawings = page.get_drawings()
    print(f"- page.get_drawings() -> 找到 {len(drawings)} 个绘图路径。")
    print("  (通常是你看到的表格线、背景色块、LOGO曲线等)")
    
    if drawings:
        print("  👇 抽出第 1 个矢量图形看看它的“真面目”：")
        sample_draw = drawings[0]
        print(f"    包含以下属性: {list(sample_draw.keys())}")
        print(f"    - rect (图形边界框): {sample_draw['rect']}")
        print(f"    - type (绘制模式): '{sample_draw['type']}' (f=填充, s=描边, fs=填充并描边)")
        print(f"    - color (线条颜色): {sample_draw['color']}")
        print(f"    - width (线条粗细): {sample_draw['width']}")
        print(f"    - items (核心！教机器怎么画的路径指令):")
        for item in sample_draw['items']:
            print(f"        {item}")
            
    # 在图上用红色画出所有的矢量图形矩形框
    for d in drawings:
        page.draw_rect(d["rect"], color=(1, 0, 0), width=0.5)
    print("\n----------------------------------------------------")

    # ==========================================
    # 4. 文本块 (Text Blocks)
    # ==========================================
    print("【4. 可读内容: 文本块 (Text)】")
    blocks = page.get_text("blocks")
    print(f"- page.get_text('blocks') -> 找到 {len(blocks)} 个文本段落块。")
    
    if blocks:
        print("  👇 抽出前 2 个文本块看看：")
        for i, b in enumerate(blocks[:2]):
            rect = fitz.Rect(b[:4])
            text = b[4].strip()
            # 根据 \n 替换为可见字符以便在控制台查看换行情况
            print(f"    [{i+1}] 坐标: {rect}")
            print(f"        内容: {repr(text)}") # repr 会把换行显示为 \n
            
    # 在图上用蓝色画出所有的文字包围盒
    for b in blocks:
        page.draw_rect(fitz.Rect(b[:4]), color=(0, 0, 1), width=0.5)
    print("==================================================\n")

    # ==========================================
    # 5. 图片元素 (Images)
    # ==========================================
    print("【5. 像素内容: 图片元素 (Images)】")
    images = page.get_images(full=True)
    print(f"- page.get_images() -> 找到 {len(images)} 个图片元素。")
    
    if images:
        print("  👇 抽出前 2 个图片看看：")
        for i, img in enumerate(images[:2]):
            xref = img[0]
            width, height = img[2], img[3]
            colorspace = img[5]
            print(f"    [{i+1}] xref (内部引用号): {xref}")
            print(f"        分辨率: {width}x{height}, 色彩空间: {colorspace}")
            # 图片可能有多个位置 (被复用)，获取所有位置的包围盒
            try:
                rects = page.get_image_rects(xref)
                print(f"        在页面上的位置坐标: {[str(r) for r in rects]}")
            except Exception as e:
                print(f"        无法获取位置坐标: {e}")
            
    # 在图上用黄色画出所有的图片包围盒
    for img in images:
        xref = img[0]
        try:
            rects = page.get_image_rects(xref)
            for r in rects:
                page.draw_rect(r, color=(1, 1, 0), width=1)
        except Exception:
            pass
    print("==================================================\n")

    # ==========================================
    # 6. 生成可视化大图
    # ==========================================
    print("正在渲染并保存最终的“全身X光片”...")
    pix = page.get_pixmap(dpi=200) # 稍微提高清晰度
    pix.save(output_image_path)
    print(f"=== 完成！请打开 {output_image_path} 查看可视化结果 ===")
    print("\n[颜色图例说明]")
    print("🟥 红色细线：所有由线条/填充色块构成的矢量图 (Drawings)")
    print("🟦 蓝色细线：所有的文字块区域 (Text)")
    print("🟩 绿色虚线：PDF额外添加的标签/高亮注释 (Annots)")
    print("🟪 紫色虚线：可点击的超链接区域 (Links)")
    print("🟨 黄色实线：内嵌的位图/图片 (Images)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python tutorial_inspect_all_elements.py <输入PDF路径> <输出图片路径>")
        sys.exit(1)
    
    pdf_in = sys.argv[1]
    img_out = sys.argv[2]
    
    if not os.path.exists(pdf_in):
        print(f"错误: 找不到文件 {pdf_in}")
        sys.exit(1)
        
    inspect_all_pdf_elements(pdf_in, img_out)
