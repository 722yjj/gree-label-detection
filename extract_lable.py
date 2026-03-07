import fitz  # PyMuPDF
import os


def extract_red_box_info(pdf_path, target_dpi=300):
    # 1. 准备输出路径
    base_name = os.path.splitext(pdf_path)[0]

    if not os.path.exists(pdf_path):
        print(f"错误：找不到文件 {pdf_path}")
        return

    doc = fitz.open(pdf_path)

    for page_index, page in enumerate(doc):
        # 2. 寻找红色虚线框
        paths = page.get_drawings()
        target_rects = []

        for path in paths:
            color = path.get("color")
            dashes = path.get("dashes")

            # 颜色判定逻辑（RGB 红色）
            is_red = color and color[0] > 0.8 and color[1] < 0.2 and color[2] < 0.2
            # 虚线判定逻辑
            is_dashed = dashes is not None and dashes != "[]"

            if is_red and is_dashed:
                target_rects.append(path["rect"])

        # 3. 如果找到了符合条件的区域
        if target_rects:
            # 合并多个线段为一个大矩形
            final_rect = target_rects[0]
            for r in target_rects[1:]:
                final_rect |= r

            print(f"\n>>> 在第 {page_index + 1} 页发现目标区域: {final_rect}")

            # --- 步骤 A: 提取文字 ---
            # clip 参数将提取范围锁定在红框坐标内
            text_content = page.get_text("text", clip=final_rect).strip()

            # --- 步骤 B: 截取高分辨率图片 ---
            output_img_path = f"{base_name}_page{page_index+1}.png"
            pix = page.get_pixmap(clip=final_rect, dpi=target_dpi)
            pix.save(output_img_path)

            # --- 4. 结果输出 ---
            print(f"【文字内容提取自第 {page_index+1} 页】：")
            if text_content:
                print("-" * 30)
                print(text_content)
                print("-" * 30)

                # 可选：将提取的文字存入 txt 文件
                with open(
                    f"{base_name}_page{page_index+1}.txt", "w", encoding="utf-8"
                ) as f:
                    f.write(text_content)
            else:
                print("(该区域未检测到可直接读取的文字)")

            print(f"【图片已保存】：{output_img_path} ({target_dpi} DPI)")
            return output_img_path
    doc.close()


# --- 执行 ---
# 请确保 example.pdf 与此脚本在同一文件夹，或填写完整路径
# output_img_path = extract_red_box_info(r"600004075219-01.pdf", target_dpi=300)
# print(output_img_path)

# 示例：如果文件名为 "GWH24AGD.pdf"，输出将为 "GWH24AGD_page1.png"
# extract_content_by_red_dashed_box("600004075219.pdf")
