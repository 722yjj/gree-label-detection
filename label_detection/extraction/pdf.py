from pathlib import Path

import fitz  # PyMuPDF


def extract_red_box_info(pdf_path, target_dpi=300, output_dir=None):
    """Extract the dashed red-box area from a PDF page as an image."""
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        print(f"错误：找不到文件 {pdf_path}")
        return None

    output_root = Path(output_dir) if output_dir else pdf_path.parent
    output_root.mkdir(parents=True, exist_ok=True)
    base_name = pdf_path.stem

    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            target_rects = []

            for path in page.get_drawings():
                color = path.get("color")
                dashes = path.get("dashes")
                is_red = color and color[0] > 0.8 and color[1] < 0.2 and color[2] < 0.2
                is_dashed = dashes is not None and dashes != "[]"

                if is_red and is_dashed:
                    target_rects.append(path["rect"])

            if not target_rects:
                continue

            final_rect = target_rects[0]
            for rect in target_rects[1:]:
                final_rect |= rect

            print(f"\n>>> 在第 {page_index + 1} 页发现目标区域: {final_rect}")

            text_content = page.get_text("text", clip=final_rect).strip()
            output_img_path = output_root / f"{base_name}_page{page_index + 1}.png"
            output_txt_path = output_root / f"{base_name}_page{page_index + 1}.txt"

            pix = page.get_pixmap(clip=final_rect, dpi=target_dpi)
            pix.save(output_img_path)

            print(f"【文字内容提取自第 {page_index + 1} 页】：")
            if text_content:
                print("-" * 30)
                print(text_content)
                print("-" * 30)
                output_txt_path.write_text(text_content, encoding="utf-8")
            else:
                print("(该区域未检测到可直接读取的文字)")

            print(f"【图片已保存】：{output_img_path} ({target_dpi} DPI)")
            return str(output_img_path)

    return None
