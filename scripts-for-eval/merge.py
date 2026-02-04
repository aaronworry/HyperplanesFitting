import fitz  # PyMuPDF的导入别名，核心库
from reportlab.lib.units import mm  # 仅用于便捷设置毫米边距，也可直接用数字（单位：点）

def merge_pdfs_to_single_page(
    input_paths,  # 待合并PDF路径列表
    output_path,  # 输出单页PDF路径
    rows=2,       # 排版行数
    cols=2,       # 排版列数
    margin=10*mm  # 页面边距（1mm≈2.835点，用数字则直接传点值，如28.35）
):
    # 1. 创建新的单页PDF文档（A4尺寸：595.28x841.89点，可自定义）
    new_pdf = fitz.open()
    a4_width, a4_height = 595.28, 841.89  # A4标准尺寸（点）
    new_page = new_pdf.new_page(width=a4_width, height=a4_height)  # 新建单页
    
    # 2. 计算单个PDF的可用绘制尺寸（均分页面，含间距，无重叠）
    single_width = (a4_width - 2 * margin - (cols - 1) * margin) / cols
    single_height = (a4_height - 2 * margin - (rows - 1) * margin) / rows

    # 3. 遍历所有PDF，逐页排版到新页面
    for idx, pdf_path in enumerate(input_paths):
        # 打开待合并的PDF，读取第一页
        src_pdf = fitz.open(pdf_path)
        src_page = src_pdf[0]  # 取第一页，需多页可修改索引（如src_pdf[1]）
        
        # 计算当前PDF的绘制坐标（左到右、上到下排版）
        col = idx % cols
        row = idx // cols
        x0 = margin + col * (single_width + margin)  # 绘制左上角X
        y0 = margin + row * (single_height + margin) # 绘制左上角Y
        x1 = x0 + single_width  # 绘制右下角X
        y1 = y0 + single_height # 绘制右下角Y
        
        # 核心：将源PDF页面绘制到新页面的指定矩形区域（自动适配尺寸，无变形）
        new_page.show_pdf_page(
            fitz.Rect(x0, y0, x1, y1),  # 绘制区域（矩形）
            src_pdf                 # 源PDF文档对象
        )
        src_pdf.close()  # 关闭源PDF，释放内存

    # 4. 保存最终的单页合并PDF
    new_pdf.save(output_path)
    new_pdf.close()
    print(f"合并完成！文件保存至：{output_path}")

# 示例调用（直接运行，替换为你的PDF路径即可）
if __name__ == "__main__":
    INPUT_PDFS = ["1.pdf", "2.pdf", "3.pdf"]  # 你的PDF文件列表
    OUTPUT_PDF = "merged_pymupdf.pdf"
    merge_pdfs_to_single_page(
        input_paths=INPUT_PDFS,
        output_path=OUTPUT_PDF,
        rows=3, cols=1,  # 2行2列排版，可修改为3行2列/1行4列等
        margin=3*mm     # 10毫米边距，可直接改数字（如20）表示20点
    )