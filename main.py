from PIL import Image
import re
import os
import glob
import numpy as np
from paddleocr import PaddleOCR

# 包含多个页面图片的输入目录
IMAGE_DIR = 'exam_pages' 
# 图片文件匹配模式 (确保按页码顺序)
IMAGE_PATTERN = 'page*.jpg'  # 例如 page01.jpg, page02.jpg ... 或 scan_*.png
# 输出分割后图片的文件夹
OUTPUT_DIR_IMAGES = 'split_questions_multi'
# 输出包含所有文本的TXT文件路径
OUTPUT_TEXT_FILE = 'all_questions_text.txt'
# 用于识别题号的正则表达式 (例如: 02., 03., 4., 5.) 有时候点会被识别为逗号
question_num_pattern = re.compile(r'^(\d{1,2})[\.\,]$')
# 题号大致位于图片左侧的X坐标范围比例 (0.0 到 0.2 表示左侧20%)
question_col_x_range = (0.08, 0.12)

# 裁剪时在题号上方保留的像素
crop_margin_top = 10
# 最小有效题目区域高度（像素）
min_question_height = 20

lang_ocr = 'ch'
# 是否使用GPU (如果安装了GPU版本的paddlepaddle且有兼容GPU)
use_gpu_ocr = True

found_debug = False

temp_image_path = './temp.jpg'


# --- 辅助函数：垂直拼接图片 ---
def stack_images_vertically(image_list):
    """
    垂直拼接PIL Image对象列表。
    处理不同宽度的图片，以最宽图片的宽度为准，用白色填充较窄图片。
    """
    if not image_list:
        return None

    arrays = [np.array(img.convert('RGB')) for img in image_list] # 统一转为RGB

    max_width = max(arr.shape[1] for arr in arrays)

    padded_arrays = []
    for arr in arrays:
        h, w, c = arr.shape
        if w < max_width:
            # 创建白色填充 (RGB: 255, 255, 255)
            pad_width = max_width - w
            # 在右侧填充
            padding = np.full((h, pad_width, c), 255, dtype=arr.dtype)
            padded_arr = np.hstack((arr, padding))
            padded_arrays.append(padded_arr)
        else:
            padded_arrays.append(arr)

    # 垂直堆叠
    try:
      combined_array = np.vstack(padded_arrays)
      return Image.fromarray(combined_array)
    except ValueError as e:
        print(f"Error stacking images: {e}")
        # 尝试打印各个数组的形状以进行调试
        # for i, arr in enumerate(padded_arrays):
        #    print(f"Array {i} shape: {arr.shape}")
        return None # 返回None表示拼接失败


# --- 辅助函数：在单张图片上查找题号标记 ---
def find_question_markers_on_page(img, page_index, img_width, img_height, pattern, x_range, ocr_engine):
    """在单张图片上查找题号标记并返回其信息"""
    markers = []
    question_col_x_min = int(img_width * x_range[0])
    question_col_x_max = int(img_width * x_range[1])
    question_col_y_max = int(img_height)

    try:
        # 裁剪图片到指定的X范围
        cropped_img = img.crop((question_col_x_min, 0, question_col_x_max, question_col_y_max))
        cropped_img.save(temp_image_path)
        # 在裁剪后的图片上进行OCR
        
        result = ocr_engine.ocr(temp_image_path, det=True)
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                text = line[1][0]
                if text[-1] not in ['.',','] and len(text) == 2:
                    text += '.'
                match = pattern.match(text)
                if found_debug:
                    print(f'当前页面识别结果:{text}')
                x = line[0][0][0] + question_col_x_min
                y = line[0][0][1]
                try:
                    num_int = int(match.group(1))
                except ValueError:
                    num_int = float('inf')  # 非数字题号排在后面
                
                current_item = {
                    'number_str': match.group(1),  # 用于文件名和字典键
                    'number_int': num_int,       # 用于排序
                    'y': y,
                    'page_index': page_index,
                    'text': text
                }
                markers.append(current_item)
                print(f"  Page {page_index}: Found potential marker '{text}' (Num: {match.group(1)}) at Y={y}, X={x}")  # 调试信息
    except Exception as e:
        print(f"Error during OCR on page {page_index}: {e}")
    # 对当前页的标记按Y坐标排序，并进行简单去重（同一行附近可能识别出多个）
    markers.sort(key=lambda m: m['y'])

    temp_page_markers = {} # 按题号去重，保留Y最小的
    for item in markers:
         num_str = item['number_str']
         if num_str not in temp_page_markers or item['y'] < temp_page_markers[num_str]['y']:
             temp_page_markers[num_str] = item
    
    # 再次排序
    sorted_unique_markers = sorted(list(temp_page_markers.values()), key=lambda item: item['y'])

    return sorted_unique_markers


# --- 主程序 ---
def process_exam_paper():
    print("开始处理试卷...")

    # 1. 查找并排序输入图片文件
    image_files = sorted(glob.glob(os.path.join(IMAGE_DIR, IMAGE_PATTERN)))
    if not image_files:
        print(f"错误：在目录 '{IMAGE_DIR}' 中未找到匹配 '{IMAGE_PATTERN}' 的图片文件。")
        return

    print(f"找到 {len(image_files)} 张图片页面: {', '.join(os.path.basename(p) for p in image_files)}")

    # 2. 加载所有图片并查找所有题号标记
    all_markers = []
    loaded_images = []
    print("正在加载图片并查找题号标记...")
    ocr_engine = PaddleOCR(lang='en', use_gpu=use_gpu_ocr, show_log=False)
    for i, img_path in enumerate(image_files):
        try:
            img = Image.open(img_path)
            loaded_images.append(img)
            w, h = img.size
            print(f"  处理页面 {i+1}/{len(image_files)} ({os.path.basename(img_path)})...")
            markers_on_page = find_question_markers_on_page(img, i, w, h, question_num_pattern, question_col_x_range, ocr_engine)
            all_markers.extend(markers_on_page)
        except FileNotFoundError:
            print(f"错误: 无法找到图片文件 '{img_path}'")
            return
        except Exception as e:
            print(f"加载或处理图片 '{img_path}' 时出错: {e}")
            return # or continue? Decide based on desired robustness

    # 3. 按页面索引和Y坐标全局排序标记
    all_markers.sort(key=lambda m: (m['page_index'], m['y']))

    if not all_markers:
        print("\n警告：在所有页面上均未能识别到任何符合条件的题号。请检查配置和图片。")
        # (可以加入之前版本中的详细检查提示)
        return

    print(f"\n总共找到 {len(all_markers)} 个题号标记。开始分割和合并...")
    if found_debug:
        return
    # 4. 创建输出目录
    os.makedirs(OUTPUT_DIR_IMAGES, exist_ok=True)

    # 5. 处理标记，生成最终图片并提取文本
    output_texts = {}
    num_markers = len(all_markers)
    ocr_engine = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=use_gpu_ocr, show_log=False)
    for i in range(num_markers):
        current_marker = all_markers[i]
        question_num_str = current_marker['number_str']
        print(f"\n处理题号 {question_num_str} (标记 {i+1}/{num_markers})...")

        # 确定当前题目的起始位置
        start_page_index = current_marker['page_index']
        start_y = current_marker['y']
        
        # 向上微调起始Y坐标
        crop_y_start_on_first_page = max(0, start_y - crop_margin_top)

        # 确定当前题目的结束位置（即下一个题号的起始位置，或文档末尾）
        end_page_index = -1
        end_y = -1

        if i + 1 < num_markers:
            next_marker = all_markers[i+1]
            end_page_index = next_marker['page_index']
            end_y = next_marker['y'] # 下一个题号的Y坐标是当前题目的结束边界
            print(f"  下一个题号 '{next_marker['number_str']}' 在 Page {end_page_index+1} at Y={end_y}")
        else:
            # 这是最后一个题号，区域延伸到最后一页的底部
            end_page_index = len(loaded_images) - 1
            end_y = loaded_images[end_page_index].height
            print(f"  这是最后一个题号，延伸到 Page {end_page_index+1} 的底部 (Y={end_y})")

        # --- 裁剪和拼接逻辑 ---
        images_to_stack = []
        current_page_img = loaded_images[start_page_index]
        current_page_width, current_page_height = current_page_img.size

        if start_page_index == end_page_index:
            # 情况1：题目完全在同一页内
            print(f"  题目在同一页 (Page {start_page_index+1}) 内，从 Y={crop_y_start_on_first_page} 到 Y={end_y}")
            crop_box = (0, crop_y_start_on_first_page, current_page_width, end_y)
            # 检查裁剪区域是否有效
            if crop_box[3] - crop_box[1] >= min_question_height:
                 img_crop = current_page_img.crop(crop_box)
                 images_to_stack.append(img_crop)
            else:
                 print(f"  警告: 题号 {question_num_str} 的分割区域过小，跳过。 Y={crop_box[1]} to {crop_box[3]}")

        else:
            # 情况2：题目跨越多页
            print(f"  题目跨页：从 Page {start_page_index+1} (Y={crop_y_start_on_first_page}) 到 Page {end_page_index+1} (Y={end_y})")
            # 2a. 截取起始页的下半部分
            crop_box_start = (0, crop_y_start_on_first_page, current_page_width, current_page_height)
            if crop_box_start[3] - crop_box_start[1] >= min_question_height/2 : # 至少有一点内容
                img_crop_start = current_page_img.crop(crop_box_start)
                images_to_stack.append(img_crop_start)
                print(f"    添加起始页部分: Page {start_page_index+1}, Y={crop_box_start[1]} to {crop_box_start[3]}")


            # 2b. 添加中间的完整页面 (如果存在)
            for page_idx in range(start_page_index + 1, end_page_index):
                print(f"    添加完整页面: Page {page_idx+1}")
                images_to_stack.append(loaded_images[page_idx])

            # 2c. 截取结束页的上半部分
            if end_page_index < len(loaded_images): # 确保结束页索引有效
                end_page_img = loaded_images[end_page_index]
                end_page_width, end_page_height = end_page_img.size
                crop_box_end = (0, 0, end_page_width, end_y)
                if crop_box_end[3] - crop_box_end[1] >= min_question_height/2: # 至少有一点内容
                    img_crop_end = end_page_img.crop(crop_box_end)
                    images_to_stack.append(img_crop_end)
                    print(f"    添加结束页部分: Page {end_page_index+1}, Y={crop_box_end[1]} to {crop_box_end[3]}")
            else:
                 print(f"  警告: 计算出的结束页索引 ({end_page_index}) 超出范围。")


        # --- 拼接、保存、OCR ---
        if images_to_stack:
            print(f"  正在拼接 {len(images_to_stack)} 个图片部分...")
            final_question_image = stack_images_vertically(images_to_stack)

            if final_question_image:
                # 保存拼接后的图片
                output_image_filename = os.path.join(OUTPUT_DIR_IMAGES, f"question_{question_num_str}.png")
                try:
                    final_question_image.save(output_image_filename)
                    print(f"  图片已保存: {output_image_filename}")
                except Exception as save_err:
                    print(f"  错误：无法保存图片 {output_image_filename}: {save_err}")


                # 对最终图片进行OCR以提取文本
                print(f"  对题号 {question_num_str} 的最终图片进行OCR...")
                try:
                    final_image_np = np.array(final_question_image.convert('RGB')) # Ensure RGB
                    ocr_result_final = ocr_engine.ocr(final_image_np, cls=True)

                    question_text = ""
                    # Check structure: ocr_result_final = [[line1_info, line2_info,...]]
                    if ocr_result_final and ocr_result_final[0]: # Check if the inner list is not empty
                         # Extract text from each line's tuple (text, conf)
                        lines = [line[1][0] for line in ocr_result_final[0] if line and line[1]]
                        question_text = "\n".join(lines).strip() # Join detected lines with newline
                    else:
                         question_text = "[No text detected by PaddleOCR]"

                    output_texts[question_num_str] = question_text
                    print(f"  OCR 完成 (题号 {question_num_str}).")
                except RuntimeError as timeout_error:
                    print(f"  OCR 超时或出错 (题号 {question_num_str}): {timeout_error}")
                    output_texts[question_num_str] = f"[OCR Failed or Timed Out for Question {question_num_str}]"
                except Exception as ocr_err:
                     print(f"  OCR时发生意外错误 (题号 {question_num_str}): {ocr_err}")
                     output_texts[question_num_str] = f"[OCR Error for Question {question_num_str}: {ocr_err}]"
            else:
                print(f"  警告: 题号 {question_num_str} 的图片拼接失败，跳过OCR。")
                output_texts[question_num_str] = f"[Image Stitching Failed for Question {question_num_str}]"
        else:
            print(f"  警告: 没有为题号 {question_num_str} 生成有效的图片部分，跳过处理。")
            output_texts[question_num_str] = f"[No Valid Image Parts Found for Question {question_num_str}]"


    # 6. 将所有提取的文本写入TXT文件
    print(f"\n正在将所有识别文本写入到: {OUTPUT_TEXT_FILE}")
    written_count = 0
    with open(OUTPUT_TEXT_FILE, 'w', encoding='utf-8') as f:
        # 尝试按数字顺序对键（题号）进行排序
        try:
            # 找到所有在all_markers中记录的题号顺序
            sorted_keys = [m['number_str'] for m in all_markers]
            # 去重同时保持顺序 (如果一个题号被错误识别多次)
            ordered_unique_keys = list(dict.fromkeys(sorted_keys))
        except:
            # 如果排序失败（例如题号不是纯数字），则按字符串排序
            ordered_unique_keys = sorted(output_texts.keys())

        for q_num in ordered_unique_keys:
            if q_num in output_texts:
              f.write(f"--- 题号 {q_num} ---\n")
              f.write(output_texts[q_num])
              f.write("\n\n")
              written_count += 1
            else:
              print(f"  警告: 未找到题号 {q_num} 的文本记录。")


    print(f"\n处理完成！共处理 {len(all_markers)} 个题号标记，")
    print(f"分割/合并的图片保存在 '{OUTPUT_DIR_IMAGES}' 文件夹中。")
    print(f"{written_count} 个题目的识别文本已写入 '{OUTPUT_TEXT_FILE}'。")


# --- 执行主程序 ---
if __name__ == "__main__":
    process_exam_paper()