from PIL import Image
import os
def test_crop_image(image_path, x_range, output_path):
    """测试函数：读取图片，按照x_range裁剪并保存到本地"""
    try:
        # 打开图片
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # 计算裁剪区域
        question_col_x_min = int(img_width * x_range[0])
        question_col_x_max = int(img_width * x_range[1])
        
        # 裁剪图片
        cropped_img = img.crop((question_col_x_min, img_height * 0.05, question_col_x_max, img_height * 0.94))
        
        # 保存裁剪后的图片
        cropped_img.save(output_path)
        print(f"Cropped image saved to {output_path}")
    
    except Exception as e:
        print(f"Error during cropping image: {e}")
test_crop_image('exam_pages\page_页面_027.jpg',(0.08, 0.12),'output.jpg')
# for root, dirs, files in os.walk('G:\project\examimage2text\img'):
#         for name in files:
#             file_path = os.path.abspath(os.path.join(root, name))
#             test_crop_image(file_path,(0, 1),file_path)