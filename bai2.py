import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Định nghĩa toán tử Sobel (Gx và Gy) để phát hiện biên
# Toán tử Sobel sử dụng hai kernel để phát hiện biên theo chiều ngang và chiều dọc.
Gx = np.array([[-1, 0, 1],  # Kernel cho biên theo chiều ngang
               [-2, 0, 2],
               [-1, 0, 1]])

Gy = np.array([[-1, -2, -1],  # Kernel cho biên theo chiều dọc
               [0,  0,  0],
               [1,  2,  1]])

# Định nghĩa bộ lọc Laplacian of Gaussian (LoG) để phát hiện biên
# LoG kết hợp lọc Gaussian với toán tử Laplacian để nhấn mạnh các vùng có sự thay đổi lớn trong ảnh.
LoG = np.array([[ 0,  0, -1,  0,  0],
                [ 0, -1, -2, -1,  0],
                [-1, -2, 16, -2, -1],  # Tâm của kernel được tăng cường với trọng số lớn hơn
                [ 0, -1, -2, -1,  0],
                [ 0,  0, -1,  0,  0]])

# Hàm để áp dụng phép tích chập giữa ảnh và kernel
def apply_filter(image, kernel):
    img_h, img_w = image.shape  # Lấy chiều cao và chiều rộng của ảnh
    kernel_h, kernel_w = kernel.shape  # Lấy chiều cao và chiều rộng của kernel
    pad_h, pad_w = kernel_h // 2, kernel_w // 2  # Tính toán kích thước padding cho ảnh
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')  # Padding ảnh
    
    output = np.zeros_like(image, dtype=np.float32)  # Khởi tạo ảnh đầu ra với kiểu dữ liệu float32 để có độ chính xác tốt hơn
    
    # Vòng lặp qua từng pixel trong ảnh
    for i in range(img_h):
        for j in range(img_w):
            region = padded_image[i:i+kernel_h, j:j+kernel_w]  # Lấy vùng ảnh tương ứng với kernel
            output[i, j] = np.sum(region * kernel)  # Tính tổng sản phẩm giữa vùng ảnh và kernel
    
    return output  # Trả về ảnh đã được lọc

# Hàm phát hiện biên Sobel
def sobel_edge_detection(image):
    sobel_x = apply_filter(image, Gx)  # Áp dụng toán tử Sobel theo chiều ngang
    sobel_y = apply_filter(image, Gy)  # Áp dụng toán tử Sobel theo chiều dọc
    sobel_edge = np.sqrt(sobel_x**2 + sobel_y**2)  # Tính độ lớn biên từ hai chiều
    return np.clip(sobel_edge, 0, 255)  # Giới hạn giá trị trong khoảng [0, 255] để đảm bảo hợp lệ

# Hàm Laplacian of Gaussian
def laplacian_of_gaussian(image):
    return np.clip(apply_filter(image, LoG), 0, 255)  # Áp dụng bộ lọc LoG và giới hạn giá trị

# Hàm để tải ảnh và chuyển đổi sang ảnh xám
def load_image(image_path):
    try:
        img = Image.open(image_path)  # Tải ảnh gốc
        img = img.convert('RGB')  # Đảm bảo ảnh ở chế độ RGB
        img_array = np.array(img)  # Chuyển đổi ảnh sang mảng numpy
        
        # Chuyển đổi RGB sang ảnh xám sử dụng công thức trọng số
        # Công thức trọng số phản ánh độ nhạy của mắt người với các màu:
        # 0.299 * R + 0.587 * G + 0.114 * B
        gray_image = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
        return img_array, gray_image.astype(np.uint8)  # Trả về cả ảnh gốc và ảnh xám
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None

# Hàm lưu ảnh đầu ra vào tệp
def save_image(image, output_path):
    output_img = Image.fromarray(image.astype(np.uint8))  # Đảm bảo đầu ra ở định dạng uint8
    output_img.save(output_path)  # Lưu ảnh

# Hiển thị ảnh gốc, ảnh xám và các ảnh đã xử lý
def display_images(original, gray_image, sobel_edges, log_edges):
    plt.figure(figsize=(20, 5))

    # Ảnh gốc
    plt.subplot(1, 4, 1)
    plt.imshow(original)
    plt.title('Ảnh Gốc')
    plt.axis('off')

    # Ảnh xám
    plt.subplot(1, 4, 2)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Ảnh Xám')
    plt.axis('off')

    # Phát hiện biên Sobel
    plt.subplot(1, 4, 3)
    plt.imshow(sobel_edges, cmap='gray')
    plt.title('Phát Hiện Biên Sobel')
    plt.axis('off')

    # Phát hiện biên Laplacian of Gaussian
    plt.subplot(1, 4, 4)
    plt.imshow(log_edges, cmap='gray')
    plt.title('Phát Hiện Biên LoG')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Hàm chính để thực hiện xử lý
def main(image_path):
    original_image, gray_image = load_image(image_path)  # Tải ảnh gốc và ảnh xám
    if original_image is None or gray_image is None:
        return  # Thoát nếu tải ảnh không thành công

    sobel_edges = sobel_edge_detection(gray_image)  # Áp dụng phát hiện biên Sobel
    log_edges = laplacian_of_gaussian(gray_image)  # Áp dụng phát hiện biên LoG

    display_images(original_image, gray_image, sobel_edges, log_edges)  # Hiển thị các ảnh

    save_image(sobel_edges, 'sobel_edges_output.jpg')  # Lưu kết quả Sobel
    save_image(log_edges, 'log_edges_output.jpg')      # Lưu kết quả LoG
    save_image(gray_image, 'gray_image_output.jpg')    # Lưu ảnh xám

# Thay thế với đường dẫn thực tế tới ảnh của bạn
image_path = '/XLA/BTXLA/input.jpg'
main(image_path)  # Thực hiện hàm chính
