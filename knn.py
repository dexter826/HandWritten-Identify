import os
import PIL
from PIL import ImageTk, Image as PILImage, ImageDraw
from tkinter import *
import numpy as np
import cv2  # vẫn giữ lại nếu cần dùng cho xử lý ảnh khác
import random

# --- Import TensorFlow và các module liên quan cho Deep Learning ---
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Kích thước canvas vẽ (300x300 pixels)
width = 300
height = 300
white = (255, 255, 255)
black = (0, 0, 0)

# ---------------------- Xây dựng & huấn luyện các model CNN ---------------------- #
def load_or_train_models():
    """
    Hàm tạo và huấn luyện 5 model khác nhau:
    1. Model cố tình bị overfit nghiêm trọng (không có kỹ thuật regularization + nhiều epochs)
    2. Model chỉ với Dropout
    3. Model chỉ với L2 Regularization
    4. Model chỉ với Early Stopping
    5. Model chỉ với Data Augmentation
    """
    models = {}
    
    # Kiểm tra và load các model đã lưu
    model_files = {
        "overfit_severe": "mnist_model_overfit_severe.h5",
        "dropout": "mnist_model_dropout.h5",
        "l2_reg": "mnist_model_l2_reg.h5",
        "early_stopping": "mnist_model_early_stopping.h5",
        "data_augment": "mnist_model_data_augment.h5"
    }
    
    # Xóa các model cũ nếu được yêu cầu training lại
    if os.path.exists("retrain_flag.txt"):
        for name, file_path in model_files.items():
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Đã xóa model {file_path} để huấn luyện lại")
        os.remove("retrain_flag.txt")
    
    # Load các model nếu đã tồn tại
    for name, file_path in model_files.items():
        if os.path.exists(file_path):
            print(f"Loading pre-trained {name} model...")
            models[name] = load_model(file_path)
    
    # Nếu đã có đủ 5 model, return luôn
    if len(models) == 5:
        return models
    
    # Nếu chưa có đủ model, huấn luyện
    print("Training missing models...")
    # Load dữ liệu MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Chọn một tập con nhỏ từ dữ liệu huấn luyện để tạo model overfit nghiêm trọng
    # Lấy 1000 mẫu cho mỗi chữ số để dễ overfit
    train_size_per_digit = 1000
    x_train_small = []
    y_train_small = []
    
    for digit in range(10):
        indices = np.where(y_train == digit)[0][:train_size_per_digit]
        x_train_small.extend(x_train[indices])
        y_train_small.extend(y_train[indices])
    
    # Chuyển về numpy array
    x_train_small = np.array(x_train_small)
    y_train_small = np.array(y_train_small)
    
    # Chuẩn hóa dữ liệu về [0,1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train_small = x_train_small.astype("float32") / 255.0
    
    # Thêm chiều kênh (số kênh = 1 đối với ảnh grayscale)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    x_train_small = np.expand_dims(x_train_small, -1)

    # Chuẩn bị data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    datagen.fit(x_train)
    
    # Early Stopping callback
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # 1. Model cố tình bị overfit nghiêm trọng
    if "overfit_severe" not in models:
        overfit_model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation='relu'),  # Thêm layer để tăng khả năng overfit
            Flatten(),
            Dense(256, activation='relu'),  # Tăng số neurons lên để dễ overfit
            Dense(10, activation='softmax')
        ])
        overfit_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        
        # Huấn luyện nhiều epochs trên bộ dữ liệu nhỏ để cố tình gây ra overfit nghiêm trọng
        print("Huấn luyện mô hình overfitting nghiêm trọng...")
        overfit_model.fit(x_train_small, y_train_small, batch_size=64, epochs=30, validation_data=(x_test, y_test))
        
        overfit_model.save("mnist_model_overfit_severe.h5")
        models["overfit_severe"] = overfit_model
    
    # 2. Model chỉ với Dropout
    if "dropout" not in models:
        dropout_model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.3),  # Thêm Dropout sau pooling
            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),   # Thêm Dropout sau fully connected
            Dense(10, activation='softmax')
        ])
        dropout_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        
        # Huấn luyện trên toàn bộ tập dữ liệu với Dropout
        print("Huấn luyện mô hình Dropout...")
        dropout_model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))
        
        dropout_model.save("mnist_model_dropout.h5")
        models["dropout"] = dropout_model
    
    # 3. Model chỉ với L2 Regularization
    if "l2_reg" not in models:
        weight_decay = 1e-4
        l2_model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1),
                  kernel_regularizer=l2(weight_decay)),
            Conv2D(64, kernel_size=(3, 3), activation='relu', 
                  kernel_regularizer=l2(weight_decay)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation='relu', 
                  kernel_regularizer=l2(weight_decay)),
            Flatten(),
            Dense(256, activation='relu', kernel_regularizer=l2(weight_decay)),
            Dense(10, activation='softmax')
        ])
        l2_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        
        # Huấn luyện với L2 Regularization
        print("Huấn luyện mô hình L2 Regularization...")
        l2_model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))
        
        l2_model.save("mnist_model_l2_reg.h5")
        models["l2_reg"] = l2_model
    
    # 4. Model chỉ với Early Stopping
    if "early_stopping" not in models:
        early_stop_model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(10, activation='softmax')
        ])
        early_stop_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        
        # Huấn luyện chỉ dùng Early Stopping
        print("Huấn luyện mô hình Early Stopping...")
        early_stop_model.fit(x_train, y_train, batch_size=128, epochs=30, 
                            validation_data=(x_test, y_test), 
                            callbacks=[early_stop])
        
        early_stop_model.save("mnist_model_early_stopping.h5")
        models["early_stopping"] = early_stop_model
    
    # 5. Model chỉ với Data Augmentation
    if "data_augment" not in models:
        data_augment_model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(10, activation='softmax')
        ])
        data_augment_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        
        # Huấn luyện chỉ dùng Data Augmentation
        print("Huấn luyện mô hình Data Augmentation...")
        train_generator = datagen.flow(x_train, y_train, batch_size=128)
        data_augment_model.fit(train_generator, steps_per_epoch=len(x_train) // 128,
                              epochs=10, validation_data=(x_test, y_test))
        
        data_augment_model.save("mnist_model_data_augment.h5")
        models["data_augment"] = data_augment_model
    
    return models

# ---------------------- Hàm xử lý nhận dạng chữ số ---------------------- #
def RUN():
    """
    Hàm RUN sẽ lưu ảnh hiện tại trên canvas, chuyển đổi về ảnh grayscale,
    resize về kích thước 28x28 (theo chuẩn của MNIST) và dự đoán chữ số bằng các model khác nhau.
    """
    # Xóa kết quả cũ
    box.delete(1.0, END)
    
    # Lưu ảnh hiện tại tạm thời
    filename = "image.png"
    image1.save(filename)

    # Mở ảnh, chuyển về grayscale và resize về 28x28
    image = PILImage.open(filename).convert("L")
    new_image = image.resize((28, 28), PILImage.LANCZOS)
    
    # Lưu ảnh đã resize để hiển thị
    resized_filename = "resized_image.png"
    new_image.save(resized_filename)
    
    # Hiển thị ảnh đã resize
    display_resized_image(resized_filename)
    
    # Chuyển ảnh thành mảng numpy và chuẩn hóa về [0,1]
    image_array = np.array(new_image).astype("float32") / 255.0

    # Nếu cần đảo màu (nến nền trắng, số đen), uncomment dòng dưới.
    # image_array = 1.0 - image_array

    # Định dạng lại mảng cho model: (1, 28, 28, 1)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = np.expand_dims(image_array, axis=-1)

    # Thêm nhiễu cho hình vẽ nếu được yêu cầu
    if add_noise_var.get():
        # Tạo bản sao để giữ nguyên ảnh gốc
        noisy_image_array = image_array.copy()
        # Thêm nhiễu Gaussian
        noise = np.random.normal(0, 0.1, noisy_image_array.shape)
        noisy_image_array = noisy_image_array + noise
        # Giới hạn giá trị trong khoảng [0,1]
        noisy_image_array = np.clip(noisy_image_array, 0, 1)
        
        # Lưu và hiển thị ảnh nhiễu
        noisy_img = PILImage.fromarray((noisy_image_array[0, :, :, 0] * 255).astype(np.uint8))
        noisy_filename = "noisy_image.png"
        noisy_img.save(noisy_filename)
        display_noisy_image(noisy_filename)
        
        # Dùng ảnh nhiễu để dự đoán
        pred_image = noisy_image_array
    else:
        pred_image = image_array

    # Dự đoán chữ số bằng từng model
    results = {}
    for name, model in models.items():
        prediction = model.predict(pred_image, verbose=0)
        digit = np.argmax(prediction)
        confidence = float(prediction[0][digit])
        results[name] = {"digit": int(digit), "confidence": confidence}
    
    # Hiển thị kết quả từ mỗi model
    box.delete(1.0, END)
    box.insert(END, "KẾT QUẢ NHẬN DẠNG:\n\n", "highlight")
    
    # Tiêu đề cho các mô hình
    model_names = {
        "overfit_severe": "Mô hình Overfit Nghiêm trọng",
        "dropout": "Mô hình với Dropout",
        "l2_reg": "Mô hình với L2 Regularization",
        "early_stopping": "Mô hình với Early Stopping", 
        "data_augment": "Mô hình với Data Augmentation"
    }
    
    # Tìm chữ số được dự đoán nhiều nhất từ các mô hình có regularization
    reg_predictions = {}
    for name, result in results.items():
        if name != "overfit_severe":  # Loại bỏ mô hình overfit
            digit = result['digit']
            if digit not in reg_predictions:
                reg_predictions[digit] = 0
            reg_predictions[digit] += 1
    
    # Chữ số được dự đoán nhiều nhất từ các mô hình regularization
    most_common_digit = max(reg_predictions.items(), key=lambda x: x[1])[0] if reg_predictions else None
    
    # Sắp xếp các mô hình theo độ tin cậy
    confidence_ranks = [(name, results[name]["confidence"]) for name in results]
    confidence_ranks.sort(key=lambda x: x[1], reverse=True)  # Sắp xếp giảm dần theo độ tin cậy
    
    # Hiển thị kết quả theo xếp hạng độ tin cậy
    box.insert(END, "XẾP HẠNG THEO ĐỘ TIN CẬY:\n\n", "highlight")
    
    # Các tag cho từng hạng
    rank_tags = ["rank1", "rank2", "rank3", "rank4", "rank5"]
    
    for i, (name, confidence) in enumerate(confidence_ranks):
        digit = results[name]["digit"]
        model_display_name = model_names[name]
        
        # Chọn tag dựa trên thứ hạng
        rank_tag = rank_tags[min(i, len(rank_tags)-1)]
        
        # Hiển thị thứ hạng, tên mô hình và độ tin cậy
        box.insert(END, f"{i+1}. {model_display_name}\n", rank_tag)
        box.insert(END, f"   • Dự đoán: {digit}\n")
        box.insert(END, f"   • Độ tin cậy: {confidence:.4f}\n")
        
        # Thêm thông tin về việc dự đoán có khớp với đa số hay không
        if most_common_digit is not None:
            if digit == most_common_digit:
                box.insert(END, f"   • Khớp với dự đoán đa số: Có\n")
            else:
                box.insert(END, f"   • Khớp với dự đoán đa số: Không\n")
        
        box.insert(END, "\n")
    
    # Hiển thị kết luận ngắn gọn
    if most_common_digit is not None:
        box.insert(END, f"Đa số các mô hình dự đoán đây là chữ số: {most_common_digit}\n\n", "highlight")

# ---------------------- Các hàm hỗ trợ giao diện ---------------------- #
def CLEAR():
    """Xóa nội dung của canvas và khung Text hiển thị kết quả."""
    box.delete(1.0, END)
    box.insert(END, "Vẽ một chữ số và nhấn NHẬN DẠNG để bắt đầu...", "highlight")
    
    global image1, draw
    cv.delete("all")
    image1 = PIL.Image.new("RGB", (canvas_width, canvas_height), black)
    draw = ImageDraw.Draw(image1)
    
    # Xóa các ảnh được hiển thị
    clear_displayed_images()

def paint(event):
    """Vẽ nét vẽ trên canvas và cập nhật lên ảnh PIL."""
    x1, y1 = (event.x - 3), (event.y - 3)
    x2, y2 = (event.x + 3), (event.y + 3)
    cv.create_line(x1, y1, x2, y2, fill="white", width=15)
    draw.line([x1, y1, x2, y2], fill="white", width=15)

def RETRAIN():
    """Hàm tạo flag để huấn luyện lại các model khi chạy lại chương trình."""
    with open("retrain_flag.txt", "w") as f:
        f.write("retrain")
    box.insert(END, "Đã đặt cờ huấn luyện lại. Các model sẽ được huấn luyện lại khi khởi động lại ứng dụng.\n")

def display_resized_image(filename):
    """Hiển thị ảnh đã resize."""
    try:
        # Xóa ảnh cũ nếu có
        if hasattr(display_resized_image, "label") and display_resized_image.label:
            display_resized_image.label.destroy()
            
        # Mở và hiển thị ảnh mới
        img = PILImage.open(filename)
        img = img.resize((120, 120), PILImage.LANCZOS)  # Phóng to để dễ nhìn
        img_tk = ImageTk.PhotoImage(img)
        
        # Tạo label mới và lưu tham chiếu
        display_resized_image.label = Label(processed_frame, image=img_tk, bd=2, relief=RIDGE)
        display_resized_image.label.image = img_tk  # Giữ tham chiếu
        display_resized_image.label.pack(side=LEFT, padx=10, pady=5)
        
        # Thêm label text
        if hasattr(display_resized_image, "text_label") and display_resized_image.text_label:
            display_resized_image.text_label.destroy()
        display_resized_image.text_label = Label(processed_frame, text="Ảnh 28x28", 
                                              font=normal_font, bg=bg_color, fg=text_color)
        display_resized_image.text_label.pack(side=LEFT, padx=0, pady=5)
        
    except Exception as e:
        print(f"Lỗi hiển thị ảnh: {e}")

def display_noisy_image(filename):
    """Hiển thị ảnh đã thêm nhiễu."""
    try:
        # Xóa ảnh cũ nếu có
        if hasattr(display_noisy_image, "label") and display_noisy_image.label:
            display_noisy_image.label.destroy()
            
        # Mở và hiển thị ảnh mới
        img = PILImage.open(filename)
        img = img.resize((120, 120), PILImage.LANCZOS)  # Phóng to để dễ nhìn
        img_tk = ImageTk.PhotoImage(img)
        
        # Tạo label mới và lưu tham chiếu
        display_noisy_image.label = Label(processed_frame, image=img_tk, bd=2, relief=RIDGE)
        display_noisy_image.label.image = img_tk  # Giữ tham chiếu
        display_noisy_image.label.pack(side=LEFT, padx=10, pady=5)
        
        # Thêm label text
        if hasattr(display_noisy_image, "text_label") and display_noisy_image.text_label:
            display_noisy_image.text_label.destroy()
        display_noisy_image.text_label = Label(processed_frame, text="Ảnh nhiễu", 
                                            font=normal_font, bg=bg_color, fg=text_color)
        display_noisy_image.text_label.pack(side=LEFT, padx=0, pady=5)
        
    except Exception as e:
        print(f"Lỗi hiển thị ảnh nhiễu: {e}")

def clear_displayed_images():
    """Xóa các ảnh được hiển thị."""
    # Xóa ảnh resize
    if hasattr(display_resized_image, "label") and display_resized_image.label:
        display_resized_image.label.destroy()
        display_resized_image.label = None
    if hasattr(display_resized_image, "text_label") and display_resized_image.text_label:
        display_resized_image.text_label.destroy()
        display_resized_image.text_label = None
        
    # Xóa ảnh nhiễu
    if hasattr(display_noisy_image, "label") and display_noisy_image.label:
        display_noisy_image.label.destroy()
        display_noisy_image.label = None
    if hasattr(display_noisy_image, "text_label") and display_noisy_image.text_label:
        display_noisy_image.text_label.destroy()
        display_noisy_image.text_label = None

# ---------------------- Tạo giao diện người dùng ---------------------- #
root = Tk()
root.title("Nhóm 5 - Nhận Dạng Chữ Số - So Sánh Các Kỹ Thuật Chống Overfitting")

# Cấu hình để chạy full màn hình
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}+0+0")
root.state('zoomed')  # Chế độ full màn hình trên Windows

# Thiết lập màu nền và font chung
bg_color = "#f0f0f0"
accent_color = "#3498db"
button_color = "#2980b9"
text_color = "#2c3e50"
heading_font = ("Segoe UI", 14, "bold")
normal_font = ("Segoe UI", 11)
button_font = ("Segoe UI", 12, "bold")
root.configure(bg=bg_color)

# Tạo frame chính
main_frame = Frame(root, bg=bg_color)
main_frame.pack(fill=BOTH, expand=True, padx=20, pady=20)

# Frame bên trái cho vẽ và điều khiển
left_frame = Frame(main_frame, bg=bg_color, bd=2, relief=RIDGE)
left_frame.pack(side=LEFT, fill=BOTH, expand=False, padx=10, pady=10)

# Frame bên phải cho kết quả
right_frame = Frame(main_frame, bg=bg_color, bd=2, relief=RIDGE)
right_frame.pack(side=RIGHT, fill=BOTH, expand=True, padx=10, pady=10)

# Tiêu đề
title_label = Label(left_frame, text="NHẬN DẠNG CHỮ SỐ VIẾT TAY", 
                   font=("Segoe UI", 16, "bold"), bg=bg_color, fg=accent_color)
title_label.pack(pady=10)

# Canvas để vẽ
canvas_frame = Frame(left_frame, bg="black", bd=2, relief=RAISED)
canvas_frame.pack(padx=10, pady=10)

# Tăng kích thước canvas
canvas_width = 300
canvas_height = 300
cv = Canvas(canvas_frame, width=canvas_width, height=canvas_height, bg='black', cursor="pencil")
cv.pack()
cv.bind("<B1-Motion>", paint)

# Khởi tạo ảnh và công cụ để vẽ
image1 = PIL.Image.new("RGB", (canvas_width, canvas_height), black)
draw = ImageDraw.Draw(image1)

# Frame cho các nút điều khiển
control_frame = Frame(left_frame, bg=bg_color)
control_frame.pack(fill=X, padx=10, pady=10)

# Nút với thiết kế đẹp hơn
recognize_btn = Button(control_frame, text="NHẬN DẠNG", font=button_font, 
                     bg=button_color, fg="white", padx=15, pady=8, 
                     relief=RAISED, command=RUN)
recognize_btn.pack(side=LEFT, padx=10)

clear_btn = Button(control_frame, text="XÓA", font=button_font, 
                 bg=button_color, fg="white", padx=15, pady=8, 
                 relief=RAISED, command=CLEAR)
clear_btn.pack(side=LEFT, padx=10)

# Frame cho ảnh đã xử lý
processed_frame = Frame(left_frame, bg=bg_color)
processed_frame.pack(fill=X, padx=10, pady=10)

# Checkbox thêm nhiễu với thiết kế đẹp hơn
noise_frame = Frame(left_frame, bg=bg_color)
noise_frame.pack(fill=X, padx=10, pady=5)

add_noise_var = BooleanVar()
add_noise_var.set(False)
Checkbutton(noise_frame, text="Thêm nhiễu vào hình ảnh", variable=add_noise_var, 
           font=normal_font, bg=bg_color, fg=text_color).pack(side=LEFT)

# Nút RETRAIN
retrain_btn = Button(left_frame, text="HUẤN LUYỆN LẠI", font=normal_font, 
                   bg="#e74c3c", fg="white", padx=10, pady=5, 
                   relief=RAISED, command=RETRAIN)
retrain_btn.pack(pady=10)

# Tiêu đề kết quả
result_title = Label(right_frame, text="KẾT QUẢ NHẬN DẠNG", 
                    font=("Segoe UI", 16, "bold"), bg=bg_color, fg=accent_color)
result_title.pack(pady=10)

# Thiết lập màu cho Text widget
box = Text(right_frame, width=60, height=30, font=normal_font, bg="white", fg=text_color)
box.pack(fill=BOTH, expand=True, padx=15, pady=10)
box.tag_configure("highlight", foreground=accent_color, font=("Segoe UI", 12, "bold"))
box.tag_configure("overfit_wrong", foreground="#e74c3c", font=("Segoe UI", 11, "bold"))
box.tag_configure("overfit_correct", foreground="#2ecc71", font=("Segoe UI", 11, "bold"))
box.tag_configure("regular_correct", foreground="#2ecc71", font=("Segoe UI", 11))
box.tag_configure("regular_wrong", foreground="#e74c3c", font=("Segoe UI", 11))
box.tag_configure("rank1", foreground="#2ecc71", font=("Segoe UI", 12, "bold"))
box.tag_configure("rank2", foreground="#27ae60", font=("Segoe UI", 11, "bold"))
box.tag_configure("rank3", foreground="#f39c12", font=("Segoe UI", 11))
box.tag_configure("rank4", foreground="#e67e22", font=("Segoe UI", 11))
box.tag_configure("rank5", foreground="#e74c3c", font=("Segoe UI", 11))

# Thêm scrollbar cho khung Text
scrollbar = Scrollbar(right_frame, command=box.yview)
scrollbar.pack(side=RIGHT, fill=Y)
box.config(yscrollcommand=scrollbar.set)

# ---------------------- Load hoặc huấn luyện model ---------------------- #
# Khởi tạo các biến toàn cục cho việc hiển thị ảnh
display_resized_image.label = None
display_resized_image.text_label = None
display_noisy_image.label = None
display_noisy_image.text_label = None

models = load_or_train_models()

# Hiển thị thông báo chào mừng
box.insert(END, "CHÀO MỪNG ĐẾN VỚI ỨNG DỤNG NHẬN DẠNG CHỮ SỐ VIẾT TAY\n\n", "highlight")
box.insert(END, "Ứng dụng này so sánh hiệu quả của các kỹ thuật chống Overfitting\ntrong Deep Learning khi nhận dạng chữ số viết tay.\n\n")
box.insert(END, "Các mô hình được sử dụng:\n")
box.insert(END, "1. Mô hình Overfit Nghiêm trọng\n", "rank5")
box.insert(END, "2. Mô hình với Dropout\n", "rank2")
box.insert(END, "3. Mô hình với L2 Regularization\n", "rank3")
box.insert(END, "4. Mô hình với Early Stopping\n", "rank4")
box.insert(END, "5. Mô hình với Data Augmentation\n\n", "rank1")
box.insert(END, "Hướng dẫn sử dụng:\n")
box.insert(END, "- Vẽ một chữ số từ 0-9 vào ô màu đen\n")
box.insert(END, "- Nhấn nút NHẬN DẠNG để xem kết quả\n")
box.insert(END, "- Có thể bật tùy chọn thêm nhiễu để kiểm tra độ bền của mô hình\n\n")
box.insert(END, "Vẽ một chữ số và nhấn NHẬN DẠNG để bắt đầu...\n")

root.mainloop()
