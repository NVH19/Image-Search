import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Định nghĩa Dataset để đọc ảnh
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    self.image_files.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_name 

# Biến đổi để chuẩn bị ảnh cho mô hình
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize về [-1, 1]
])

# Tải mô hình ResNet đã huấn luyện trước
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# Hàm trích xuất đặc trưng của ảnh
def extract_features(image, model):
    image = image.unsqueeze(0)  

    # Trích xuất đặc trưng từ mô hình
    with torch.no_grad():
        features = model(image)  
        features = features.view(features.size(0), -1) 
    return features

# Hàm tính toán độ tương đồng cosine giữa các đặc trưng
def find_similar_images(query_feature, train_features, train_filenames, top_k=50):
    similarities = []
    for i, feature in enumerate(train_features):
        sim = cosine_similarity(query_feature.numpy().reshape(1, -1), feature.numpy().reshape(1, -1))
        similarities.append((sim[0][0], train_filenames[i]))  

    # Sắp xếp theo độ tương đồng giảm dần và trả về k ảnh tương tự nhất
    similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities[:top_k]

# Dữ liệu huấn luyện 
train_dataset = ImageDataset(image_dir='data\image', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Trích xuất đặc trưng cho tất cả ảnh trong dữ liệu huấn luyện
train_features = []
train_filenames = []
for images, filenames in train_loader:
    with torch.no_grad():
        features = model(images)  # Trích xuất đặc trưng từ mô hình
        features = features.view(features.size(0), -1)  # Flatten đặc trưng
        train_features.append(features)
        train_filenames.extend(filenames)

train_features = torch.cat(train_features, dim=0)  # Kết hợp tất cả đặc trưng vào 1 tensor
# print(train_features[0])
# # Ví dụ: Tìm ảnh tương tự cho một ảnh truy vấn
# query_image_path = "data/img_test/ao-so-mi-ysl-hoa-tiet-da-bao-1-637683777866476556.jpg"
# query_image = Image.open(query_image_path).convert('RGB')

# # Tiền xử lý ảnh truy vấn
# query_image = transform(query_image)

# # Trích xuất đặc trưng cho ảnh truy vấn
# query_feature = extract_features(query_image, model)

# # Tìm ảnh tương tự
# similar_images = find_similar_images(query_feature, train_features, train_filenames, top_k=5)

# # Hiển thị kết quả
# for score, img_path in similar_images:
#     print(f"Similar image: {img_path} with similarity score: {score}")

# import matplotlib.pyplot as plt
# # Trực quan hóa ảnh
# def display_similar_images(similar_images, query_image_path):
#     # Hiển thị ảnh truy vấn
#     query_image = Image.open(query_image_path)
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 6, 1)
#     plt.imshow(query_image)
#     plt.title("Query Image")
#     plt.axis("off")

#     # Hiển thị các ảnh tương tự
#     for i, (score, img_path) in enumerate(similar_images):
#         img = Image.open(img_path)
#         plt.subplot(1, 6, i + 2)
#         plt.imshow(img)
#         plt.title(f"Score: {score:.4f}")
#         plt.axis("off")

#     plt.show()

# # Hiển thị kết quả
# display_similar_images(similar_images, query_image_path)
