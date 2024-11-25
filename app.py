from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch

from main import extract_features, transform,find_similar_images,train_features,train_filenames,model

app = Flask(__name__, static_url_path='/data', static_folder='data')


@app.route('/')
def index():
    return render_template('home.html')  
@app.route('/search', methods=['POST'])
def search_similar_images():
    # Nhận file ảnh từ request
    file = request.files['query_image']
    query_image = Image.open(file).convert('RGB')
    
    # Tiền xử lý và trích xuất đặc trưng ảnh truy vấn
    query_image = transform(query_image)
    query_feature = extract_features(query_image, model)
    
    # Tìm kiếm ảnh tương tự
    similar_images = find_similar_images(query_feature, train_features, train_filenames, top_k=50)
    
    # Trả về kết quả JSON
    return jsonify({"similar_images": [{"path": img_path, "score": float(score)} for score, img_path in similar_images]})


if __name__ == '__main__':
    app.run(debug=False)
