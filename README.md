# FaceAnalyzer

## I. Giới thiệu

Module FaceAnalyzer này giải quyết bài toán nhận diện khuôn mặt trong một video đã cho trước. Cụ thể hơn, module này có thể xác định được khi nào một nhân vật xuất hiện và rời khỏi khung hình trong một video đã cho trước.

Module này gồm một class `Analyzer`, có khả năng nhận diện và nhận dạng khuôn mặt trong một video đã cho trước.

## II. Hướng dẫn sử dụng

### 1. Cài đặt các thư viện

Module này được code bằng Python 3.10; tuy vậy, module này không sử dụng những tính năng được giới thiệu sau Python 3.6 hoặc 3.7.

Module này sử dụng một số thư viện bên ngoài, bao gồm insightface, opencv, annoy và vidgear. Để cài đặt tất cả các thư viện mà module này sử dụng, ta có thể dùng lệnh

```bash
pip install -r requirements.txt
```

### 2. Sử dụng module

Để sử dụng class Analyzer của module này, bạn có thể import class Analyzer như sau

```python
from FaceAnalyzer import Analyzer
```

Với hình ảnh ở thư mục có tên `images`, và video cần phân tích ở path `video.mp4`, bạn có thể bắt đầu quá trình nhận diện và nhận dạng gương mặt như sau:

```py
analyzer = Analyzer()
analyzer.embedding_extract("images", "images")
analyzer.analyze("video.mp4", 'dossier', "video")
```

### 3. Chạy thử code

Để chạy thử file module_test trong module này, ta có thể sử dụng câu lệnh sau đây

```bash
python module_test.py <img_folder> <video_path> [<video_id>]
```

Ta cần phải truyền ít nhất 2 command-line arguments theo thứ tự như sau:

- img_folder: command-line argument đầu tiên sau video.py. Đây là thư mục chứa hình ảnh của các nhân vật mà ta cần nhận diện trong video.
- video_path: command-line argument thứ 2 sau video.py. Đây là path dẫn tới video mà chúng ta cần xử lý.
- video_id: command-line argument thứ 3 sau video.py, không bắt buộc. Đây là id của video; trong trường hợp chúng ta để trống, giá trị mặc định của video_id sẽ là giá trị của img_folder.

Notebook `notebook.ipynb` có thông tin chi tiết hơn về hai method có trong class `Analyzer`.

### 4. Lưu ý

#### Cấu trúc thư mục ảnh

Cấu trúc của thư mục chứa ảnh khuôn mặt các nhân vật mà ta cần nhận diện như sau:

```
<dataset>/
  A/
    A1.jpg
    A2.jpg
  B/
    B1.jpg
    B2.jpg
  ...
```

Khi sử dụng cấu trúc thư mục này, module sẽ hiểu là ảnh A1.jpg và A2.jpg chứa khuôn mặt của nhân vật A, B1.jpg và B2.jpg chứa khuôn mặt của nhân vật B, v.v..

#### Lưu ý về Insightface

Khi khởi tạo Insightface, thư viện này sẽ initialize tất cả các model nó nhìn thấy trong thư mục ~/.insightface, _kể cả những model ta không sử dụng._ Để tăng tốc quá trình khởi tạo Insightface, ta có thể làm một trong hai cách sau.

1. Xoá hết tất cả các file trong thư mục `~/.insightface/models/buffalo_l`, chỉ để lại hai file `det_10g.onnx` và `w600_r50.onnx`

2. Edit code của insightface: Trong file `ínsightface/app/face_analysis.py`, trong method `__init__` của class `FaceAnalysis`, sau dòng `model = model_zoo.get_model(onnx_file, **kwargs)`, thêm hai dòng nữa như sau:

   ```py
   if model.taskname != 'detection' and model.taskname != 'recognition':
       continue
   ```

### 5. Biên dịch về dạng mã nhị phân

Để biên dịch module FaceAnalyzer về dạng mã nhị phân, trong Terminal, chạy câu lệnh sau đây

```shell
python compile.py build_ext --inplace
```

## III. Cải tiến

So với đoạn code ban đầu, code trong file `video.py` có những cải tiến sau:

- Lọc bớt các model được sử dụng trong việc phân tích khuôn mặt trong `face_model`: thay vì sử dụng cả 5 model có sẵn, ta đã lọc bỏ những model không cần thiết cho việc nhận diện/nhận dạng khuôn mặt.
- Sử dụng lập trình hướng đối tượng: ta sử dụng class `Analyzer` để chứa hai method `analyze` và `embedding_extract`.
- Sử dụng thư viện `vidgear`: Bằng việc sử dụng thư viện này thay cho `opencv`, ta có thể làm cho việc đọc (và viết) video nhanh hơn.
