# OptimizeFace

## I. Giới thiệu

Module OptimizeFace này giải quyết bài toán nhận diện khuôn mặt trong một video đã cho trước. Cụ thể hơn, module này có thể xác định được khi nào một nhân vật xuất hiện và rời khỏi khung hình trong một video đã cho trước.

Module này gồm một class `Analyzer`, có khả năng nhận diện và nhận dạng khuôn mặt trong một video đã cho trước. 

## II. Hướng dẫn sử dụng

### 1. Cài đặt các thư viện

Module này được code bằng Python 3.10; tuy vậy, module này không sử dụng những tính năng được giới thiệu sau Python 3.6 hoặc 3.7.

Module này sử dụng một số thư viện bên ngoài, bao gồm insightface, opencv, annoy và vidgear. Để cài đặt tất cả các thư viện mà module này sử dụng, ta có thể dùng lệnh

```bash
pip install -r requirements.txt
```

### 2. Chạy thử code

Để chạy thử file video.py trong module này, ta có thể sử dụng câu lệnh sau đây

```bash
python video.py <img_folder> <video_path> [<video_id>]
```

Ta cần phải truyền ít nhất 2 command-line arguments theo thứ tự như sau:

- img_folder: command-line argument đầu tiên sau video.py. Đây là thư mục chứa hình ảnh của các nhân vật mà ta cần nhận diện trong video.
- video_path: command-line argument thứ 2 sau video.py. Đây là path dẫn tới video mà chúng ta cần xử lý.
- video_id: command-line argument thứ 3 sau video.py, không bắt buộc. Đây là id của video; trong trường hợp chúng ta để trống, giá trị mặc định của video_id sẽ là giá trị của img_folder.

Notebook `notebook.ipynb` có thông tin chi tiết hơn về hai method có trong class Analyzer.

### 3. Lưu ý

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

## III. Cải tiến

So với đoạn code ban đầu, code trong file `video.py` có những cải tiến sau:

- Lọc bớt các model được sử dụng trong việc phân tích khuôn mặt trong `face_model`: thay vì sử dụng cả 5 model có sẵn, ta đã lọc bỏ những model không cần thiết cho việc nhận diện/nhận dạng khuôn mặt.
- Sử dụng lập trình hướng đối tượng: ta sử dụng class `Analyzer` để chứa hai method `analyze` và `embedding_extract`.
- Sử dụng thư viện `vidgear`: Bằng việc sử dụng thư viện này thay cho `opencv`, ta có thể làm cho việc đọc (và viết) video nhanh hơn.
