# OptimizeFace

## Giới thiệu

File video.py là một file Python có khả năng nhận vào một tập hợp các khuôn mặt chúng ta cần nhận diện, cùng với một video mà ta cần nhận diện. File python này sẽ xuất ra một video với thông tin nhận dạng khuôn mặt có tên `output.mp4`, kèm với một file `dossier.log` và một thư mục `dossier`. File `dossỉer.log` này sẽ ghi lại thời điểm những nhân vật chúng ta cần xem xét xuất hiện và rời khỏi màn hình, và thư mục `dossier` sẽ chứa hình ảnh của những nhân vật khi chúng ta nhận diện họ.

## Hướng dẫn cài đặt

Trước hết, ta cần phải cài đặt Python 3 (khuyên dùng 3.6 và 3.7).

Sau đó, cài đặt tất cả các package trong file `requirements.txt` bằng câu lệnh `pip install -r requirements.txt`

## Hướng dẫn chạy thử code

Để chạy thử hàm main của file video.py, sử dụng câu lệnh sau đây

```bash
python video.py <img_folder> <video_path> [<video_id>]
```

Ta cần phải truyền ít nhất 2 command-line arguments theo thứ tự như sau:

- img_folder: command-line argument đầu tiên sau video.py. Đây là thư mục chứa hình ảnh của các nhân vật mà ta cần nhận diện trong video.
- video_path: command-line argument thứ 2 sau video.py. Đây là path dẫn tới video mà chúng ta cần xử lý.
- video_id: command-line argument thứ 3 sau video.py, không bắt buộc. Đây là id của video; trong trường hợp chúng ta để trống, giá trị mặc định của video_id sẽ là giá trị của img_folder.

Notebook `notebook.ipynb` có thông tin chi tiết hơn về hai method có trong class Analyzer.

## Cải tiến

So với đoạn code ban đầu, code trong file `video.py` có những cải tiến sau:

- Lọc bớt các model được sử dụng trong việc phân tích khuôn mặt trong `face_model`: thay vì sử dụng cả 5 model có sẵn, ta đã lọc bỏ những model không cần thiết cho việc nhận diện/nhận dạng khuôn mặt.
- Sử dụng lập trình hướng đối tượng: thay vì 2 function riêng biệt, ta sử dụng class `Analyzer` để chứa hai method thay thế hai function trong đoạn code đã cho trước.
- Sử dụng thư viện `vidgear`: Bằng việc sử dụng thư viện này thay cho `opencv`, ta có thể làm cho việc đọc (và viết) video nhanh hơn.
