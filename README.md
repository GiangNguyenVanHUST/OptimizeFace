# OptimizeFace

## Giới thiệu

File video.py là một file Python có khả năng nhận vào một tập hợp các khuôn mặt chúng ta cần nhận diện, cùng với một video mà ta cần nhận diện. File python này sẽ xuất ra một video với thông tin nhận dạng khuôn mặt có tên `output.mp4`, kèm với một file `dossier.log` và một thư mục `dossier`. File `dossỉe.log` này sẽ ghi lại thời điểm những nhân vật chúng ta cần xem xét xuất hiện và rời khỏi màn hình, và thư mục `dossier` sẽ chứa hình ảnh của những nhân vật khi chúng ta nhận diện họ.

## Hướng dẫn cài đặt

Trước hết, ta cần phải cài đặt Python 3.

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
