# OptimizeFace

## Giới thiệu

Đoạn code được trình bày trong Jupyter Notebook dưới đây gồm hai function:

- `embedding_extract`: Tạo ra các embedding từ các ảnh đã được cho trước
- `analyze`: Dựa vào các embedding được tạo ở function `embedding_extract`, analyze những khuôn mặt xuất hiện trong một video có sẵn

Hai function này tương ứng với hai method cùng tên của class Analyzer, và code được trình bày trong Jupyter Notebook này gần tương tự với code trong class Analyzer.

## Import

Đầu tiên, chúng ta import một số module cần thiết cho việc vận hành của notebook này.

## Biến global

Sau đó, chúng ta thiết lập một số biến global cần thiết như sau.

Ở đây, sau khi thiết lập `face_model`, ta dùng một dictionary comprehension và giữ lại models detection và recognition. Việc chỉ giữ lại hai model này sẽ giúp quá trình phân tích khuân mặt trong hàm `analyze` nhanh hơn rất nhiều, vì `face_model` giờ đây chỉ cần tập trung vào việc nhận diện và nhận dạng khuôn mặt đã cho trước, thay vì cần phải phân tích cảnh hay phân tích độ tuổi / giới tính.

Trong đoạn code được chứa trong `video.py`, những biến global này tương ứng với những attribute của class Analyzer.

## Hàm `extract_embedding`

Hàm này có hai tham số truyền vào như sau:

- `imgs_folder`: Thư mục chứa ảnh của những nhân vật chúng ta cần nhận dạng
- `id2name`: Giá trị mới của `name_mapping`

Hàm này sẽ thêm những cặp key-value vào `id_mapping`. Các cặp key-value sẽ có cấu trúc như sau:

- `key`: 0, 1, 2, ...
- `value`: Tên của những nhân vật được định dạng

Để gắn tên `X` với một ảnh `p`, ta đặt `p` vào thư mục con của `imgs_folder` có tên `X`. Trong ví dụ mà ta chạy trong notebook này, ảnh của nhân vật có tên `lin-manuel` được đặt trong thư mục `hamilton/lin-manuel`, và ảnh của nhân vật có tên `alex` được đặt trong thư mục `hamilton/alex`.

## Hàm analyze

Hàm này sẽ nhận vào 4 tham số như sau:

- `video_path`: đường dẫn tới video ta cần phân tích
- `sub_dir`: thư mục chứa những ảnh dossier được tìm thấy trong quá trình phân tích video
- `video_id`: id của video chúng ta cần phân tích, giá trị mặc định là `""`
- `outputs_video`: một boolean với giá trị mặc định là `False`. Nếu như giá trị của boolean này là `True`, ta sẽ xuất ra một video tên là output.mp4 hoặc trong thư mục outputs, với khuôn mặt của các nhân vật được bao trong một hình chữ nhật viền xanh lá cây.

Hàm này sẽ trả về một list các tuple. Mỗi tuple trong hàm này có 3 phần như sau:

- Tên của nhân vật được nhận dạng
- Đường dẫn tới ảnh dossier
- Thời gian nhân vật xuất hiện lần đầu trên màn hình

Bên cạnh đó, hàm này cũng sẽ xuất ra một file `dossier.log` ghi lại thời điểm mỗi nhân vật xuất hiện và rời khỏi màn hình.
