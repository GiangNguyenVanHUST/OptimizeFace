import insightface
from imutils import paths
import cv2
import os
from annoy import AnnoyIndex
import time
import uuid
from tqdm import tqdm
from vidgear.gears import CamGear, WriteGear
import time
import FaceAnalyzerConfig


class Analyzer:
    def __init__(self,
                 f=FaceAnalyzerConfig.ANNOY_DEFAULT_F,
                 threshold=FaceAnalyzerConfig.DET_SCORE_THRESHOLD,
                 min_similarity=FaceAnalyzerConfig.MIN_SIMILARITY):
        self.f = f
        self.annoy = AnnoyIndex(f, 'angular')
        self.face_model = insightface.app.FaceAnalysis()
        self.face_model.prepare(ctx_id=-1, det_size=(640, 640))
        self.face_model.models = {key: value for key, value in self.face_model.models.items(
        ) if key in ['detection', 'recognition']}
        self.id_mapping = {}
        self.name_mapping = ""
        self.font = FaceAnalyzerConfig.FONT
        self.DETECT_THRESHOLD = threshold
        self.MIN_SIMILARITY = min_similarity

    def embedding_extract(self, imgs_folder: str, id2name: str):
        """
        Create embeddings based on a set of images.

        Parameters
        ---
        imgs_folder: str
            Name of directory storing the images

        id2name: str
            Name of name mapping

        """
        self.name_mapping = id2name
        count = 0

        for img_path in tqdm(paths.list_images(imgs_folder)):
            try:
                dossier_id = img_path.split(os.path.sep)[-2]
                img = cv2.imread(img_path)
                img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
                faces = self.face_model.get(img)
                if len(faces) == 1:
                    for face in faces:
                        if face.det_score < self.DETECT_THRESHOLD:
                            continue
                        embedding = face.embedding.flatten()
                        self.annoy.add_item(count, embedding)
                        self.id_mapping[str(count)] = dossier_id
                        count += 1
            except Exception as e:
                continue

        self.annoy.build(200)
        print("[DEBUG] extract successful")

    def analyze(self,
                video_path: str,
                sub_dir: str,
                video_id='',
                outputs_video=False,
                output_folder=None):
        """
        Analyze a given video

        Parameters
        ---
        video_path: str
            Full path to video file

        sub_dir: str
            Folder to save dossier images in

        video_id: str, optional
            ID of video. Defaults to ''

        outputs_video: bool, str
            If True, outputs annotated video. Defaults to False

        output_folder: str
            Folder to save output video in. If value is None, output folder will be named 'output'. Defaults to None

        Return
        ---
        A list of tuples, whose ordered elements are as followed:
            - The first element: name of person who appeared in the video
            - The second element: path to dossier image
            - The third element: timestamp of the corresponding dossier image.
        """
        results = []

        cap = CamGear(source=video_path).start()
        frame_rate = cap.framerate

        if outputs_video:
            output_destination = output_folder if output_folder else 'output'
            if not os.path.exists(output_destination):
                os.mkdir(output_destination)

            output_name = video_id if len(video_id) else str(uuid.uuid4())
            output_movie = WriteGear(
                output_filename=f"./{output_destination}/{output_name}_output.mp4")

        start_time = time.time()
        captured = 0
        frame_id = 0
        on_screen = set()

        try:
            while 1:
                frame = cap.read()
                if frame is None:
                    print('end')
                    break

                frame_id += 1
                if frame_id % 5 == 0:
                    captured += 1
                    small_frame = cv2.resize(
                        frame, (0, 0), fx=0.25, fy=0.25)
                    faces = self.face_model.get(small_frame)

                    if len(faces) <= 0:
                        if outputs_video:
                            output_movie.write(frame)
                        continue

                    if len(faces) > 0:
                        time_duration = time.time() - start_time
                        print(
                            f'[{frame_id}] identified {len(faces)} faces at {time_duration}')

                    for face in faces:
                        if face.det_score < self.DETECT_THRESHOLD:
                            continue

                        left, top, right, bottom = tuple(
                            face.bbox.astype(int).flatten())
                        embedding = face.embedding.flatten()
                        closest_indices, distances = self.annoy.get_nns_by_vector(
                            embedding, n=1, include_distances=True)
                        similarity = (2. - distances[0] ** 2) / 2.

                        if similarity >= self.MIN_SIMILARITY:
                            crop_img = small_frame[top:bottom, left:right]
                            dossier_id = self.id_mapping[str(
                                closest_indices[0])]

                            if dossier_id not in on_screen:
                                filename = f'{dossier_id}_{str(uuid.uuid4())}.jpg'
                                path_to_save = os.path.join(
                                    sub_dir, filename)
                                writer = WriteGear(
                                    output_filename=path_to_save, compression_mode=False)
                                writer.write(crop_img)
                                timecode = frame_id / frame_rate
                                results.append(
                                    (dossier_id, path_to_save, timecode))

                            if outputs_video:
                                cv2.rectangle(frame, (left * 4, top * 4),
                                              (right * 4, bottom * 4), (0, 255, 0), 2)
                                cv2.putText(frame, dossier_id, (left * 4 + 6, bottom * 4 - 6),
                                            self.font, 0.5, (255, 255, 255), 1)

                    if outputs_video:
                        output_movie.write(frame)
        finally:
            cap.stop()
            if outputs_video:
                output_movie.close()
            analysis_length = time.time() - start_time
            print(f"[INFO] finished video analysis in {analysis_length}")

        return results
