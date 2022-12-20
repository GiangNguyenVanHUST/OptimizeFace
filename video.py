import insightface
from insightface.app import FaceAnalysis
from imutils import paths
import cv2
import os
from annoy import AnnoyIndex
import time
import uuid
from tqdm import tqdm
from vidgear.gears import CamGear, WriteGear
import time


class Analyzer:
    f: int  # number of dimensions
    annoy: AnnoyIndex
    face_mode: FaceAnalysis
    id_mapping: dict[str, str]
    name_mapping: str
    DETECT_THRESHOLD: float

    def __init__(self, f=512):
        self.f = 512
        self.annoy = AnnoyIndex(f, 'angular')
        self.face_model = insightface.app.FaceAnalysis()
        self.face_model.prepare(ctx_id=-1, det_size=(640, 640))
        self.id_mapping = {}
        self.name_mapping = ""
        self.DETECT_THRESHOLD = 0.6

    def embedding_extract(self, imgs_folder, id2name):
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

    def analyze(self, video_path, sub_dir, video_id=''):
        """ Analyze video

        Args:
            video_path  : video full path
            sub_dir     : where to save dossier images found
            video_id    : ID of video

        Returns:
            results     : List<(dossier_id/name, image_path)>
        """
        results = []

        status = 'PROCESSING'

        cap = CamGear(source=video_path).start()
        frame_rate = cap.framerate

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        print(cap.frame.shape)
        output_movie = cv2.VideoWriter(
            'output.mp4', fourcc, frame_rate, (cap.frame.shape[1], cap.frame.shape[0]))

        start_time = time.time()

        captured = 0
        frame_id = 0
        on_screen = set()

        with open('dossier.log', 'w') as f:
            try:
                while 1:
                    frame = cap.read()

                    if frame is None:
                        print('end')
                        break

                    frame_id += 1

                    if frame_id % 5 == 0:
                        captured += 1
                        width, height, _ = frame.shape
                        # evaluate bottleneck
                        small_frame = cv2.resize(
                            frame, (0, 0), fx=0.25, fy=0.25)
                        faces = self.face_model.get(small_frame)

                        if len(faces) <= 0:
                            output_movie.write(frame)
                            continue

                        if len(faces) > 0:
                            timestamp = time.time() - start_time
                            print(
                                f'[{frame_id}] identified {len(faces)} faces at {timestamp}')

                        seen_faces = set()

                        for face in faces:
                            if face.det_score < 0.6:
                                continue

                            # NOTE - bounding box for the face
                            # NOTE - maybe remove anything that's not needed?
                            left, top, right, bottom = tuple(
                                face.bbox.astype(int).flatten())
                            if left < 0:
                                left = 1
                            if top < 0:
                                top = 1
                            if right > width:
                                right = width - 1
                            if bottom > height:
                                bottom = height - 1

                            embedding = face.embedding.flatten()
                            # ANCHOR - bottleneck?
                            closest_indices, distances = self.annoy.get_nns_by_vector(
                                embedding, n=1, include_distances=True)
                            similarity = (2. - distances[0] ** 2) / 2.

                            if similarity >= 0.2:
                                crop_img = small_frame[top:bottom, left:right]
                                dossier_id = self.id_mapping[str(
                                    closest_indices[0])]
                                seen_faces.add(dossier_id)

                                if dossier_id not in on_screen:
                                    print(
                                        f'{dossier_id} on screen since {frame_id / frame_rate}', file=f)
                                    on_screen.add(dossier_id)
                                    filename = dossier_id + '_' + \
                                        str(uuid.uuid4()) + '.jpg'
                                    path_to_save = os.path.join(
                                        sub_dir, filename)
                                    writer = WriteGear(
                                        output_filename=path_to_save, compression_mode=False)
                                    writer.write(crop_img)

                                cv2.rectangle(frame, (left * 4, top * 4),
                                              (right * 4, bottom * 4), (0, 255, 0), 2)
                                cv2.rectangle(frame, (left * 4, bottom * 4 + 10),
                                              (right * 4, bottom * 4), (0, 255, 0), cv2.FILLED)
                                font = cv2.FONT_HERSHEY_DUPLEX
                                cv2.putText(frame, dossier_id, (left * 4 + 6, bottom * 4 - 6),
                                            font, 0.5, (255, 255, 255), 1)

                        for person in on_screen:
                            if person not in seen_faces:
                                print(
                                    f'{person} left screen at {frame_id / frame_rate}', file=f)
                                # del tagged_trackers[person]

                        on_screen = seen_faces
                        output_movie.write(frame)
            finally:
                cap.stop()
                analysis_length = time.time() - start_time
                print(f"[INFO] finished video analysis in {analysis_length}")

        return results, status


if __name__ == '__main__':
    analyzer = Analyzer()
    analyzer.embedding_extract('hamilton', 'hamilton')
    analyzer.analyze('short_hamilton_clip.mp4', 'dossier', 'hamilton')
