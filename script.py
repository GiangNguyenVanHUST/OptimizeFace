import insightface
from imutils import paths, resize
import cv2
import os
from annoy import AnnoyIndex
import time
import numpy as np
import uuid
from tqdm import tqdm

f = 512  # num of dimensions
annoy = AnnoyIndex(f, 'angular')

face_model = insightface.app.FaceAnalysis()
face_model.prepare(ctx_id=-1, det_size=(640, 640))
id_mapping = {}
name_mapping = ""

DETECT_THRESHOLD = 0.6


def embedding_extract(imgs_folder, id2name):
    global id_mapping, name_mapping

    id_mapping = {}
    name_mapping = id2name
    count = 0

    for img_path in tqdm(paths.list_images(imgs_folder)):
        try:
            dossier_id = img_path.split(os.path.sep)[-2]
            print(f"[DEBUG] dossier_id: {dossier_id}")
            img = cv2.imread(img_path)
            # TODO resize
            img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
            faces = face_model.get(img)
            if len(faces) == 1:
                for face in faces:
                    if face.det_score < DETECT_THRESHOLD:
                        continue
                    embedding = face.embedding.flatten()
                    print(f"[DEBUG] embedding size: {embedding.shape}")
                    annoy.add_item(count, embedding)
                    id_mapping[str(count)] = dossier_id
                    count += 1
        except Exception as e:
            continue

    annoy.build(4)
    print("[DEBUG] extract successful")


def analyze(filename, sub_dir, video_id=''):
    """ Analyze video

    Args:
        video_path  : video full path
        sub_dir     : where to save dossier images found
        video_id    : ID of video

    Returns:
        results     : List<(dossier_id/name, image_path)>
    """
    print(f'[DEBUG] id_mapping: {id_mapping}')
    results = []

    status = 'PROCESSING'
    # before_time = time.time()

    image = cv2.imread(filename)
    height, width, _ = image.shape
    start_time = time.time()
    completed_frames = 0

    try:
        completed_frames += 1
        small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        faces = face_model.get(small_frame)

        if len(faces) > 0:
            timestamp = time.time() - start_time
            print(f"[INFO] found {len(faces)} faces in {timestamp}")

        print([x.det_score for x in faces])
        for face in faces:
            print(f" >> det_score: {face.det_score}")
            if face.det_score < 0.6:
                continue
            print(f" >> found face with p: {face.det_score}")

            embedding = face.embedding.flatten()
            closest_indices, distances = annoy.get_nns_by_vector(
                embedding, n=1, include_distances=True)
            closest_embedding = annoy.get_item_vector(
                closest_indices[0])
            print(closest_indices)
            similarity = (2. - distances[0] ** 2) / 2.
            print(f" >> sim: {similarity}")

            left, top, right, bottom = tuple(
                face.bbox.astype(np.int).flatten())
            if left < 0:
                left = 1
            if top < 0:
                top = 1
            if right > width:
                right = width - 1
            if bottom > height:
                bottom = height - 1
            face_size = (right - left) * (bottom - top)
            print(left, top, right, bottom)

            if similarity >= 0.2:
                print(f" >> guess: {id_mapping[str(closest_indices[0])]}")
                crop_img = small_frame[top:bottom, left:right]
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)

                dossier_id = id_mapping[str(closest_indices[0])]
                filename = dossier_id + '_' + \
                    str(uuid.uuid4()) + '.jpg'
                path_to_save = os.path.join(
                    sub_dir, filename)

                cv2.rectangle(image, (left * 4, top * 4),
                              (right * 4, bottom * 4), (0, 255, 0), 2)
                cv2.rectangle(image, (left * 4, bottom * 4 + 10),
                              (right * 4, bottom * 4), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(image, dossier_id, (left * 4 + 6, bottom * 4 - 6),
                            font, 0.5, (255, 255, 255), 1)

                cv2.imwrite(path_to_save, crop_img)
                results.append(
                    (dossier_id, path_to_save))
    finally:
        print("[INFO] finished video analysing")

    cv2.imshow('lesserafim', image)
    cv2.waitKey(0)
    return results, status


if __name__ == '__main__':
    embedding_extract('lesserafim', 'lesserafim')
    analyze('input_2.jpg', 'dossier', 'lesserafim')
