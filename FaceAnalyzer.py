import cv2
import os
import time
import uuid
import json
import insightface
from imutils import paths
from annoy import AnnoyIndex
from tqdm import tqdm
from vidgear.gears import CamGear, WriteGear


class Analyzer:
    def __init__(self,
                 config=None, 
                 path_model_root = None,
                 font=cv2.FONT_HERSHEY_DUPLEX):

        if config != None:
            self.config = config
        else:
            return False

        self.annoy = AnnoyIndex(self.config['f'], 'angular')
        self.face_model = insightface.app.FaceAnalysis()
        self.face_model.prepare(ctx_id=-1, det_size=(640, 640))
        self.face_model.models = {key: value for key, value in self.face_model.models.items(
        ) if key in ['detection', 'recognition']}        
        
        self.path_model_root = path_model_root
        self.id_mapping = {}
        self.name_mapping = ""
        self.font = font


    def reload_model(self,
                    model_id = None):
        """
        Reload model which include annoy, id_mapping and name_mapping

        Parameters
        ---
        model_id: str
            Name of directory storing the model

        """
        if self.path_model_root == None or model_id == None:
            print("path_model_root or model_id is None")
            return False

        path_id_mapping = os.path.join(self.path_model_root, str(model_id) + '_id_mapping.json')
        path_name_mapping = os.path.join(self.path_model_root, str(model_id) + '_name_mapping.json')
        path_annoy = os.path.join(self.path_model_root, str(model_id) + ".ann")

        if not os.path.exists(path_id_mapping) or \
            not os.path.exists(path_name_mapping) or \
            not os.path.exists(path_annoy):
            print("id_mapping or name_mapping or path_annoy is not exists")
            return False
                        
        with open(path_id_mapping) as f:
            self.id_mapping = json.load(f)

        with open(path_name_mapping) as f:
            self.name_mapping = json.load(f)
        
        self.annoy = AnnoyIndex(self.config['f'], 'angular')        
        self.annoy.load(path_annoy)
        return True


    def __save_model(self,
                    model_id = None):
        """
        Save model which include annoy, id_mapping and name_mapping

        Parameters
        ---
        model_id: str
            Name of directory storing the model

        """
        if self.path_model_root == None or model_id == None:
            print("path_model_root or model_id is None")
            return False

        path_id_mapping = os.path.join(self.path_model_root, str(model_id) + '_id_mapping.json')
        path_name_mapping = os.path.join(self.path_model_root, str(model_id) + '_name_mapping.json')
        path_annoy = os.path.join(self.path_model_root, str(model_id) + ".ann")
                                
        with open(path_id_mapping, 'w') as f:
            json.dump(self.id_mapping)

        with open(path_name_mapping, 'w') as f:
            json.dump(self.name_mapping)
            
        self.annoy.save(path_annoy)        
        return True


    def __normalize(self,
                    left,
                    top,
                    right, 
                    bottom,
                    frame):
        
        height, width, _ = frame.shape
        left = 0 if left < 0 else left
        top = 0 if top < 0 else top                    
        right = width - 1 if right > width else width
        bottom = height - 1 if bottom > height else height
        
        return left, top, right, bottom
        
    
    def __crop_face(self,
                    left,
                    top,
                    right, 
                    bottom,
                    frame):
        
        face_hight = bottom - top
        face_width = right - left
        left = left - face_width * self.config["FACE_INFLATE_RATIO"]
        right = right  + face_width * self.config["FACE_INFLATE_RATIO"]
        top = top - face_hight * self.config["FACE_INFLATE_RATIO"]
        bottom = bottom + face_hight*self.config["FACE_INFLATE_RATIO"]
        
        left, top, right, bottom = self.__normalize(left, top, right, bottom, frame)

        crop_img = frame[int(top):int(bottom), int(left):int(right)]
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
        return crop_img        
        

    def embedding_extract(self, model_id: str, imgs_folder: str, id2name: str):
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
                        if face.det_score < self.config['DETECT_THRESHOLD']:
                            continue
                        embedding = face.embedding.flatten()
                        self.annoy.add_item(count, embedding)
                        self.id_mapping[str(count)] = dossier_id
                        count += 1
            except Exception as e:
                continue

        self.annoy.build(int(self.config['NUMBER_OF_TREES']))
        self.__save_model(model_id)
        print("[DEBUG] extract successful")


    def analyze_image(self,
                frame,                
                frame_id='',
                video_id='', 
                timecode=None,               
                outputs_image=False,
                output_folder=None):
        """
        Analyze a given image

        Parameters
        ---
        image_path: str
            Full path to image file

        sub_dir: str
            Folder to save dossier images in
        
        frame_id: str, optional
            ID of image. Defaults to ''
        
        video_id: str, optional
            ID of image. Defaults to ''

        outputs_image: bool, str
            If True, outputs annotated image. Defaults to False

        output_folder: str
            Folder to save output image in. If value is None, output folder will be named 'output'. Defaults to None

        Return
        ---
        A list of tuples, whose ordered elements are as followed:
            - The first element: name of person who appeared in the video
            - The second element: path to dossier image
            - The third element: timestamp of the corresponding dossier image.
        """
        
        results = []
        
        if outputs_image:
            output_destination = output_folder if output_folder else 'output'
            if not os.path.exists(output_destination):
                os.mkdir(output_destination)

            output_name = video_id if len(video_id) else str(uuid.uuid4())
            path_output_folder = os.path.join(output_destination, output_name)
            if not os.path.exists(path_output_folder):
                os.mkdir(path_output_folder)

        start_time = time.time()
        faces = self.face_model.get(frame)

        if len(faces) <= 0:            
            if len(faces) > 0:
                time_duration = time.time() - start_time
                print(
                    f'[{frame_id}] identified {len(faces)} faces at {time_duration}')

                for face in faces:
                    if face.det_score < self.config['DETECT_THRESHOLD']:
                        continue
                    
                    left, top, right, bottom = tuple(
                        face.bbox.astype(int).flatten())                        
                    left, top, right, bottom = self.__normalize(left, top, right, bottom, frame)
                    face_size = (right - left) * (bottom - top)

                    try:
                        if face_size < pow(self.config["SMALL_IMG_THRESH"], 2) * frame.shape[0] * frame.shape[1]:
                            continue
                    except Exception as e:
                        print(f"[DEBUG] {e}")
                        continue

                    crop_img = self.__crop_face(left, top, right, bottom, frame)

                    embedding = face.embedding.flatten()
                    closest_indices, distances = self.annoy.get_nns_by_vector(
                        embedding, n=int(self.config['N_CLOSEST']), include_distances=True)
                    similarity = (2. - distances[0] ** 2) / 2.

                    if similarity >= self.config['MIN_SIMILARITY']:                            
                        dossier_id = self.id_mapping[str(
                            closest_indices[0])]
                    else:
                        dossier_id = "unknown"

                    filename = f'{frame_id}_{dossier_id}_{str(uuid.uuid4())}.jpg'                                
                    path_to_save = os.path.join(
                                path_output_folder, filename)
                    writer = WriteGear(
                    output_filename=path_to_save, compression_mode=False)
                    writer.write(crop_img)
                    
                    results.append(
                        (dossier_id, path_to_save, timecode))
        
        return results
                    
    
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
        
        try:
            while 1:
                frame = cap.read()
                if frame is None:
                    print('end')
                    break

                frame_id += 1
                if frame_id % 5 == 0:
                    captured += 1
                    
                    faces = self.face_model.get(frame)

                    if len(faces) <= 0:
                        if outputs_video:
                            output_movie.write(frame)
                        continue

                    if len(faces) > 0:
                        time_duration = time.time() - start_time
                        print(
                            f'[{frame_id}] identified {len(faces)} faces at {time_duration}')

                    for face in faces:
                        if face.det_score < self.config['DETECT_THRESHOLD']:
                            continue
                        
                        left, top, right, bottom = tuple(
                            face.bbox.astype(int).flatten())                        
                        left, top, right, bottom = self.__normalize(left, top, right, bottom, frame)
                        face_size = (right - left) * (bottom - top)

                        try:
                            if face_size < pow(self.config["SMALL_IMG_THRESH"], 2) * frame.shape[0] * frame.shape[1]:
                                continue
                        except Exception as e:
                            print(f"[DEBUG] {e}")
                            continue

                        crop_img = self.__crop_face(left, top, right, bottom, frame)

                        embedding = face.embedding.flatten()
                        closest_indices, distances = self.annoy.get_nns_by_vector(
                            embedding, n=int(self.config['N_CLOSEST']), include_distances=True)
                        similarity = (2. - distances[0] ** 2) / 2.

                        if similarity >= self.config['MIN_SIMILARITY']:                            
                            dossier_id = self.id_mapping[str(
                                closest_indices[0])]
                        else:
                            dossier_id = "unknown"

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
