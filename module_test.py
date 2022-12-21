from FaceAnalyzer import Analyzer
from sys import argv

if __name__ == '__main__':
    img_folder = argv[1]
    video_path = argv[2]
    video_id = argv[3] if len(argv) > 3 else img_folder

    analyzer = Analyzer()
    analyzer.embedding_extract(img_folder, img_folder)
    analyzer.analyze(video_path, 'dossier', video_id, outputs_video=True)
