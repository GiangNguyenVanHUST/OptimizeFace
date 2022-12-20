import os

YALE_DIR = "./data/yale"
files = os.listdir(YALE_DIR)[1:]
for i, img in enumerate(files):
    # print("original name: ", img)
    new_ext_name = "_".join(img.split(".")) + ".gif"
    # print("new name: ",  new_ext_name)
    os.rename(os.path.join(YALE_DIR, img),
              os.path.join(YALE_DIR, new_ext_name))
