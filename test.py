from PIL import Image

img = Image.open('irene_test.jpeg')
width, height = img.size
img.thumbnail((width // 2, height // 2), Image.Resampling.LANCZOS)
img.save('mini_irene.jpeg')
