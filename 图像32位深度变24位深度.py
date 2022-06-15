import PIL.Image
import os

path = r'C:/Users/eseen_new/Desktop/20220611all_dirty_water'
images = [os.path.join(path,item) for item in os.listdir(path)]
for item in images:
    try:
        rgba_image = PIL.Image.open(item)
        rgb_image = rgba_image.convert('RGB')
        rgb_image.save(item)
    except Exception as ex:
        print(item)
        continue
