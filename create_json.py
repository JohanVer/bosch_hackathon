import json
from os import listdir
from os.path import isfile, join
from PIL import Image

label_tool_path = '/home/vertensj/js-segment-annotator/data/'
images_path = label_tool_path + 'originals/'
annotations_path = label_tool_path + 'annotations/'
images_save_path = label_tool_path + 'images/'
json_store = label_tool_path + 'example.json'

new_width = 500
new_height = 250

data = {}
data['labels'] = ['triangle', 'rectangle', 'circle']

images = [f for f in listdir(images_path) if isfile(join(images_path, f))]
#print images
paths = []
labels = []
for i in images:
	if 'png' in i:
		org_image = Image.open(images_path + i)
		org_image = org_image.resize((new_width, new_height), Image.ANTIALIAS)
		org_image.save(images_save_path + i, 'PNG', transparency=0)
		paths.append(images_save_path + i)

		img = Image.new('RGBA',(new_width, new_height))		
		img.save(annotations_path + i, 'PNG', transparency=0)
		labels.append(annotations_path + i)
		
data['imageURLs'] = paths
data['annotationURLs'] = labels

with open(json_store, 'w') as f:
     json.dump(data, f)


