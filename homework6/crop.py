from PIL import Image
import sys


def crop(input_img_name : str, output_img_name : str):
	img = Image.open(input_img_name)
	new_img = img.resize((640, 480))
	new_img.save(output_img_name)
	
if __name__ == '__main__':
	crop(sys.argv[1],sys.argv[2])