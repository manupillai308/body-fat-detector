from flask import Flask, request, jsonify, render_template
import cv2
import os


face_csc = cv2.CascadeClassifier('haarcascade_fullbody.xml')
def get_bf(face_csc, im_path, weight, h1):
	bf = 0
	img = cv2.imread(im_path)
	   
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	faces = face_csc.detectMultiScale(gray, 1.1, 4)
	
	for (x, y, w, h) in faces:
		cv2.rectangle(gray, (x+20,y+30), (x+w-20, y+h), (0,0,0), 5)
		height = y+30-y-h
		width = x + 20 - x - w + 20
		breadth = h1/height * width
		area = breadth * h1
		vol = area * breadth * 70/100
		bf = (495/(weight*10000/vol)) - 450 #body fat
	return bf

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
	if request.method == 'GET':
		return render_template('home.html')
	else:
		weight = request.form.get('weight')
		h1=	request.form.get('height')
		image = request.files.get('image')
		im_path = os.path.join('./image', image.filename)
		image.save(im_path)
		bf = get_bf(face_csc, im_path, int(weight), int(h1))
		return jsonify(bf)
	
if __name__ == "__main__":
	app.run(use_reloader = True, debug=True)
