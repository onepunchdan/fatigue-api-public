# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
import sys
import os
import skvideo
from splitvideo import *
from facefeats import *

os.environ["KERAS_BACKEND"] = "tensorflow"
from keras.models import Sequential, load_model
model_path='model/lstm-fatigue_class.027-0.601.hdf5'

import numpy as np
import flask
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER']='data'
model = None

def load_keras_model():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global model
	model=load_model(model_path)

t=0.167 # length of video in minutes, use 0.167 to get ~10.0 second clip from video file
pad=5 # padding in minutes, shorten this to 0.0167 to get ~1.0 second omission from end of video clip
seqlen=200 # number of frames to use in lstm sequence (must match trained model seq length)
fps=20 # frames per second, will upsample or downsample video to match this fps

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("video"):
			# read the image in PIL format
			file = flask.request.files["video"]
			filename = file.filename
			print(filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename+'.mp4'))
			vp = os.path.join(app.config['UPLOAD_FOLDER'], filename+'.mp4')

			splitvideo(vp, t, pad, seqlen, fps, start=False)
			# os.path.dirname(vp), 
			out=os.path.join('sequences', 'images', os.path.basename(vp)[:-4]+'-'+'-'.join([str(t), str(fps), str(seqlen), 'end']))
			try:
				featarr, agglistlist = parseseq(out, save=True)
			except:
				print('Keypoint detection failed on: ' + out)
				return(None)

			npy=os.path.join('sequences', 'framefeats', os.path.basename(vp)[:-4]+'-'+'-'.join([str(t), str(fps), str(seqlen), 'end']), '0.npy')
			X = np.empty((1, 200, 11))
			X[0, :, :] = np.load(npy)[:,:11]
			preds=model.predict(X)
			# print('Fatigue class: ', prediction)

			agglist = agglistlist[0]
			num_blinks = float(agglist[1])
			avg_blink_frames = float(agglist[2])

			# print('Detected: ', num_blinks, 'blinks')
			# print('Avg blink length: %.3f seconds per blink' %(avg_blink_frames/20.0))
			# print('Projceted blink rate: %d blinks per minute' %(num_blinks*6.0))

			data["predictions"] = []

			# loop over the results and add them to the list of
			# returned predictions
			for (label, prob) in enumerate(preds[0]):
				print(prob)
				print(type(prob))
				r = {"fatigue_class": label, "probability": round(float(prob), 4)}
				data["predictions"].append(r)

			data['num_blinks_in_seq'] = num_blinks
			data['avg_blink_length'] = round(avg_blink_frames/(1.0*fps), 3)
			data['proj_blink_rate'] = round(num_blinks*(60/(seqlen/fps)), 4)
			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_keras_model()
	# print(skvideo._FFMPEG_SUPPORTED_ENCODERS)
	app.run(host= '0.0.0.0')
