# USAGE
# python simple_request.py

# import the necessary packages
import requests
import pprint

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://0.0.0.0:5000/predict"
IMAGE_PATH = "/home/dan/Videos/test_10m.mp4"

# load the input image and construct the payload for the request
video = open(IMAGE_PATH, "rb").read()
payload = {"video": video}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# ensure the request was sucessful
if r["success"]:
	# loop over the predictions and display them
	#for (i, result) in enumerate(r["predictions"]):
		#print("fatigue {} prob: {:.4f}".format(result["fatigue_class"],
			#result["probability"]))
	#print("blinks detected: {:.0f}".format(r['num_blinks_in_seq']))
	#print("average blink duration: {:.1f} seconds".format(r['avg_blink_length']))
	#print("projected rate: {:.1f} blinks per minute".format(r['proj_blink_rate']))
	pprint.pprint(r)
# otherwise, the request failed
else:
	print("Request failed")
