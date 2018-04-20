
import sys
import os
from splitvideo import *
from facefeats import *

os.environ["KERAS_BACKEND"] = "tensorflow"
from keras.models import Sequential, load_model
model_path='model/lstm-fatigue_class.027-0.601.hdf5'

model=load_model(model_path)

vp=sys.argv[1]
# vp='data\week10_dan_1.mp4'
# t=0.167
# pad=0.0167
t=0.167
pad=5
seqlen=200
fps=20

def main():
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
    prediction=np.argmax(model.predict(X))
    print('Fatigue class: ', prediction)

    agglist = agglistlist[0]
    num_blinks = float(agglist[1])
    avg_blink_frames = float(agglist[2])

    print('Detected: ', num_blinks, 'blinks')
    print('Avg blink length: %.3f seconds per blink' %(avg_blink_frames/20.0))
    print('Projceted blink rate: %d blinks per minute' %(num_blinks*6.0))

    return(prediction)

if __name__ == '__main__':
    main()