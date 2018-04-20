import sys
import os
from splitvideo import *
from facefeats import *

vp, se=sys.argv[1:3]
t=5
pad=5
seqlen=200
fps=20

if se=='True':
    sebool=True
    sestring='start'
else:
    sebool=False
    sestring='end'

def main():
    splitvideo(vp, t, pad, seqlen, fps, start=sebool)    
    out=os.path.join(os.path.dirname(vp), 'sequences', 'images', os.path.basename(vp)[:-4]+'-'+'-'.join([str(t), str(fps), str(seqlen), sestring]))
    try:
        parseseq(out, save=True)
    except:
        print('Keypoint detection failed on: ' + out)

if __name__ == '__main__':
    main()