import skvideo
import skvideo.io
import cv2
import os

# path to ffmpeg and ffprobe
# skvideo.setFFmpegPath(r'C:\Users\Bill\Anaconda2\envs\deeplearning\Library\bin')

def splitvideo(vp, t, pad, seqlen, fps, start=True):
    # vp = path to video
    # t = number of minutes to convert
    # pad = padding in minutes
    # fps = frames per second
    # start = parse from beginning of video (instead of end)
    # seqlen must be integer multiple of t*fps*60
#     if t*60*int(fps)%int(seqlen)!=0:
#         print('Invalid sequence length, must be a multiple of total num of output frames.')
#         return None
    
    meta=skvideo.io.ffprobe(vp)['video']
    fn,fd=meta['@r_frame_rate'].split('/')
    fr=int(float(fn)/float(fd)) #framerate
    vw=int(float(meta['@width']))
    vh=int(float(meta['@height']))
    vlen=float(meta['@duration'])
    flen=int(vlen*fr)

    # ss = starting time (seconds)
    # r = output framerate
    # vframes = number of frames to save (calculated)
    # from start: ss=padding*60
    # from end: ss=vlen-(padding+t)*60
    
    if start:
        ss=pad*60
    else:
        ss=int(vlen)-(pad+t)*60
    
    videogen=skvideo.io.FFmpegReader(vp, inputdict={'-ss': str(ss)}, outputdict={'-r': str(fps), '-vsync': '0', '-vframes': str(int(fps*t*60))})
#     videogen=skvideo.io.FFmpegReader(vp, inputdict={'-ss': str(ss)}, outputdict={'-vf': str(fps), 'vsync': '0', '-t': str(int(t*60))})
    
    seqind=0
    frmind=0
    sestring = 'start' if start else 'end'
    for frame in videogen.nextFrame():
        # skip first frame, for some reason
#         if frmind==0:
#             frmind+=1
#             continue
        # for every frame, save image by sequence:
        # ..\data\sequences\images\[videoname]-[time]-[fps]-[seqlen]-[start_end]\seqnum-framenum.jpg
        savedir=os.path.join('sequences', 'images', \
                              os.path.basename(vp).split('.')[0] + '-' + str(t) + '-' + str(fps) + \
                              '-' + str(seqlen) + '-' + sestring)
        
        os.makedirs(savedir, exist_ok=True)
        filename=str(seqind) + '-' + str(frmind) + '.jpg'
        cv2.imwrite(os.path.join(savedir, filename), cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
        
        if frmind==seqlen-1:
            seqind+=1
            frmind=0
        else:
            frmind+=1
    
    print('images written to: ' + savedir)
    return None


