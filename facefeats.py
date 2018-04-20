import cv2
import os
import dlib
import math
import numpy as np
from scipy.signal import savgol_filter as savgol
from imutils import face_utils
from sklearn.preprocessing import normalize

# sequence length needed for padding
slen = 200

# path to dlib pretrained facial keypoints model
predictor = dlib.shape_predictor("detector/shape_predictor_68_face_landmarks.dat")

# path to opencv face detection model
face_cascade=cv2.CascadeClassifier("detector/haarcascade_frontalface_alt.xml")

# eye indices for dlib-predicted output
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


def detect(img, cascade = face_cascade, minimumFeatureSize=(20, 20)):
    if cascade.empty():
        raise (Exception("There was a problem loading your Haar Cascade xml file."))
    rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=1, minSize=minimumFeatureSize)

    # if it doesn't return rectangle return array
    # with zero lenght
    if len(rects) == 0:
        return []

    #  convert last coord from (width,height) to (maxX, maxY)
    rects[:, 2:] += rects[:, :2]
    return rects


def mindist(eyerect):
    eyerect=np.asfarray(eyerect)
    upair=eyerect[[1,2]]
    lpair=eyerect[[4,5]]
    return min([np.sqrt((u[0]-l[0])**2+(u[1]-l[1])**2) for u in upair for l in lpair])


def parseseq(seqpath, save=False):
    # given sequence root filestring, detect face rect, facial keypoints, eyelid distance, save to seq_dfeats.npy
    duration, fps, seqlen = os.path.basename(seqpath).split('-')[1:4]
    fps=np.float(fps)
    
    impaths=[f for f in os.listdir(seqpath) if f.endswith('.jpg')]
    impaths.sort()
    
    featlist=[]
   
    for im in impaths:
        feats=[]
        seq, fnum = im[:-4].split('-')
        seq=int(float(seq))
        fnum=int(float(fnum))
        feats.append(seq)
        feats.append(fnum)
        
        gray=cv2.imread(os.path.join(seqpath, im), 0)
        height, width = gray.shape[:2]
        te = detect(gray, minimumFeatureSize=(80, 80))
        if len(te) == 0:
#             feats+=[np.nan]*144 # pad to output seq length
#             featlist.append(feats)
            continue
        elif len(te) > 1:
            face = te[0]
        elif len(te) == 1:
            [face] = te

        # append face rectangle coords
#         feats=feats+face.tolist()
        feats.append(1.0*face[0]/gray.shape[1]) # left / image width
        feats.append(1.0*face[1]/gray.shape[0]) # top / image height
        feats.append(1.0*face[2]/gray.shape[1]) # right / image width
        feats.append(1.0*face[3]/gray.shape[0]) # bottom / image height
        
        # keep the face region from the whole frame x0, y0, x1, y1
        face_rect = dlib.rectangle(left = int(face[0]), top = int(face[1]),
                                    right = int(face[2]), bottom = int(face[3]))

        # determine the facial landmarks for the face region
#         clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
#         shape = predictor(clahe.apply(gray), face_rect)
        shape = predictor(gray, face_rect)
        shape = face_utils.shape_to_np(shape)

        #  grab the indexes of the facial landmarks for the left and
        #  right eye, respectively
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        # extract the left and right eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        # mean face intensity
        mfi = gray[int(face[1]):int(face[3]) , int(face[0]):int(face[2])].mean()
        
        # mean environment intensity (crop out face)
        upper_env = gray[:int(face[1]) , int(face[0]):int(face[2])]
        lower_env = gray[int(face[3]): , int(face[0]):int(face[2])]
        left_env = gray[: , :int(face[0])]
        right_env = gray[: , int(face[2]):]
        envi = 1.0*(upper_env.sum() + lower_env.sum() + left_env.sum() + right_env.sum())/(np.product(upper_env.shape) + np.product(lower_env.shape) + np.product(left_env.shape) + np.product(right_env.shape))

        #  face lighting & non-face lighting
        feats.append(mfi/255.0)
        feats.append(envi/255.0)
        
        # Eye Aspect Ratio
        lear=(np.linalg.norm(leftEye[1]-leftEye[5])+np.linalg.norm(leftEye[2]-leftEye[4]))/(2*np.linalg.norm(leftEye[0]-leftEye[3]))
        rear=(np.linalg.norm(rightEye[1]-rightEye[5])+np.linalg.norm(rightEye[2]-rightEye[4]))/(2*np.linalg.norm(rightEye[0]-rightEye[3]))
        feats.append(lear)
        feats.append(rear)

        # Head Pose
        roll, pitch, yaw = calcpose(gray.shape, shape)
        feats.append(roll)
        feats.append(pitch)
        feats.append(yaw)

        # Save all 68 landmark points (x&y = 2*68)
        feats=feats+[1.0*lmx/gray.shape[1] for lmx in shape[:,0].tolist()]
        feats=feats+[1.0*lmy/gray.shape[0] for lmy in shape[:,1].tolist()]
        
        # features list in order: seq[0], framenum[1], faceRect[2:6], mean_face_intensity[6], mean_env_intensity[7], leftEAR[8], rightEAR[9], roll[10], pitch[11], yaw[12], landmarks_x[13:81], landmarks_y[81:149]
        featlist.append(feats)
        
    featarr=np.array(featlist)
    print(featarr.shape)
    sortinds=np.lexsort((featarr[:,1], featarr[:,0]))
    
    savedir=os.path.join(os.path.dirname(os.path.dirname(seqpath)), 'framecsvs')
    os.makedirs(savedir, exist_ok=True)
    filename=os.path.join(savedir, os.path.basename(seqpath) + '.csv')
    with open(filename, 'w') as fout:
        fout.write('\n'.join([','.join([str(f) for f in fs]) for fs in featlist]))
    
    if save:
        agglist=[]
        for s in np.unique(featarr[:,0]):
            snum=str(int(s))
            seqarr=featarr[featarr[:,0]==s]
            seqsortinds=seqarr[:,1].argsort()
            savearr=seqarr[seqsortinds][:,2:]
            
            # normalize
#             savearr=normalize(savearr)
            
            if savearr.shape[0]<slen:
                savearr=np.vstack((savearr, np.zeros((slen-savearr.shape[0], savearr.shape[1]))))
            # for sequence number, save array as binary npy:
            # ..\data\sequences\framefeats\[videoname]-[time]-[fps]-[seqlen]-[start_end]\seqnum.npy
            savedir=os.path.join(os.path.dirname(os.path.dirname(seqpath)), 'framefeats', \
                                  os.path.basename(seqpath))

            os.makedirs(savedir, exist_ok=True)
            filename=os.path.join(savedir, snum + '.npy')
            np.save(filename, savearr)
            
            # aggregate features here seq_number[0], blinks[1], blink_avgframecount[2], face_stdev_x/face_width[3]
            # face_stdev_y/face_height[4], mean_face_pixel_val[5], mean_environment_pixel_val[6], 
            # head_roll_stdev[7], head_pitch_stdev[8], head_yaw_stdev[9] ==> 10 feats
            aggfeats=[]
            aggfeats.append(snum)
            blink, bavg = countblinks(seqarr, int(seqlen))
            aggfeats.append(str(blink))
            aggfeats.append(str(bavg))
            # stdev face_x position vs mean face width
            aggfeats.append(str(seqarr[:,2].std()/((seqarr[:,4]-seqarr[:,2]).mean())))
            # stdev face_y position vs mean face height
            aggfeats.append(str(seqarr[:,3].std()/((seqarr[:,5]-seqarr[:,3]).mean())))
            # mean of mean face pixel intensity
            aggfeats.append(str(seqarr[:,6].mean()))
            # mean of mean environment pixel intensity
            aggfeats.append(str(seqarr[:,7].mean()))
            # stdev roll, pitch, yaw
            aggfeats.append(str(seqarr[:,10].std()))
            aggfeats.append(str(seqarr[:,11].std()))
            aggfeats.append(str(seqarr[:,12].std()))
            
            agglist.append(aggfeats)
            
            
        # for image folder name, aggregate and save csv:
        # ..\data\sequences\aggfeats\[videoname]-[time]-[fps]-[seqlen]-[start_end].csv
        savedir=os.path.join(os.path.dirname(os.path.dirname(seqpath)), 'aggfeats')
        os.makedirs(savedir, exist_ok=True)
        csvname=os.path.join(savedir, os.path.basename(seqpath)+'.csv')
        with open(csvname, 'w') as f:
            f.write('\n'.join([','.join(row) for row in agglist]))
        
        return(featarr[sortinds], agglist)
        
    else:
        return(featarr[sortinds])


def countblinks(seqarr, framecount):
    farr=seqarr
    lear=farr[:,8]
    rear=farr[:,9]
    lsteady=savgol(lear, 31, 2)
    rsteady=savgol(rear, 31, 2)
    lthresh=lear.std()
    rthresh=rear.std()
    linds=np.where(lsteady-lear>=lthresh)
    rinds=np.where(rsteady-rear>=rthresh)
    carr=np.union1d(linds, rinds)
    blinks=0
    cont=[]
    frame=1
    for i in range(len(carr))[1:]: #count number of contiguous frames with eyes closed
        if carr[i]<=carr[i-1]+20: #allow 1 second gap
            frame+=1
            continue
        else:
            blinks+=1
            cont.append(frame)
            frame=1
    if len(cont)>0:
        c_avg=1.0*np.sum(cont)/len(cont)
    else:
        c_avg=0
    return blinks, c_avg

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])


def calcpose(size, shape):
    #2D image points. If you change the image, you need to change vector
    image_points = np.array([
                                shape[30],     # Nose tip
                                shape[8],     # Chin
                                shape[36],     # Left eye left corner
                                shape[45],     # Right eye right corne
                                shape[48],     # Left Mouth corner
                                shape[54]      # Right mouth corner
                            ], dtype="double")

    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner

                            ])

    # Camera internals
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    return rotationMatrixToEulerAngles(cv2.Rodrigues(rotation_vector)[0])