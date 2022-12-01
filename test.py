from datetime import time
import cv2
from src.dataprovider.test.interface import AnnotatorInterface
from src.utils.drawer import Drawer
import time
from opt import opt


def start_video(movie_path, max_persons):

    annotator = AnnotatorInterface.build(224,model=opt.testmodel,max_persons=max_persons)

    cap = cv2.VideoCapture(movie_path)

    while(True):

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        tmpTime = time.time()
        persons = annotator.update(frame)
        fps = int(1/(time.time()-tmpTime))

        poses = [p['pose_2d'] for p in persons]

        ids = [p['id'] for p in persons]
        frame,key_points = Drawer.draw_scene(frame, poses, ids, fps, cap.get(cv2.CAP_PROP_POS_FRAMES))

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.waitKey(33) == ord(' '):
            curr_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(curr_frame + 30))

    annotator.terminate()
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    print("start frontend")
    # #for webcame or video
    max_persons = 1
    default_media = 0
    cam_num = 0
    start_video(cam_num, max_persons)







