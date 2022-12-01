import cv2
from src.utils.pose import Pose2D, PoseConfig
from opt import opt

"""
Draw annotation over an image
"""
class Drawer:

    BONE_COLOR = (0,255,222)
    JOINT_COLOR =(5,5,5)

    PID_FOREGROUND = (0,255,222)
    PID_BACKGROUND = (35, 35, 35)
    PID_LETTERS = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

    TXT_FOREGROUND = (225,225,225)
    TXT_BACKGROUND = (25,25,25)

    #BONE_COLOR = (110, 249, 227)
    #JOINT_COLOR = (60, 199, 157)


    """ Return a new image with the 2D pose depicted for a given src.utils.pose.pose2D"""
    @staticmethod
    def draw_2d_pose(img, pose_2d, thickness=2):

        img = img.copy()
        if opt.totaljoints == 13 :
            bones = PoseConfig.BONES
        elif opt.totaljoints == 9 :
            bones = PoseConfig.REBONES
        elif opt.dataset == "MPII":
            bones = PoseConfig.MPIIBONES
        else:
            raise ValueError("Your dataset name is wrong")

        joints = pose_2d.get_joints()

        joints[:,0] = (joints[:,0] * img.shape[1])
        joints[:,1] = (joints[:,1] * img.shape[0])
        joints = joints.astype(int)

        is_active_mask = pose_2d.get_active_joints()

        for bone_id in range(len(bones)):

            joint_ids = bones[bone_id]

            joint_1 = joints[joint_ids[0]]
            joint_2 = joints[joint_ids[1]]

            if is_active_mask[joint_ids[0]] and is_active_mask[joint_ids[1]]:
                cv2.line(img, tuple(joint_1), tuple(joint_2), Drawer.BONE_COLOR, thickness)


        for i in range(0,joints.shape[0]):
            color = Drawer.BONE_COLOR if i == 0 else Drawer.JOINT_COLOR
            cv2.circle(img, (joints[i,0], joints[i,1]), 3, color, -1)

        return img


    # @staticmethod
    # def draw_text(img, position, txt, size=32,color=(0,0,0), fontpath = "ressources/fonts/Open_Sans/OpenSans-Bold.ttf"):
    #
    #     font = ImageFont.truetype(fontpath, size)
    #     img_pil = Image.fromarray(img.copy())
    #     draw = ImageDraw.Draw(img_pil)
    #     draw.text(position, txt, font=font, fill=(color[0], color[1], color[2],255))
    #     img = np.array(img_pil)
    #
    #     return img


    """ Return a new image with all 2D pose depicted for a given list of annotations"""
    @staticmethod
    def draw_scene(img, poses_2d, person_ids, fps=None, curr_frame=None):

        img = img.copy()

        img = cv2.rectangle(img, (0, 0), (img.shape[1], 20), Drawer.TXT_BACKGROUND, cv2.FILLED)
        #
        # if not isinstance(fps, type(None)):
        #     img = Drawer.draw_text(img, (img.shape[1]-178,0), "running at "+str(fps)+" fps",size=13,color=Drawer.TXT_FOREGROUND)
        #
        #
        # if not isinstance(curr_frame, type(None)):
        #     img = Drawer.draw_text(img, (40,0), "frame "+str(int(curr_frame)), color=Drawer.TXT_FOREGROUND, size=13)
        keypoints = []
        for pid in range(len(poses_2d)):

            # Draw the skeleton
            img = Drawer.draw_2d_pose(img, poses_2d[pid])

            # The person id is written on the gravity center

            tmp = poses_2d[pid].get_gravity_center()

            if opt.totaljoints == 13:
                tmp[0] = (tmp[0]+poses_2d[pid].get_joints()[PoseConfig.HEAD, 0])/2.0
                tmp[1] = (tmp[1]+poses_2d[pid].get_joints()[PoseConfig.HEAD, 1])/2.0
            elif opt.dataset == "MPII":
                tmp[0] = (tmp[0] + poses_2d[pid].get_joints()[PoseConfig.MPIIhead_top, 0]) / 2.0
                tmp[1] = (tmp[1] + poses_2d[pid].get_joints()[PoseConfig.MPIIhead_top, 1]) / 2.0
            elif opt.totaljoints == 9:
                tmp[0] = (tmp[0]+poses_2d[pid].get_joints()[PoseConfig.REHEAD, 0])/2.0
                tmp[1] = (tmp[1]+poses_2d[pid].get_joints()[PoseConfig.REHEAD, 1]) / 2.0
            else:
                raise ValueError("Your dataset name is wrong")

            x, y = int(tmp[0]*img.shape[1]), int(tmp[1]*img.shape[0])
            keypoints.append(x)
            keypoints.append(y)
            # img = cv2.rectangle(img, (x-13, y-23), (x + 17, y+7), Drawer.PID_FOREGROUND, cv2.FILLED)
            # img = cv2.rectangle(img, (x-7, y-17), (x + 23, y+13), Drawer.PID_BACKGROUND, cv2.FILLED)
            #img = Drawer.draw_text(img,  (x, y-20), Drawer.PID_LETTERS[person_ids[pid]],size=50, color=Drawer.PID_FOREGROUND, fontpath = "ressources/fonts/Open_Sans/OpenSans-Bold.ttf")

        return img,keypoints



