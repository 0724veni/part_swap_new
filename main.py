import cv2
import dlib

from lip_synthesis import lip_synthesis
from partial_synthesis import partial_synthesis
from actor_mouth_synthesis import mouth_synthesis
from skin_color_change import skin_color_change
from faceswap_3d import faceswap
from imutils import face_utils
import os
from face_repaint import face_repaint



def partswap(user_img, actor_img, resource_path):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    user_face = detector(user_img)[0]

    # predictor(detector object)
    # user_landmark type : <class 'dlib.full_object_detection'>
    user_landmark = predictor(user_img, user_face)
    # print("predictor user_landmark type : {}".format(type(user_landmark)))

    # user_landmark type : <class 'numpy.ndarray'>, len : 68, shape : (68, 2), user_landmark[0] : [194 297]
    user_landmark = face_utils.shape_to_np(user_landmark)
    # print("shape_to_np user_landmark type : {}, len : {}, shape : {}, user_landmark[0] : {}".format(type(user_landmark), len(user_landmark), user_landmark.shape, user_landmark[0]))

    # 위와 유사함
    actor_face = detector(actor_img)[0]
    actor_landmark = predictor(actor_img, actor_face)
    actor_landmark = face_utils.shape_to_np(actor_landmark)

    # 배우 얼굴을 지운다.
    actor_repaint_img = face_repaint(actor_img, actor_landmark, resource_path)

    # 유저 이미지 배우 피부색에 맞추기
    user_img = skin_color_change(user_img, user_landmark, actor_img, actor_landmark, resource_path)

    # 유저 랜드마크 다시 구하기
    user_face = detector(user_img)[0]
    user_landmark = predictor(user_img, user_face)
    user_landmark = face_utils.shape_to_np(user_landmark)

    # 배우와 전체합성 한 유저 이미지
    # 해당 이미지를 배우 이미지에 부분합성 한다
    user_img = faceswap(user_img, user_landmark, actor_img, actor_landmark, resource_path)
    cv2.imwrite(resource_path + "/3d_faceswap.jpg", user_img)

    # 배우 repaint_img + 액터 입술색 합성
    # repaint_img이미지에 입술색을 합성하지 않으면 입을 합성할때 입술색이 날라가서 피부색으로 나옴
    actor_repaint_img = lip_synthesis(actor_repaint_img, actor_img, actor_landmark ,resource_path)

    # 눈/코/입 부분 합성
    partial_synthesis_img = partial_synthesis(actor_img, user_img, user_landmark, actor_repaint_img)
    cv2.imwrite(resource_path + "/partial_synthesis.jpg", partial_synthesis_img)
    # 입 안 합성
    res = mouth_synthesis(partial_synthesis_img, actor_img, actor_landmark, resource_path)

    cv2.imwrite("images/result/"+actor_img_path.split(".")[0]+user_img_path, res)
    cv2.imwrite(resource_path+"/" + actor_img_path.split(".")[0] + user_img_path, res)


if __name__ == "__main__":

    global user_img_path, actor_img_path

    user_img_path = "user (28).jpg"
    actor_img_path = "actor (12).jpg"

    # 유저, 배우 합성 과정에 생길 이미지들이 담길 디렉토리 생성
    resource_path = "images/resource/" + actor_img_path.split(".")[0] + "_" + user_img_path.split(".")[0]
    if not (os.path.isdir(resource_path)):
        os.makedirs(os.path.join(resource_path))

    user_img = cv2.imread("images/user/"+user_img_path)
    actor_img = cv2.imread("images/actor/"+actor_img_path)

    partswap(user_img, actor_img, resource_path)

