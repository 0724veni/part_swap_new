import time
starttime = time.time()
from utils import threeaverave
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
import imageio
import json
import shutil
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def main(user_file_name, user_img_upload_dir, actor_file_name, actor_frame_path):
    actor_name = actor_file_name.split(".")[0]

    # 합성이 될 영상의 경로
    actor_image_name_list = os.listdir(actor_frame_path)
    actor_image_name_list = list(filter(lambda x: x.find('.jpg') != -1 or x.find('.png') != -1, actor_image_name_list))
    actor_image_name_list = sorted(actor_image_name_list)

    # 유저의 이미지를 불러올 경로
    user_image_name = user_file_name
    user_image_path = user_img_upload_dir+"/"+user_file_name
    # 합성에 사용될 유저 이미지
    user_image = cv2.imread(user_image_path)
    # 유저 랜드마크
    user_face = detector(user_image)[0]
    user_landmark = predictor(user_image, user_face)
    user_landmark = face_utils.shape_to_np(user_landmark)

    # 배우 동영상 총 프레임 수만큼 합성하자
    # 유저, 배우 합성 과정에 생길 이미지들이 담길 디렉토리 생성
    resource_dir_path = "contents/resource/gif/" + actor_name + "_" + user_image_name.split(".")[0]
    if not (os.path.isdir(resource_dir_path)):
        os.makedirs(resource_dir_path)

    # 유저, 배우 합성 결과가 담길 디렉토리 생성
    result_file_path = "contents/swap_result/gif/" + actor_name + "_" + user_image_name.split(".")[0]
    if not (os.path.isdir(result_file_path)):
        os.makedirs(result_file_path)

    # 배우 영상에 대한 프로토콜 파일 불러오기
    with open(actor_frame_path + "/eye_facecount_protocol.json") as protocol_file:
        eyeface_protocol_dict = json.load(protocol_file)

    # 배우 프로토콜 파일 불러서 인식된 얼굴 개수가 0인 배우 이미지는 결과 디렉토리에 복사하기
    # actor_image_name_protocol_dict에 얼굴 인식 안된 프레임은 "zero"라는 값을 가짐
    eye_protocol_dict = dict()
    actor_image_name_protocol_dict = dict()
    for actor_image_name in actor_image_name_list:
        actor_img_facecount = eyeface_protocol_dict[actor_image_name]["facecount"]

        if actor_img_facecount == "zero" or actor_img_facecount == "overtwo":
            shutil.copy(actor_frame_path + "/" + actor_image_name, result_file_path + "/" + actor_image_name)
            actor_image_name_protocol_dict[actor_image_name] = "zero"
            continue
        actor_image_name_protocol_dict[actor_image_name] = actor_image_name
        eye_protocol_dict[actor_image_name] = eyeface_protocol_dict[actor_image_name]["eye"]

    # 5번 배우 이미지와 조정된 배우 랜드마크를 각각 doc에 넣기. 키 : 배우 이미지 파일, 값 :배우 랜드마크, 이미지
    actor_image_dict, actor_adjust_lf_dict = actorlandmark_adjust(actor_frame_path, actor_image_name_list, actor_image_name_protocol_dict)

    for actor_image_name in actor_image_name_list:
        # print("actor_image_name",actor_image_name)
        user_imagecopy = user_image.copy()
        actor_face_protocol = actor_image_name_protocol_dict[actor_image_name]
        if actor_face_protocol == "zero" or actor_face_protocol == "overtwo":
            continue

        actor_image = actor_image_dict[actor_image_name]
        actor_landmark = actor_adjust_lf_dict[actor_image_name]
        result = actor_user_faceswap(actor_image_name, actor_image, actor_landmark, user_imagecopy, user_landmark,
                            resource_dir_path, result_file_path)
        if result is None:
            shutil.copy(actor_frame_path + "/" + actor_image_name, result_file_path + "/" + actor_image_name)

    images = []
    result_image_name_list = os.listdir(result_file_path)
    result_image_name_list = list(filter(lambda x: x.find('.jpg') != -1 or x.find('.png') != -1, result_image_name_list))
    result_image_name_list = sorted(result_image_name_list)
    for filename in result_image_name_list:
        images.append(imageio.imread(result_file_path+"/"+filename))
    imageio.mimsave(result_file_path+"/"+actor_name+user_image_name.split(".")[0]+".gif", images, duration=0.06)

def actor_user_faceswap(actor_image_name, actor_image, actor_landmark, user_image, user_landmark,
                        resource_dir_path,result_file_path):
    # 배우 얼굴을 지운다.
    actor_repaint_img = face_repaint(actor_image, actor_landmark, resource_dir_path)

    # 유저 이미지 배우 피부색에 맞추기
    user_image = skin_color_change(user_image, user_landmark, actor_image, actor_landmark, resource_dir_path)
    # 유저 랜드마크 다시 구하기
    user_face = detector(user_image)[0]
    user_landmark = predictor(user_image, user_face)
    user_landmark = face_utils.shape_to_np(user_landmark)

    # 배우와 전체합성 한 유저 이미지
    # 해당 이미지를 배우 이미지에 부분합성 한다
    try:
        user_img = faceswap(user_image, user_landmark, actor_image, actor_landmark, resource_dir_path)
    except:
        return None
    # cv2.imwrite(resource_path + "/3d_faceswap.jpg", user_img)

    # 배우 repaint_img + 액터 입술색 합성
    # repaint_img이미지에 입술색을 합성하지 않으면 입을 합성할때 입술색이 날라가서 피부색으로 나옴
    actor_repaint_img = lip_synthesis(actor_repaint_img, actor_image, actor_landmark, resource_dir_path)

    # 눈/코/입 부분 합성
    partial_synthesis_img = partial_synthesis(actor_image, user_img, user_landmark, actor_repaint_img)
    # cv2.imwrite(resource_path + "/partial_synthesis.jpg", partial_synthesis_img)
    # 입 안 합성
    res = mouth_synthesis(partial_synthesis_img, actor_image, actor_landmark, resource_dir_path)

    cv2.imwrite(result_file_path+"/"+actor_image_name, res)
    return "good"

def actorlandmark_adjust(actor_path, actor_image_name_list, actor_image_name_protocol_dict):
    # 1. 배우의 모든 프레임과 그에 해당하는 랜드마크를 리스트에 담는다.
    # 2. 랜드마크를 조정한다.
    # 3. 배우 이미지 한 개, 랜드마크 한 개씩 꺼내서 합성한다.
    actor_image_dict = dict()
    actor_adjust_lf_dict = dict()
    actor_adjust_lf_list = []
    framenum = 0
    # print("actor_path", actor_path)
    # print("actor_image_name_list",actor_image_name_list)
    # print("actor_image_name_protocol_dict", actor_image_name_protocol_dict)
    for actor_image_name in actor_image_name_list:
        # 배우 이미지 프로토콜 (얼굴 인식 o = 이미지 파일 이름, 얼굴 인식 x = "zero")
        actor_image_protocol = actor_image_name_protocol_dict[actor_image_name]
        # print("actor_image_name",actor_image_name)
        # 배우 이미지
        actor_image = cv2.imread(actor_path + "/" + actor_image_name)

        # 랜드마크를 리스트에 담음, 배우 이미지를 이미지 이름이 키인 값으로 담
        # 배우 이미지 얼굴 인식이 안된 경우
        if actor_image_protocol == "zero" or actor_image_protocol == "overtwo":
            actor_adjust_lf_list.append("zero")
        else:
            # 배우 이미지에서 얼굴 랜드마크 검출
            # print("actor_image_name",actor_image_name)
            actor_face = detector(actor_image)[0]
            actor_landmark = predictor(actor_image, actor_face)
            actor_landmark = face_utils.shape_to_np(actor_landmark)
            actor_adjust_lf_list.append(actor_landmark)
        # 배우 이미지 이름을 키로, 배우 이미지를 값으로 담음
        actor_image_dict[actor_image_name] = actor_image

    # 배우 동영상의 모든 프레임에 나온 얼굴 랜드마크 조정
    actor_adjustedlflist = threeaverave(actor_adjust_lf_list)

    # list에 있는 값을 dict로 옮긴다.
    indexnum = 0
    for actor_image_name in actor_image_name_list:
        actor_adjust_lf_dict[actor_image_name] = actor_adjustedlflist[indexnum]
        indexnum += 1
    return actor_image_dict, actor_adjust_lf_dict

if __name__ == "__main__":
    user_file_name = "user-115.jpg"
    user_img_upload_dir = "contents/user_images"
    actor_file_name = "actor-1-3.gif"
    actor_frame_path = "contents/gif/frames/" + actor_file_name

    main(user_file_name, user_img_upload_dir, actor_file_name, actor_frame_path)
    print("gif 합성하는데 걸린 시간 : ", time.time()- starttime)
