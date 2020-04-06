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
import json
import shutil
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def main(user_file_name, user_img_upload_dir, actor_file_name, actor_frame_path):
    actor_name = actor_file_name.split(".")[0]
    # 합성이 될 비디오 프레임 경로

    # 비디오 모든 프레임 이름을 가진 리스트
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

    # 유저, 배우 합성 과정에 생길 이미지들이 담길 디렉토리 생성
    resource_dir_path = "contents/resource/video/" + actor_name + "_" + user_image_name.split(".")[0]
    if not (os.path.isdir(resource_dir_path)):
        os.makedirs(resource_dir_path)

    # 유저, 배우 합성 결과가 담길 디렉토리 생성
    result_file_path = "contents/swap_result/video/" + actor_name + "_" + user_image_name.split(".")[0]
    if not (os.path.isdir(result_file_path)):
        os.makedirs(result_file_path)

    # 배우 영상에 대한 프로토콜 파일 불러오기
    with open(actor_frame_path+"/eye_facecount_protocol.json") as protocol_file:
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

    # 배우 동영상 총 프레임 수만큼 합성하자
    for actor_image_name in actor_image_name_list:
        user_imagecopy= user_image.copy()
        actor_face_protocol = actor_image_name_protocol_dict[actor_image_name]
        if actor_face_protocol == "zero" or actor_face_protocol == "overtwo":
            continue
        actor_image = actor_image_dict[actor_image_name]
        # 랜드마크의 정확도를 알기 위해
        # 조정된 랜드마크를 원본 얼굴에 찍어보기
        # actor_landmark = actor_adjust_lf_dict[actor_image_name]
        # for actor_lf in actor_landmark:
        #     cv2.line(actor_image, (actor_lf[0], actor_lf[1]), (actor_lf[0], actor_lf[1]), (255,0,0), 5)
        # cv2.imwrite(result_file_path+"/"+actor_image_name, actor_image)
        actor_landmark = actor_adjust_lf_dict[actor_image_name]
        actor_user_faceswap(actor_image_name, actor_image, actor_landmark, user_imagecopy, user_landmark,
                            resource_dir_path, result_file_path)
    # # 합성된 동영상을 만듬
    result_image_name_list = os.listdir(result_file_path)
    result_image_name_list = list(filter(lambda x: x.find('.jpg') != -1 or x.find('.png') != -1, result_image_name_list))
    result_image_name_list = sorted(result_image_name_list)

    # 원본 동영상에 대한 정보를 가져올 capture
    capture = cv2.VideoCapture("contents/video/original/" + actor_file_name)
    # 원본 동영상 프레임 속도의(fps) 2분 1로 합성한다
    fps = int(round(capture.get(cv2.CAP_PROP_FPS) / 2))
    capture.release()
    actor_image = actor_image_dict[actor_image_name_list[0]]
    # width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = actor_image.shape[1]
    height = actor_image.shape[0]
    # codec1 = cv2.VideoWriter_fourcc(*'XVID')

    video_result_path = result_file_path + '/' + actor_name + user_image_name.split(".")[0] + ".mp4"
    video = cv2.VideoWriter(video_result_path, cv2.VideoWriter_fourcc('m','p','4','v'), fps, (int(width), int(height)))

    for filename in result_image_name_list:
        result_image = cv2.imread(result_file_path + "/" + filename)
        video.write(result_image)
    video.release()

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
    user_img = faceswap(user_image, user_landmark, actor_image, actor_landmark, resource_dir_path)
    cv2.imwrite(resource_dir_path + "/3d_faceswap.jpg", user_img)

    # 배우 repaint_img + 액터 입술색 합성
    # repaint_img이미지에 입술색을 합성하지 않으면 입을 합성할때 입술색이 날라가서 피부색으로 나옴
    actor_repaint_img = lip_synthesis(actor_repaint_img, actor_image, actor_landmark, resource_dir_path)
    cv2.imwrite(resource_dir_path + "/lip_synthesis.jpg", actor_repaint_img)
    # 눈/코/입 부분 합성
    partial_synthesis_img = partial_synthesis(actor_image, user_img, user_landmark, actor_repaint_img)
    cv2.imwrite(resource_dir_path + "/partial_synthesis.jpg", partial_synthesis_img)
    # 입 안 합성
    res = mouth_synthesis(partial_synthesis_img, actor_image, actor_landmark, resource_dir_path)

    cv2.imwrite(result_file_path+"/"+actor_image_name, res)
    os.chmod(result_file_path +"/" + actor_image_name, 0o777)  # 권한 변경

    # cv2.imwrite(resource_path + "/" + actor_file_name.split(".")[0] + user_file_name, res)

def actorlandmark_adjust(actor_path, actor_image_name_list, actor_image_name_protocol_dict):
    # 1. 배우의 모든 프레임과 그에 해당하는 랜드마크를 리스트에 담는다.
    # 2. 랜드마크를 조정한다.
    # 3. 배우 이미지 한 개, 랜드마크 한 개씩 꺼내서 합성한다.
    actor_image_dict = dict()
    actor_adjust_lf_dict = dict()
    actor_adjust_lf_list = []
    framenum = 0
    # print("actor_path", actor_path)
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
    # print("len(actor_image_name_list)", len(actor_image_name_list))
    # print("len(actor_adjustedlflist)", len(actor_adjustedlflist))
    for actor_image_name in actor_image_name_list:
        # print("actorimg_name", actorimg_name)
        actor_adjust_lf_dict[actor_image_name] = actor_adjustedlflist[indexnum]
        indexnum += 1
    return actor_image_dict, actor_adjust_lf_dict


if __name__ == "__main__":
    user_file_name = "cropuser (15).jpg"
    user_img_upload_dir = "contents/user_images"
    actor_file_name = "iupalette_2.mp4"
    actor_frame_path = "contents/video/frames/" + actor_file_name

    main(user_file_name, user_img_upload_dir, actor_file_name, actor_frame_path)
    print("video 합성하는데 걸린 시간 : ", time.time()-starttime)
