import cv2
import numpy as np
import copy
from resize_point import *

def lip_synthesis(repaint_actor_img, actor_img, user_landmarks, resource_path):
    """
    :param repaint_actor_img: 이목구비 지운 액터 이미지
    :param actor_img: 액터 이미지
    :param user_landmarks: 유저 랜드마크
    :param resource_path: 입술 바뀐 이미지 저장 경로
    :return:
    """

    #  마스크 만들기
    #  한번에 np.array에 입술 랜드마크를 다 넣고 fillConvexPoly에 넣으면 다각형 안이 완전히 채워지지 않는 일이 발생하는 경우가 있음.
    #  에러를 찾아봤는데 polygon이 convex가 아니라는 답변글 하나만 발견 ㅜ
    #  우선 임시방편으로 입술을 6개의 다각형으로 쪼개서 마스크를 만들었음

    mouth_top_left = np.array([user_landmarks[48], user_landmarks[49], user_landmarks[50], user_landmarks[61], user_landmarks[60], user_landmarks[48]], np.int32)
    mouth_top_middle = np.array([user_landmarks[50], user_landmarks[51], user_landmarks[52], user_landmarks[63], user_landmarks[62], user_landmarks[61]], np.int32)
    mouth_top_right = np.array([user_landmarks[52], user_landmarks[53], user_landmarks[54], user_landmarks[64], user_landmarks[63], user_landmarks[52]], np.int32)
    mouth_bottom_left = np.array([user_landmarks[48], user_landmarks[60], user_landmarks[67], user_landmarks[58], user_landmarks[59], user_landmarks[48]], np.int32)
    mouth_bottom_middle = np.array([user_landmarks[67], user_landmarks[66], user_landmarks[65], user_landmarks[56], user_landmarks[57], user_landmarks[58]], np.int32)
    mouth_bottom_right = np.array([user_landmarks[65], user_landmarks[64], user_landmarks[54], user_landmarks[55], user_landmarks[56], user_landmarks[65]], np.int32)

    mouth_top_left = np.array(mouth_top_left, np.int32)
    mouth_top_middle = np.array(mouth_top_middle, np.int32)
    mouth_top_right = np.array(mouth_top_right, np.int32)
    mouth_bottom_left = np.array(mouth_bottom_left, np.int32)
    mouth_bottom_middle = np.array(mouth_bottom_middle, np.int32)
    mouth_bottom_right = np.array(mouth_bottom_right, np.int32)

    # 입술은 흰색, 배경은 검정색인 마스크 생성 (흰색인 부분만 합성)
    mask = np.zeros(actor_img.shape, dtype=np.float64)
    mask = cv2.fillConvexPoly(mask, mouth_top_left, (255, 255, 255))
    mask = cv2.fillConvexPoly(mask, mouth_top_middle, (255, 255, 255))
    mask = cv2.fillConvexPoly(mask, mouth_top_right, (255, 255, 255))
    mask = cv2.fillConvexPoly(mask, mouth_bottom_left, (255, 255, 255))
    mask = cv2.fillConvexPoly(mask, mouth_bottom_middle, (255, 255, 255))
    mask = cv2.fillConvexPoly(mask, mouth_bottom_right, (255, 255, 255))

    # 자연스럽게 합성하기 위해 블러 처리
    mask = cv2.GaussianBlur(mask * 255.0, (1, 1), 0) / 255.0
    mask[np.where(mask > 0.9)] = 1.0


    # 합성한 입술 경계선을 흐리게 만들어 주기 위해서 다시 블러 처리
    mask = cv2.GaussianBlur(mask * 255.0, (21, 21), 0) / 255.0

    temp = repaint_actor_img * (1.0 - mask)
    temp2 = actor_img * mask


    # repaint 배우 이미지에 입술 합성
    res = temp + temp2

    # 저장했다가 써야 에러 안남
    cv2.imwrite(resource_path+"/lip_synthesis.jpg", res)
    res = cv2.imread(resource_path+"/lip_synthesis.jpg")

    return res