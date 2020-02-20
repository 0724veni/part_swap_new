import numpy as np
import cv2

def mouth_crop(actor_img, actor_landmarks):
    """
    :param actor_img: 유저 이미지
    :param actor_landmarks: 유저 랜드마크
    :return: 입술 안쪽만 크롭한 이미지
    """

    # 검정색 마스크 만들기
    mask = np.ones(actor_img.shape, dtype=np.uint8)
    mask.fill(0)

    # 입술 안쪽만 따기
    mouth_point = np.array([[actor_landmarks[60], actor_landmarks[61], actor_landmarks[62],
                             actor_landmarks[63], actor_landmarks[64], actor_landmarks[65],
                             actor_landmarks[66], actor_landmarks[67], actor_landmarks[60]]], dtype=np.int32)
    # 입술 안쪽 부분 흰색으로 채우기
    cv2.fillPoly(mask, mouth_point, (255, 255, 255))

    # 입술 안쪽 top ,left, right ,bottom
    top_mouth = [actor_landmarks[60][1], actor_landmarks[61][1], actor_landmarks[62][1], actor_landmarks[63][1],
                 actor_landmarks[64][1]]
    bottom_mouth = [actor_landmarks[65][1], actor_landmarks[66][1], actor_landmarks[67][1]]
    top = min(top_mouth)
    left = actor_landmarks[60][0]
    right = actor_landmarks[64][0]
    bottom = max(bottom_mouth)

    width = right - left
    height = bottom - top

    # user_img 에서 입술 안쪽을 제외한 부분은 검은색
    masked_image = cv2.bitwise_and(actor_img, mask)
    # 입술 안쪽만 자르기
    resize_masked_image = masked_image[top:top + height, left:left + width]

    return resize_masked_image


def get_mouth_mask(img, landmarks):
    """
     :param img: 원본 이미지
     :param landmarks: 랜드마크
     :return: 입 마스크 (입 검정 배경 흰색)
    """

    lip_mask = np.zeros(img.shape, dtype=np.float64)
    # 입술 안쪽만 따기
    left_mouth_point = np.array([[landmarks[60], landmarks[61], landmarks[62],
                                  landmarks[66], landmarks[67], landmarks[60]]], dtype=np.int32)
    right_mouth_point = np.array([[landmarks[62], landmarks[63], landmarks[64],
                                   landmarks[65], landmarks[66], landmarks[62]]], dtype=np.int32)

    lip_mask = cv2.fillConvexPoly(lip_mask, left_mouth_point, [1, 1, 1])
    lip_mask = cv2.fillConvexPoly(lip_mask, right_mouth_point, [1, 1, 1])

    return lip_mask


def mouth_synthesis(user_img, actor_img, actor_landmarks, resource_path):

    # 배우 입 안 이미지
    mouth_img = mouth_crop(actor_img, actor_landmarks)

    # 입 안은 흰색, 나머지는 검정색
    user_mask = get_mouth_mask(user_img, actor_landmarks)

    # 유저 사진에서 합성할 부위 검정색으로 만들기
    temp1 = user_img * (1.0 - user_mask)

    # 합성할 이미지 top,left 좌표 값
    left = actor_landmarks[60][0]
    top = [actor_landmarks[60][1], actor_landmarks[61][1], actor_landmarks[62][1], actor_landmarks[63][1],
           actor_landmarks[64][1]]
    top = min(top)

    user_img[top: top + mouth_img.shape[0], left: left + mouth_img.shape[1]] = mouth_img

    temp2 = user_img * user_mask

    res = temp1 + temp2

    # 저장했다가 써야 에러 안남
    cv2.imwrite(resource_path+"/in_actormouth_synthesis.jpg", res)
    res = cv2.imread(resource_path+"/in_actormouth_synthesis.jpg")

    return res



