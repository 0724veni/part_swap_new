import numpy as np
import cv2

BLUR_AMOUNT = 51
SWAP = 1

# 왼쪽 눈의 68개 특징점에 대한 색인 값
LEFT_EYE_POINTS = list(range(36, 42))
# 오른쪽 눈
RIGHT_EYE_POINTS = list(range(42, 48))
# 얼굴의 가우스 모호함을 계산하는 데 사용
COLOUR_CORRECT_BLUR_FRAC = 0.6

def transfer_points(points1, points2):

    """
    모사 변환 행렬 획득
    :param points1:
    :param points2:
    :return:
    """

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    # 특이치 분해
    u, s, v_t = np.linalg.svd(points1.T * points2)
    r = (u * v_t).T
    return np.vstack([np.hstack(((s2 / s1) * r, c2.T - (s2 / s1) * r * c1.T)),
                      np.mat([0., 0., 1.])])

def warp_img(img, m, d_shape):
    """
    cv2.warpAffine 함수를 사용하여 이미지를 에뮬레이션
    :param igg:로컬 변환 이미지 대기 중
    :param m: 변환 행렬을 시뮬레이션하여 처음 두 줄만 사용하고,세 번째 줄은 [0,0,1]입니다.
    :param d_shape: 출력 이미지의 크기
    :return: 출력 이미지
    """
    output_img = np.zeros(d_shape, dtype=img.dtype)
    cv2.warpAffine(img,
                   m[:2],
                   (d_shape[1], d_shape[0]),
                   dst=output_img,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_img

def modify_color(img_input, img_target, landmarks1):
    """
    img_target의 이미지 색조를 img_input의 이미지 색조로 변환
    :param igg_input: 배경 이미지
    :param igg_target: 사람의 얼굴 사진
    :param landmarks1: 표적 이미지의 포인트
    :return: 조정된 이미지
    """

    # 좌우 눈금의 평균 위치[[x y]와 [x y]]
    mean1 = np.mean(landmarks1[LEFT_EYE_POINTS], axis=0)
    mean2 = np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0)
    diff = mean1 - mean2

    # 求范式，此处默认二范数，即 根号（x^2 + y^2 + z^2 + ...）
    # 即两眼之间的距离，80左右
    # 즉 두 눈 사이의 거리입니다 80 정도죠
    blur_amount = int((COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(diff))*1.5)

    # 가우스 핵은 기수여야 한다
    if blur_amount % 2 == 0:
        blur_amount += 1

    # 高斯滤波
    # 가우스 필터
    img_input_blur = cv2.GaussianBlur(img_input, (blur_amount, blur_amount), 0)
    img_target_blur = cv2.GaussianBlur(img_target, (blur_amount, blur_amount), 0)
    # 避免出现除0的错误
    # 0을 제외한 오류를 피하려면
    img_target_blur += (128 * (img_target_blur <= 1.0)).astype(img_target_blur.dtype)

    return img_target.astype(np.float64) * img_input_blur.astype(np.float64) / img_target_blur.astype(np.float64)



def skin_color_change(user_img, user_landmark, actor_img, actor_landmark, resource_path):
    """
    배우와 피부색 맞춘 유저 사진 저장
    :param user_img: 유저 이미지
    :param user_landmark: 유저 랜드마크
    :param actor_img: 배우 이미지
    :param actor_landmark: 배우 랜드마크
    :param resource_path: 피부색 바뀐 유저 사진 저장을 위한 폴더명
    """

    # 유저/배우 랜드마크를 mat형식으로 변환
    mat_user_landmarks_points = np.mat(user_landmark)
    mat_actor_landmarks_points = np.mat(actor_landmark)

    # m = 모사 변환 행렬
    m = transfer_points(mat_actor_landmarks_points, mat_user_landmarks_points)
    # 배우와 얼굴크기, 각도 맞춘 이미지
    warped_img_user = warp_img(user_img, m, actor_img.shape)
    cv2.imwrite(resource_path + "/warped_user.jpg", warped_img_user)
    # 액터와 피부색까지 맞춘 유저 이미지
    warped_corrected_img_target = modify_color(actor_img, warped_img_user, mat_actor_landmarks_points)

    cv2.imwrite(resource_path+"/warped_modify_color_user.jpg", warped_corrected_img_target)
    warped_corrected_img_target = cv2.imread(resource_path + "/warped_modify_color_user.jpg")

    return warped_corrected_img_target