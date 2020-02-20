import cv2
import numpy as np
from resize_point import *


def mouth_synthesis(actor_img, new_user_img, user_landmarks, actor_inpaint_img):

    mouth = np.array([user_landmarks[48], user_landmarks[49], user_landmarks[50], user_landmarks[51],
                      user_landmarks[52], user_landmarks[53], user_landmarks[54], user_landmarks[55],
                      user_landmarks[56], user_landmarks[57], user_landmarks[58], user_landmarks[59],
                      user_landmarks[48]], np.int32)

    mouth = np.array(mouth, np.int32)

    actor_convexhull = cv2.convexHull(mouth)
    actor_img_gray = cv2.cvtColor(actor_img, cv2.COLOR_BGR2GRAY)
    face_mask = np.zeros_like(actor_img_gray)
    mask = cv2.fillConvexPoly(face_mask, actor_convexhull, 255)

    (x, y, w, h) = cv2.boundingRect(actor_convexhull)

    mouth_img = new_user_img[y:y + h, x:x + w]

    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

    synthesis_actor_inpaint_img = actor_inpaint_img.copy()
    synthesis_actor_inpaint_img[y:y + h, x:x + w] = mouth_img

    result = cv2.seamlessClone(synthesis_actor_inpaint_img, actor_inpaint_img, mask, center_face2, cv2.NORMAL_CLONE)

    return result

def nose_synthesis(actor_img, new_user_img, user_landmarks, actor_inpaint_img, ratio) :
    # print("user_landmarks[31] 뭐니? ", user_landmarks[31])
    # print("user_landmarks[31, 1] 뭐니? ", user_landmarks[31, 1])
    # 유저 코 영역 좌표값
    user_top_nose = user_landmarks[27][1]
    user_left_nose = user_landmarks[31][0] - ratio * 10
    user_right_nose = user_landmarks[35][0] + ratio * 11
    bottom = [user_landmarks[31][1], user_landmarks[32][1], user_landmarks[33][1],
              user_landmarks[34][1], user_landmarks[35][1]]

    user_bottom_nose = max(bottom) + ratio*4

    user_nose_width = user_right_nose - user_left_nose
    user_nose_height = user_bottom_nose - user_top_nose

    # create a mask with white pixels
    user_nose_mask = np.ones(new_user_img.shape, dtype=np.uint8)
    user_nose_mask.fill(255)

    # points to be cropped
    nose_point = np.array([[(user_landmarks[27][0] - ratio * 7, user_top_nose),
                            (user_landmarks[27][0] + ratio * 7, user_top_nose),
                            (user_right_nose, user_bottom_nose),
                            (user_left_nose, user_bottom_nose)]], dtype=np.int32)
    # fill the ROI into the mask
    cv2.fillPoly(user_nose_mask, nose_point, 0)

    # applying th mask to original image
    masked_image = cv2.bitwise_or(new_user_img, user_nose_mask)
    user_nose_img = masked_image[user_top_nose:user_top_nose + user_nose_height,
                    user_left_nose:user_left_nose + user_nose_width]

    # 모든 점들을 포함하는 최소 크기의 볼록 다각형
    nose_convexhull = cv2.convexHull(nose_point)

    actor_img_gray = cv2.cvtColor(actor_img, cv2.COLOR_BGR2GRAY)
    # 마스크 만들기 위한 검은색 이미지
    face_mask = np.zeros_like(actor_img_gray)
    # face_mask 코 영역 흰색으로 그리기
    mask = cv2.fillConvexPoly(face_mask, nose_convexhull, 255)
    (x, y, w, h) = cv2.boundingRect(nose_convexhull)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

    synthesis_actor_inpaint_img = actor_inpaint_img.copy()

    synthesis_actor_inpaint_img[user_top_nose: user_top_nose + user_nose_height,
    user_left_nose: user_left_nose + user_nose_width] = user_nose_img

    # seamlessClone : 한 이미지에 객체를 복사해 다른 이미지에 붙여 넣어 매끄럽고 자연스러운 구성을 만들 수 있다
    # 합성한 이미지 / 연예인 원본 이미지 / 얼굴 흰색,배경 검정 마스크 / 연예인 랜드마크 값 계산한 것
    result = cv2.seamlessClone(synthesis_actor_inpaint_img, actor_inpaint_img, mask, center_face2, cv2.MIXED_CLONE)

    return result


def right_eye(actor_img, new_user_img, landmarks, actor_repaint_img):

    actor_img_gray = cv2.cvtColor(actor_img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(actor_img_gray)

    middle = np.array([landmarks[43], landmarks[44], landmarks[46], landmarks[47]])
    middle_figure = resize_point(middle, size=2.5)

    left = np.array([landmarks[42], landmarks[43], landmarks[47]])
    left_figure = resize_point(left, size=2.5)

    right = np.array([landmarks[44], landmarks[45], landmarks[46]])
    right_figure = resize_point(right, size=2)

    point = middle_figure + left_figure + right_figure

    figure = np.array(point, np.int32)

    actor_convexhull = cv2.convexHull(figure)
    mask = cv2.fillConvexPoly(mask, actor_convexhull, 255)

    (x, y, w, h) = cv2.boundingRect(actor_convexhull)

    right_eye_img = new_user_img[y:y + h, x:x + w]

    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
    synthesis_actor_inpaint_img = actor_repaint_img.copy()

    print ("right_eye_img = " + str(right_eye_img.shape))
    print ("synthesis_actor_inpaint_img = " + str(synthesis_actor_inpaint_img.shape))

    print ("y = " + str(y))
    print ("y + h = " + str(y + h))
    print ("x = " + str(x))
    print ("x + w = " + str(x + w))

    synthesis_actor_inpaint_img[y:y + h, x:x + w] = right_eye_img

    result = cv2.seamlessClone(synthesis_actor_inpaint_img, actor_repaint_img, mask, center_face2, cv2.MIXED_CLONE)

    return result



def left_eye(actor_img, new_user_img, user_landmarks, actor_repaint_img):
    """
        왼쪽 눈 합성
        왼쪽 눈 영역만 흰색이고 배경은 검정색인 마스크를 이용해서 합성한다.

    :param actor_img:  배우 이미지
    :param new_user_img: 전체 합성한 이미지 (비디오 프레임에 유저 얼굴 들어간 이미지)
    :param user_landmarks: 유저 랜드마크
    :param actor_repaint_img: repaint한 배우 이미지
    :return:
    """

    # 배우 이미지 크기와 같은 마스크 생성
    actor_img_gray = cv2.cvtColor(actor_img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(actor_img_gray)

    # 눈을 세 부분으로 나눠서 계산한 이유는
    # 눈 랜드마크를 모두 포함하는 최소 다각형을 만들어서 해당 다각형 부분만 합성하는 방식을 사용하고 있는데,
    # 하나의 np.array에 모든 눈 랜드마크를 담았더니 완전히 채워지지 않은 다각형이 나오는 현상 발생
    # 우선 임시로 눈을 세 부분으로 쪼개서 만들었더니 빈 부분은 거의 발생하지 않았음

    # 왼쪽 눈 가운데
    middle = np.array([user_landmarks[37], user_landmarks[38], user_landmarks[40], user_landmarks[41]])
    # 눈은 애교살, 쌍커풀과 같은 정보를 포함해야 하기 때문에 찍어준 랜드마크보다 영역이 더 커야한다
    # resize_point 함수를 통해 좌표값을 크게 한다
    middle_figure = resize_point(middle, size=2.5)

    # 왼쪽 눈 왼쪽 부분
    left = np.array([user_landmarks[36], user_landmarks[37], user_landmarks[41]])
    left_figure = resize_point(left, size=2.5)

    # 왼쪽 눈 오른쪽 부분
    right = np.array([user_landmarks[38], user_landmarks[39], user_landmarks[40]])
    right_figure = resize_point(right, size=2)

    point = middle_figure + left_figure + right_figure

    # 왼쪽 눈 좌표 값
    figure = np.array(point, np.int32)

    # 지정한 포인트를 모두 포함하는 복록한 외곽선
    actor_convexhull = cv2.convexHull(figure)

    # 왼쪽 눈 영역만 하얗게 칠함
    mask = cv2.fillConvexPoly(mask, actor_convexhull, 255)

    # 합성 부위의 시작점, 가로 /세로 좌표값
    (x, y, w, h) = cv2.boundingRect(actor_convexhull)

    left_eye_img = new_user_img[y:y + h, x:x + w]

    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
    synthesis_actor_inpaint_img = actor_repaint_img.copy()
    synthesis_actor_inpaint_img[y:y + h, x:x + w] = left_eye_img

    # 자연스러운 색상을 위함
    result = cv2.seamlessClone(synthesis_actor_inpaint_img, actor_repaint_img, mask, center_face2, cv2.MIXED_CLONE)

    return result

def partial_synthesis(actor_img, new_user_img, user_landmarks, actor_repaint_img):

    temple_left = user_landmarks[0][0]
    temple_right = user_landmarks[16][0]

    # 얼굴 비율 계산
    # 이미지마다 얼굴 크기가 다르기 때문에 부분합성 할 영역(눈/코/입)을 얼마큼 자를지 기준을 둬야함
    # 임의로 만든 것이기 때문에 더 좋은 방법이 있다면 수정하는 것이 좋다.
    ratio = int((temple_right - temple_left) / 100)

    # 부분합성 방식은 모두 똑같기 때문에 left_eye함수에만 주석 써 놓음

    # 왼쪽 눈
    actor_repaint_img = left_eye(actor_img, new_user_img, user_landmarks, actor_repaint_img)

    # 오른쪽 눈
    actor_repaint_img = right_eye(actor_img, new_user_img, user_landmarks, actor_repaint_img)
    #
    # actor_repaint_img
    # # 코
    actor_repaint_img = nose_synthesis(actor_img, new_user_img, user_landmarks, actor_repaint_img, ratio)
    #actor_repaint_img
    # # 입
    result = mouth_synthesis(actor_img, new_user_img, user_landmarks, actor_repaint_img)

    return result
