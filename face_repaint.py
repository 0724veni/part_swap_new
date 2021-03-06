import cv2
import numpy as np
from resize_point import resize_point

def make_inpaint_img(landmarks, sizes):
    """
    inpaint을 하기 위해 필요한 마스크를 생성한다.

    param
        landmarks = 얼굴의 부위를 점으로 찍은 랜드마크 좌표들이다.
        sizes = 이미지의 사이지 정보이다. (height, width)

    landmark 좌표 별 부위
    눈썹 왼쪽 : 17~21
        오른쪽 : 22~26

    눈 왼쪽 : 36~41
        오른쪽 : 42~47

    코 : 30~35

    입 : 48~59
    """

    inpaint = np.zeros((sizes[0], sizes[1], 1), np.uint8)

    # 지울 영역을 표시할 색이다.
    white_color = (255, 255, 255)

    # 눈썹을 지움
    # left_eyebrow = resize_point([landmarks[36],landmarks[37], landmarks[38],
    #                      landmarks[39], landmarks[17], landmarks[18], landmarks[19],
    #                      landmarks[20], landmarks[21]], size=0.5, x=1.5, y=1)
    # right_eyebrow = resize_point([landmarks[42],landmarks[43], landmarks[44],
    #                       landmarks[45], landmarks[22], landmarks[23], landmarks[24],
    #                       landmarks[25], landmarks[26]], size=0.5, x=1.5, y=1)
    # left_eyebrow = np.array(left_eyebrow, np.int32)
    # right_eyebrow = np.array(right_eyebrow, np.int32)
    # 눈을 지움 ,
    # left = resize_point([landmarks[36],
    #                      landmarks[39], landmarks[40], landmarks[41],
    #                      landmarks[17], landmarks[18], landmarks[19],
    #                      landmarks[20], landmarks[21]], size=1.8, x=2, y=1)
    # right = resize_point([landmarks[42],
    #                       landmarks[45], landmarks[46], landmarks[47],
    #                       landmarks[22], landmarks[23], landmarks[24],
    #                       landmarks[25], landmarks[26]], size=1.9, x=2, y=1)
    left = resize_point([landmarks[36],landmarks[37], landmarks[38],
                         landmarks[39], landmarks[40], landmarks[41]], size=1.8, x=2, y=1)
    right = resize_point([landmarks[42],landmarks[43], landmarks[44],
                          landmarks[45], landmarks[46], landmarks[47]], size=1.9, x=2, y=1)

    left = np.array(left, np.int32)
    right = np.array(right,np.int32)

    # 코
    nose_figure = resize_point([landmarks[29], landmarks[31], landmarks[33], landmarks[35]],
                               size=1.5, x=1, y=2)
    nose = np.array(nose_figure, np.int32)

    # # 입 영역 지정
    lip_figure = resize_point([landmarks[48], landmarks[49], landmarks[50], landmarks[51],
                               landmarks[52], landmarks[53], landmarks[54], landmarks[55],
                               landmarks[56], landmarks[57], landmarks[58], landmarks[59]],
                              size = 0.3)
    lip = np.array(lip_figure, np.int32)

    # 눈/코/입 흰색으로 채우기
    # inpaint = cv2.fillConvexPoly(inpaint, left_eyebrow, white_color)
    # inpaint = cv2.fillConvexPoly(inpaint, right_eyebrow, white_color)
    inpaint = cv2.fillConvexPoly(inpaint, left, white_color)
    inpaint = cv2.fillConvexPoly(inpaint, right, white_color)
    inpaint = cv2.fillConvexPoly(inpaint, nose, white_color)
    inpaint = cv2.fillConvexPoly(inpaint, lip, white_color)

    return inpaint


def face_repaint(image, landmarks, resource_path):

    # 여기서 입력 얼굴의 랜드마크 정보를 토대로
    # 지워야 하는 부분을 지정해서 칠한다!
    # inpaint type : <class 'numpy.ndarray'>, len : 1800, shape : (1800, 1440, 1)
    inpaint = make_inpaint_img(landmarks, image.shape)
    # print("inpaint type : {}, len : {}, shape : {}".format(type(inpaint), len(inpaint), inpaint.shape))
    cv2.imwrite(resource_path+"/inpaint.jpg",inpaint)
    # face_repaint type : <class 'numpy.ndarray'>, len : 1800, shape : (1800, 1440, 3)
    face_repaint = cv2.inpaint(image, inpaint, 10, cv2.INPAINT_TELEA)
    # print("face_repaint type : {}, len : {}, shape : {}".format(type(face_repaint), len(face_repaint), face_repaint.shape))
    cv2.imwrite(resource_path + "/brow_face_repaint.jpg", face_repaint)

    return face_repaint

def test_face_repaint(image, landmarks, inpaint_path):

    # 여기서 입력 얼굴의 랜드마크 정보를 토대로
    # 지워야 하는 부분을 지정해서 칠한다!
    # inpaint type : <class 'numpy.ndarray'>, len : 1800, shape : (1800, 1440, 1)
    inpaint = make_inpaint_img(landmarks, image.shape)
    # print("inpaint type : {}, len : {}, shape : {}".format(type(inpaint), len(inpaint), inpaint.shape))
    # cv2.imwrite(inpaint_path+"/inpaint.jpg",inpaint)
    # face_repaint type : <class 'numpy.ndarray'>, len : 1800, shape : (1800, 1440, 3)
    face_repaint = cv2.inpaint(image, inpaint, 10, cv2.INPAINT_TELEA)
    # print("face_repaint type : {}, len : {}, shape : {}".format(type(face_repaint), len(face_repaint), face_repaint.shape))
    cv2.imwrite(inpaint_path, face_repaint)

    return face_repaint