import numpy as np
import math, cv2, os
import glob
import imageio

def threeaverave(bfactorlf_list):
    # len(bfactorlf_list) -> 영상 프레임의 수 : 75개
    # len(bfactorlf_list[0]) -> 1번 프레임의 랜드마크 수 : 68개

    # 배우 영상의 처음,마지막 프레임을 제외하고
    # 프레임들의 랜드마크 값을 전, 대상, 후 프레임의 평균값으로 한다.
    # 모든 프레임의 랜드마크 리스트를 가져온다.
    # framenum -> 동영상의 프레임 번호
    for framenum in range(0, len(bfactorlf_list)):
        # 해당 프레임이 얼굴인식이 안되었을 경
        target_lf_list = bfactorlf_list[framenum]
        if str(type(target_lf_list)) == "<class 'str'>":
            bfactorlf_list[framenum] = "zero"
            continue

        # 1번째 프레임인 경우 두번째 프레임 랜드마크와 평균
        # 마지막 프레임인 경우 마지막 전 프레임 랜드마크와 평균
        if framenum == 0 or framenum == len(bfactorlf_list)-1:

            # 해당 프레임들의 랜드마크를 가져온다;.
            targetlf_list = []
            secondlf_list = []
            # 1번 프레임인 경우
            if framenum == 0:
                targetlf_list = bfactorlf_list[0]
                secondlf_list = bfactorlf_list[1]
                if str(type(secondlf_list)) == "<class 'str'>":
                    # 2번 프레임 얼굴 인식이 안된 경우
                    if secondlf_list == "zero" or secondlf_list == "overtwo":
                        bfactorlf_list[0] = targetlf_list
                        # 2번 프레임 랜드마크 조정 진행
                        continue
            # 마지막 프레임인 경우
            elif framenum == len(bfactorlf_list)-1:
                targetlf_list = bfactorlf_list[len(bfactorlf_list)-1]
                secondlf_list = bfactorlf_list[len(bfactorlf_list)-2]
                if str(type(secondlf_list)) == "<class 'str'>":
                    # (마지막-1)번 프레임 얼굴 인식이 안된 경우
                    if secondlf_list == "zero" or secondlf_list == "overtwo":
                        bfactorlf_list[len(bfactorlf_list)-1] = targetlf_list
                        # 랜드마크 조정 끝
                        continue

            # 2번 또는 (마지막-1)번 프레임 얼굴 인식이 된 경우
            # 1번, 마지막번 프레임 랜드마크와 평균값 낸다.
            for lf_num in range(0, 68):
                target_face_lf = targetlf_list[lf_num]
                second_face_lf = secondlf_list[lf_num]
                adjusted_x = round((target_face_lf[0] + second_face_lf[0])/2, 8)
                adjusted_y = round((target_face_lf[1] + second_face_lf[1])/2, 8)
                # print("00target_face_lf", type(target_face_lf))
                # 조정된 평균 값을 넣어준다.
                targetlf_list[lf_num] = [adjusted_x, adjusted_y]

            # 1번, 마지막번호의 조정된 랜드마크를 넣어준다.
            if framenum == 0:
                bfactorlf_list[0] = targetlf_list
            elif framenum == len(bfactorlf_list)-1:
                bfactorlf_list[len(bfactorlf_list)-1] = targetlf_list

        # 2번째 ~ 마지막 전 프레임인 경우 전,대상,후 프레임 랜드마크 평균값으로 조정한다.
        else:
            target_framelf_list = bfactorlf_list[framenum]
            bf_framelf_list = bfactorlf_list[framenum-1]
            af_framelf_list = bfactorlf_list[framenum+1]


            if str(type(bf_framelf_list)) == "<class 'str'>" or str(type(af_framelf_list)) == "<class 'str'>":
                if str(type(bf_framelf_list)) == "<class 'str'>" and str(type(af_framelf_list)) == "<class 'str'>":
                    # 이전, 이후 프레임 둘 다 얼굴 인식이 안되는 경우
                    # 현재 프레임 값을 넣는다.
                    bfactorlf_list[framenum] = target_framelf_list

                elif str(type(bf_framelf_list)) == "<class 'str'>":
                    # 이후는 얼굴 인식 되고 이전 프레임만 얼굴 인식이 안되는 경우
                    # 현재 + 이후 프레임 랜드마크 평균을 현재 프레임 랜드마크로 한다.
                    for lf_num in range(0, 68):
                        target_face_lf = target_framelf_list[lf_num]
                        af_face_lf = af_framelf_list[lf_num]
                        # print(framenum)
                        # print("target_face_lf[0], af_face_lf[0]", target_face_lf[0], af_face_lf[0])
                        adjusted_x = round((target_face_lf[0] + af_face_lf[0]) / 2, 8)
                        adjusted_y = round((target_face_lf[1] + af_face_lf[1]) / 2, 8)
                        # 조정된 평균 값을 넣어준다.
                        target_framelf_list[lf_num] = [adjusted_x, adjusted_y]
                    bfactorlf_list[framenum] = target_framelf_list

                elif str(type(af_framelf_list)) == "<class 'str'>":
                    # 이전 얼굴은 인식 되고 이후 프레임만 얼굴 인식이 안되는 경우
                    # 현재 + 이후 프레임 랜드마크 평균을 현재 프레임 랜드마크로 한다.
                    for lf_num in range(0, 68):
                        target_face_lf = target_framelf_list[lf_num]
                        bf_face_lf = bf_framelf_list[lf_num]

                        adjusted_x = round((target_face_lf[0] + bf_face_lf[0]) / 2, 8)
                        adjusted_y = round((target_face_lf[1] + bf_face_lf[1]) / 2, 8)
                        # 조정된 평균 값을 넣어준다.
                        target_framelf_list[lf_num] = [adjusted_x, adjusted_y]
                    bfactorlf_list[framenum] = target_framelf_list
            else:
                # 이전, 이후 프레임 둘 다 얼굴 인식이 되는 경우
                # 이전, 현재, 이후 랜드마크의 평균 값을 현재 랜드마크로 한다.
                for lf_num in range(0, 68):
                    target_face_lf = target_framelf_list[lf_num]
                    bf_face_lf = bf_framelf_list[lf_num]
                    af_face_lf = af_framelf_list[lf_num]

                    adjusted_x = round((target_face_lf[0] + bf_face_lf[0] + af_face_lf[0])/3, 8)
                    adjusted_y = round((target_face_lf[1] + bf_face_lf[1] + af_face_lf[1])/3, 8)
                    # print("target_face_lf", type(target_face_lf))
                    # 조정된 평균 값을 넣어준다.
                    target_framelf_list[lf_num] = [adjusted_x, adjusted_y]

                bfactorlf_list[framenum] = target_framelf_list

    return bfactorlf_list


def makegif(video_name, user_name):
    images = []
    for filename in glob.glob("result/"+video_name+"_"+user_name+"/partial/*.jpg"):
        print("filename",filename)
        images.append(imageio.imread(filename))
    imageio.mimsave("result/"+video_name+"_"+user_name+"/"+video_name+"_partial.gif", images, duration=0.06)


def hangulFilePathImageRead ( filePath ) :
    """
    한글이 들어간 경로에서 이미지를 가져오기 위해 사용하는 방식이다.
    :param filePath:
    :return:
    """
    stream = open( filePath.encode("utf-8") , "rb")
    bytes = bytearray(stream.read())
    numpyArray = np.asarray(bytes, dtype=np.uint8)

    return cv2.imdecode(numpyArray , cv2.IMREAD_UNCHANGED)

def imreadKorean(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None


def tointInTuple(a):
    return [int(x) for x in a]

def getCenterPoint(area):
    """
    여기서는 점 4개를 넣으면 그 점 4개의 중심점을 반환 해준다.
    :param area: 랜드마크의 좌표
    :return:
    """

    point_num = len(area)
    total = (0, 0)

    for point in area:
        total = (total[0] + point[0], total[1] + point[1])

    return (total[0] / point_num, total[1] / point_num)

# def getBox(area, width):
# # def get_box(area, width, height):
#     """
#     여기서는 원하는 얼굴 부위의 좌표 4개를 받는다.
#     그리고 그 부위에서 중심에서 가로로 가장 멀리 떨어진 좌표와
#     세로로 멀어진 떨어진 좌표를 가져온다.
#     :param area: 점을 가져온다/
#     :param width: 가로길이
#     :param height: 세로길이
#     :return:
#     """
#     print(area)
#
#     moveDistance = makeMoveDistance(area)
#
#     # 크기를 키운다.
#     # area = resize_inpaint(area, size=0.2)
#
#     center = getCenterPoint(area)
#
#     return getRetangle(center, width, moveDistance)


def getFaceBiasRate(area):
    """
    여기서는 점과 점사이의 거리를 통해 어느 각도로든 돌아간 좌표를 구한다.
    :param area:
    :return:
    """
    point1 = area[0]
    point2 = area[1]

    pointX = point1[0] - point2[0]

    # 이게 움직일 값이다.
    pointY = point1[1] - point2[1]

    short_line = pythagoras(point1, point2)

    # 직사각형의 높이
    # pointX / 2

    faceBiasRate = pointX / short_line
    moveDistanceRate = pointY / short_line

    return faceBiasRate, moveDistanceRate


def pythagoras (point1, point2) :
    """
    여기는 피타고라스의 정리로 점과 점사이의 직선거리를 구해서 반환 해준다.
    :param point1:
    :param point2:
    :return:
    """
    a = point1[0] - point2[0]
    b = point1[1] - point2[1]

    c = math.sqrt((a * a) + (b * b))
    return c

def makeRect (rightSidePoint, leftSidePoint, absHeight, faceBiasRate, moveDistanceRate) :
    """
    양 부위의 끝 점과 사각형의 높이 값을 통해 사각형의 좌표를 생성한다.
    :param rightSidePoint:
    :param leftSidePoint:
    :param absHeight:
    :return:
    """
    moveDistanceX = moveDistanceRate * absHeight
    moveDistanceY = faceBiasRate * absHeight

    point1 = (rightSidePoint[0] - moveDistanceX, rightSidePoint[1] - moveDistanceY)
    point2 = (leftSidePoint[0] - moveDistanceX, leftSidePoint[1] - moveDistanceY)
    point3 = (leftSidePoint[0] + moveDistanceX, leftSidePoint[1] + moveDistanceY)
    point4 = (rightSidePoint[0] + moveDistanceX, rightSidePoint[1] + moveDistanceY)

    # return (point1, point2, point3, point4)
    return (tointInTuple(point1), tointInTuple(point2), tointInTuple(point3), tointInTuple(point4))



# 얼굴 부위에 영역을 반환 해준다.
def makeAreaRect (landmark) :

    # 얼굴 치우침 비율을 가져온다
    faceBiasRate, moveDistanceRate = getFaceBiasRate([landmark[21], landmark[22]])

    # 눈썹의 영역 생성
    eyebrowsHeight = pythagoras(landmark[21], landmark[22])/2
    eyebrows = makeRect(landmark[17], landmark[26], eyebrowsHeight, faceBiasRate, moveDistanceRate)

    # 왼쪽눈 영역생성
    eyeLefeHeight = pythagoras(landmark[38], landmark[41])
    eye_left = makeRect(landmark[36], landmark[39], eyeLefeHeight, faceBiasRate, moveDistanceRate)

    # 오른쪽 눈의 높이와 사각형 영역 생성
    # eyeRightHeight = (landmark[47] - landmark[44])
    eyeRightHeight = pythagoras(landmark[47], landmark[44])
    eye_right = makeRect(landmark[42], landmark[45], eyeRightHeight, faceBiasRate, moveDistanceRate)

    # 입술의 높이 생성 및
    # lip_height = (landmark[48][1] + landmark[54][1])/2 - landmark[57][1]
    # 피타고라스의 정리로 수직 거리의 높이를 파악한다.

    lip_height = pythagoras(getCenterPoint([landmark[48], landmark[54]]), landmark[57])
    lip = makeRect(landmark[48], landmark[54], lip_height, faceBiasRate, moveDistanceRate)

    return eyebrows, eye_left, eye_right, lip

def getResizeRate (img, resizeShape) :
    """
    이미지를 640*480 크기에 비율에 맞춰서 키우도록 한다.
    :param img:
    :return:
    """
    maxWidth = resizeShape[0]
    maxHeight = resizeShape[1]

    imgWidth = img.shape[1]
    imgHeight = img.shape[0]

    widthGap = maxWidth - imgWidth
    heightGap = maxHeight - imgHeight

    if widthGap >= heightGap :
        resizeRate = (maxWidth / imgWidth)
        return (int(imgWidth*resizeRate), int(imgHeight*resizeRate))
    else :
        resizeRate = (maxHeight / maxHeight)
        return (int(imgWidth * resizeRate), int(imgHeight * resizeRate))


def imageResize (img, resizeShape = (640, 480)) :
    """
    여기서 이미지를 새로운 사이즈로 변ㄱㅇ 햊ㄴ다
    :param img: 
    :param resizeShape: 
    :return: 
    """
    return cv2.resize(img, getResizeRate(img, resizeShape), 1, 1, interpolation=cv2.INTER_CUBIC)


def make_path(file_path):
    """
    저장 양식
    얼굴 부위에서 눈과 눈썹은 왼쪽 오른쪽 구분해서 저장함
    user_protocol/(유저의 이름)/(얼굴 부위)/(표정)/표정1.jpg
    :param file_path: str
    :return:
    """
    path_names = file_path.split("/")

    file_path = ""

    # 모든 경로가 만들어져있는 지 확인하고 저장한다.
    for path in path_names:
        file_path += path
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        file_path += "/"
