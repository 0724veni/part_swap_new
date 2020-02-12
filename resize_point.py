
BLUR_AMOUNT = 51
SWAP = 1

# 왼쪽 눈의 68개 특징점에 대한 색인 값
LEFT_EYE_POINTS = list(range(36, 42))
# 오른쪽 눈
RIGHT_EYE_POINTS = list(range(42, 48))
# 얼굴의 가우스 모호함을 계산하는 데 사용
COLOUR_CORRECT_BLUR_FRAC = 0.6


def resize_point(points, size=0.3, x=1, y=1) :
    """
    여기는 도형의 크기를 늘린다.
    먼저 도형의 꼭짓점 좌표를 받은 다음 좌표들의 중심점을 찾는다.
    그냥 다 더한 다음 갯수만큼 나눌 예정이다.
    그것들을 모두 합해서 계산할 예정이다.

    args :
        points
        size
    """

    # 점들의 합산 값을 저장할 변수다.
    allx = 0
    ally = 0

    # 절대 좌표 값을 모두 합산한다.
    for index, value in enumerate(points):
        # print(index)
        # print(value)
        # print(value.item(0))
        allx += value[0]
        ally += value[1]
        # allx += value.item(0)
        # ally += value.item(1)

    centerX = allx/len(points)
    centerY = ally/len(points)

    vertices = []

    # 여기서 상대 좌표를 얻어낸다.
    for index, value in enumerate(points):
        # relativeX = value.item(0) - centerX
        # relativeY = value.item(1) - centerY
        #
        # vertices.append([value.item(0) + (relativeX * size / x), value.item(1) + (relativeY * size / y)])
        relativeX = value[0] - centerX
        relativeY = value[1] - centerY

        vertices.append([value[0] + (relativeX*size / x ), value[1] + (relativeY*size / y)])

    return vertices