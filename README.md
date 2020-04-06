# part_swap_new
전체 합성이후 눈, 코, 입 따로 합성

## 사용 방법 
1. 3dmodel 합성 방식에서 패키지 설치를 했다면 1번은 건너 뛰어도 됩니다.   
pip install -r requirements.txt

2. 사용자 이미지는 content/user_images에 위치시킨다.   
배우 이미지는 main.py -> contents/images/original   
gifswap.py -> contents/gif/frames/'배우이름'   
videoswap.py -> contents/video/frames/'배우이름'에 위치시킨다.
   
3. 이미지 1장 합성은 main.py   
gif 합성은 gifswap.py   
video 합성은 videoswap.py에서 사용자 이미지 배우 프레임의 경로를 정한다.

4. 각각 파일을 실행시켜 합성한다.   

### 유의사항 
gif, video의 경우 얼굴 프로토콜 json 파일이 각각 contents/gif/frames/'배우이름',      
contents/video/frames/'배우이름'에 위치해야 한다.

## 구조 
### 합성 파일
    main.py - 이미지 1장
    gifswap.py - gif
    videoswap.py - video

### 배우 입 안을 합성하는 파일
    actor_mouth_synthesis.py
### 배우 얼굴을 지우는 파일
    face_repaint.py
### 배우 얼굴 각도, 피부색에 맞게 사용자의 얼굴을 바꾸는 파일
    faceswap_3d.py
### 배우 입술 색깔을 사용자에게 합성하는 파일
    lip_synthesis.py
### 눈, 코, 입 부분합성 하는 파일
    partial_synthesis.py
### 눈, 코, 입 랜드마크 범위 조절 하는 파일
    resize_point.py
### 사용자의 피부색 변경하는 파일
    skin_color_change.py
### 떨림 현상을 줄이기 위해 랜드마크 좌표를 조정하는 파일
    utils.py
### 랜드마크 68개를 추출하기 위해 사용하는 모델
    shape_predictor_68_face_landmarks.dat
### gif, video에서 프레임을 추출하는 파일
    getframes.py
   
### contents
    사용자 이미지 - user_images
    배우 이미지 - images/original
    배우 gif - gif/frames : gif 파일의 프레임들, 프로토콜 json 파일
            - gif/original : gif 파일  
    배우 video - video/frames : video 파일의 프레임들, 프로토콜 json 파일
            - video/original : video 파일
    합성 과정에서 만들어진 파일들 - resource
    
