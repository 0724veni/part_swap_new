import os
import cv2
gif_path = "contents/gif"
video_path = "contents/video"
giflist = os.listdir(gif_path+"/original")
# videolist = os.listdir(video_path+"/original")

def getframes(dir_file_name, type):
    if type =="video":
        cap = cv2.VideoCapture(video_path+"/original/"+dir_file_name)
    else:
        cap = cv2.VideoCapture(gif_path + "/original/" + dir_file_name)
    framenum = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            if type == "gif":
                if framenum <= 9:
                    print(dir_file_name.split(".")[0] + "-000" + str(framenum) + ".jpg")
                    cv2.imwrite(gif_path+"/frames/"+dir_file_name.split(".")[0] + "/"+dir_file_name.split(".")[0]+"-000" + str(framenum) + ".jpg", frame)
                elif framenum <= 99:
                    cv2.imwrite(gif_path+"/frames/"+dir_file_name.split(".")[0]  + "/"+dir_file_name.split(".")[0]+"-00" + str(framenum) + ".jpg", frame)
                elif framenum <= 999:
                    cv2.imwrite(gif_path+"/frames/"+dir_file_name.split(".")[0]  + "/"+dir_file_name.split(".")[0]+"-0" + str(framenum) + ".jpg", frame)
                framenum += 1
            else:
                frame = cv2.resize(frame, dsize=(0,0), fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
                if framenum <= 9 and framenum % 2 == 0:
                    # print(dir_file_name.split(".")[0] + "-0000" + str(framenum) + ".jpg")
                    cv2.imwrite(video_path+"/frames/"+dir_file_name.split(".")[0] + "/"+dir_file_name.split(".")[0]+"-0000" + str(framenum) + ".jpg", frame)
                elif framenum <= 99 and framenum % 2 == 0:
                    cv2.imwrite(video_path+"/frames/"+dir_file_name.split(".")[0]  +"/"+dir_file_name.split(".")[0]+ "-000" + str(framenum) + ".jpg", frame)
                elif framenum <= 999 and framenum % 2 == 0:
                    cv2.imwrite(video_path+"/frames/"+dir_file_name.split(".")[0]  + "/"+dir_file_name.split(".")[0]+"-00" + str(framenum) + ".jpg", frame)
                elif framenum <= 9999 and framenum % 2 == 0:
                    cv2.imwrite(video_path+"/frames/"+dir_file_name.split(".")[0]  + "/"+dir_file_name.split(".")[0]+"-0" + str(framenum) + ".jpg", frame)
                framenum += 1
        else:
            break
    cap.release()

# for gifname in giflist:
#     # 배우 프레임들이 담길 디렉토리 생성
#     frame_dir_path = gif_path+"/frames/"+gifname.split(".")[0]
#     if not (os.path.isdir(frame_dir_path)):
#         os.makedirs(frame_dir_path)
#     type = "gif"
#     getframes(gifname, type)
videolist = ["iuletter_1.mp4", "iuletter_2.mp4","iupalette_1.mp4","iupalette_2.mp4",
             "iuletter_3.mp4", "iuletter_4.mp4","iupalette_3.mp4","iupalette_4.mp4"]

# videolist = ["actor-1.mp4", "actor-2.mp4"]
for videoname in videolist:
    # 배우 프레임들이 담길 디렉토리 생성
    frame_dir_path = video_path+"/frames/"+videoname.split(".")[0]
    if not (os.path.isdir(frame_dir_path)):
        os.makedirs(frame_dir_path)
    type = "video"
    getframes(videoname, type)

