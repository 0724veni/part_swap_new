import os
import cv2
import argparse
import dlib
import numpy as np
import scipy.spatial as spatial

PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTOR_PATH)

## Face detection
def face_detection(img,upsample_times=1):
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    detector = dlib.get_frontal_face_detector()
    faces = detector(img, upsample_times)

    return faces

def select_face(im, landmark, r=10):
    points = landmark
    im_w, im_h = im.shape[:2]
    left, top = np.min(points, 0)
    right, bottom = np.max(points, 0)

    x, y = max(0, left-r), max(0, top-r)
    w, h = min(right+r, im_h)-x, min(bottom+r, im_w)-y

    return points - np.asarray([[x, y]]), (x, y, w, h), im[y:y+h, x:x+w]

def triangular_affine_matrices(vertices, src_points, dst_points):
    """
    Calculate the affine transformation matrix for each
    triangle (x,y) vertex from dst_points to src_points
    :param vertices: array of triplet indices to corners of triangle
    :param src_points: array of [x, y] points to landmarks for source image
    :param dst_points: array of [x, y] points to landmarks for destination image
    :returns: 2 x 3 affine matrix transformation for a triangle
    """
    ones = [1, 1, 1]
    for tri_indices in vertices:
        src_tri = np.vstack((src_points[tri_indices, :].T, ones))
        dst_tri = np.vstack((dst_points[tri_indices, :].T, ones))
        mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]
        yield mat

## Face and points detection
def face_points_detection(img, bbox:dlib.rectangle):
    # Get the landmarks/parts for the face in box d.
    shape = predictor(img, bbox)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    coords = np.asarray(list([p.x, p.y] for p in shape.parts()), dtype=np.int)

    # return the array of (x, y)-coordinates
    return coords

def grid_coordinates(points):
    """ x,y grid coordinates within the ROI of supplied points
    :param points: points to generate grid coordinates
    :returns: array of (x, y) coordinates
    """
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0]) + 1
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1]) + 1

    return np.asarray([(x, y) for y in range(ymin, ymax)
                       for x in range(xmin, xmax)], np.uint32)

## 3D Transform
def bilinear_interpolate(img, coords):
    """ Interpolates over every image channel
    http://en.wikipedia.org/wiki/Bilinear_interpolation
    :param img: max 3 channel image
    :param coords: 2 x _m_ array. 1st row = xcoords, 2nd row = ycoords
    :returns: array of interpolated pixels with same shape as coords
    """
    int_coords = np.int32(coords)
    x0, y0 = int_coords
    dx, dy = coords - int_coords

    # 4 Neighour pixels
    q11 = img[y0, x0]
    q21 = img[y0, x0 + 1]
    q12 = img[y0 + 1, x0]
    q22 = img[y0 + 1, x0 + 1]

    btm = q21.T * dx + q11.T * (1 - dx)
    top = q22.T * dx + q12.T * (1 - dx)
    inter_pixel = top * dy + btm * (1 - dy)

    return inter_pixel.T


def process_warp(src_img, result_img, tri_affines, dst_points, delaunay):
    """
    Warp each triangle from the src_image only within the
    ROI of the destination image (points in dst_points).
    """
    roi_coords = grid_coordinates(dst_points)
    # indices to vertices. -1 if pixel is not in any triangle
    roi_tri_indices = delaunay.find_simplex(roi_coords)

    for simplex_index in range(len(delaunay.simplices)):
        coords = roi_coords[roi_tri_indices == simplex_index]
        num_coords = len(coords)
        out_coords = np.dot(tri_affines[simplex_index],
                            np.vstack((coords.T, np.ones(num_coords))))
        x, y = coords.T
        result_img[y, x] = bilinear_interpolate(src_img, out_coords)

    return None

def warp_image_3d(src_img, src_points, dst_points, dst_shape, dtype=np.uint8):
    rows, cols = dst_shape[:2]
    result_img = np.zeros((rows, cols, 3), dtype=dtype)

    # 배우 얼굴 랜드마크를 삼각분할 하는 과정
    # input - 배우 얼굴에서 입을 제외한 48개의(턱,눈,코) 좌표
    # output - type(delaunay) : scipy.spatial.qhull.Delaunay
    delaunay = spatial.Delaunay(dst_points)

    # print(type(delaunay.simplices), delaunay.simplices)
    # print(len(delaunay.simplices)) -> 71
    """
    # 유저 얼굴 랜드마크를 배우 얼굴에 따라 어느 정도 비틀지에 대한 정보를 받는 곳
    # input 
    # delaunay.simplices : 배우 얼굴에서 그린 삼각형 꼭짓점에 대한 정보  
    # type - numpy.ndarray len 예) 71
    # src_points : 유저 48개 랜드마크
    # dst_points : 배우 48개 랜드마크  
     type(triangular_affine_matrices(delaunay.simplices, src_points, dst_points) : generator
    """
    tri_affines = np.asarray(list(triangular_affine_matrices(delaunay.simplices, src_points, dst_points)))
    # type(tri_affines) : numpy.ndarray
    # print("type(tri_affines)",type(tri_affines))
    # print(tri_affines)
    cv2.imwrite("img_resource/tri_affines.jpg",tri_affines)
    # cv2.imshow("result_img brfore", result_img)
    process_warp(src_img, result_img, tri_affines, dst_points, delaunay)
    # cv2.imshow("result_img after", result_img)
    # cv2.waitKey()

    return result_img

def mask_from_points(size, points,erode_flag=1):
    radius = 10  # kernel size
    kernel = np.ones((radius, radius), np.uint8)

    mask = np.zeros(size, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
    if erode_flag:
        mask = cv2.erode(mask, kernel,iterations=1)

    return mask


def select_userface(im, adjusted_user_lf, r=10):
    points = np.asarray(adjusted_user_lf, dtype=np.int)
    im_w, im_h = im.shape[:2]
    left, top = np.min(points, 0)
    right, bottom = np.max(points, 0)

    x, y = max(0, left-r), max(0, top-r)
    w, h = min(right+r, im_h)-x, min(bottom+r, im_w)-y

    return points - np.asarray([[x, y]]), (x, y, w, h), im[y:y+h, x:x+w]

def faceswap(user_img, user_landmark, actor_img, actor_landmark):
    """
        유저-배우 얼굴 전체 합성
    :param user_img:
    :param user_landmark:
    :param actor_img:
    :param actor_landmark:
    :return:
    """
    # Select src face
    user_points, user_shape, user_face = select_userface(user_img, user_landmark)
    # Select dst face
    actor_points, actor_shape, actor_face = select_face(actor_img, actor_landmark)

    h, w = actor_face.shape[:2]

    warped_user_face = warp_image_3d(user_face, user_points[:68], actor_points[:68], (h, w))

    mask = mask_from_points((h, w), actor_points)
    mask_user = np.mean(warped_user_face, axis=2) > 0
    mask = np.asarray(mask * mask_user, dtype=np.uint8)

    ## Shrink the mask
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)

    ##Poisson Blending
    r = cv2.boundingRect(mask)
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    output = cv2.seamlessClone(warped_user_face, actor_face, mask, center, cv2.NORMAL_CLONE)

    x, y, w, h = actor_shape
    dst_img_cp = actor_img.copy()

    dst_img_cp[y:y + h, x:x + w] = output

    output = dst_img_cp

    return output
