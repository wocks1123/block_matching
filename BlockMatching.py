import cv2
import numpy as np

import os
from datetime import datetime
import sys
import math


MICRO_BLOCK_WIDTH   = 8
MICRO_BLOCK_HEIGHT  = 8

OVERLAPPED_WIDTH    = 10
OVERLAPPED_HEIGHT   = 10


def save_frame_files(video_name, num_of_frame):
    video_capture = cv2.VideoCapture(video_name)

    curr_date = datetime.now().strftime("_%Y-%m-%d_%H_%M_%S_")
    file_name = os.path.splitext(video_name)
    file_name = os.path.split(file_name[0])[1] # remove extension
    save_path = os.path.join("frame", file_name + curr_date + str(num_of_frame))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    step = 0
    while video_capture.isOpened():
        ret, image = video_capture.read()

        # 캡쳐된 이미지를 저장하는 함수
        cv2.imwrite(save_path + "/frame%04d.jpg" % step, image)
        print('Saved frame%d.jpg' % step)
        step += 1

        if num_of_frame == step:
            break

    print("saved at " + save_path)
    video_capture.release()
    return save_path


# return (start_w, start_h, end_w, end_h)
def get_search_range(w_min, h_min, w_max, h_max, curr_w, curr_h, w_size, h_size):
    start_w = curr_w - w_size if curr_w - w_size > w_min else w_min
    end_w = curr_w + w_size if curr_w + w_size < w_max else curr_w
    start_h = curr_h - h_size if curr_h - h_size > h_min else h_min
    end_h = curr_h + h_size if curr_h + h_size < h_max else curr_h
    return start_w, start_h, end_w, end_h


def get_mid_mv(motion_vector_list):
    res = []
    for frame in motion_vector_list:
        res_frame = []
        for it in frame:
            res_frame.append([(x / 2, y / 2) for x, y in it])
        res.append(res_frame)
    return np.array(res)


# DECLARED ...
# - use get_mv_bidirectional_blockmatching(prev_img, next_img)
def block_matching(frame_dir, flag):
    # 현재이미지(curr)와 다음 이미지(next)를 하나씩 가져온다
    file_list = os.listdir(frame_dir)

    # 파일들을 [curr, next] 형태로 묶음
    if flag is True:
        # [frame1, frame2], [frame2, frame3], ...
        frames_list = [file_list[idx:idx + 2] for idx in range(0, len(file_list), 1)]
    else:
        # [frame1, frame2], [frame3, frame4], ...
        frames_list = [file_list[idx:idx + 2] for idx in range(0, len(file_list), 2)]

    print(frames_list)

    frame_motion_vector_list = []
    for frames in frames_list:
        curr_frame = cv2.imread(frame_dir + "/" + frames[0], cv2.IMREAD_GRAYSCALE)
        height, width = curr_frame.shape
        # print(height, width)

        # last frame
        if len(frames) == 1:
            break
        next_frame = cv2.imread(frame_dir + "/" + frames[1], cv2.IMREAD_GRAYSCALE)

        # print(curr_frame)
        # print(curr_frame.shape)

        # micro block 크기 구하기...


        # 현재 이미지를 micro block으로 나눈다. 나누어진 블럭 리스트를 가져오기
        micro_block_list = []
        # 각 micro block 좌표
        micro_block_idx_list = []

        for h_idx in range(0, height, MICRO_BLOCK_HEIGHT):
            micro_block_list_row = []
            micro_block_idx_list_row = []
            for w_idx in range(0, width, MICRO_BLOCK_WIDTH):
                # print(w_idx, h_idx, w_idx + micro_block_width, h_idx + micro_block_height)
                # print(curr_frame[h_idx: h_idx + micro_block_height, w_idx: w_idx + micro_block_width].shape)
                micro_block_list_row.append(curr_frame[h_idx: h_idx + MICRO_BLOCK_HEIGHT, w_idx: w_idx + MICRO_BLOCK_WIDTH])
                micro_block_idx_list_row.append((w_idx, h_idx))
            micro_block_idx_list.append(micro_block_idx_list_row)
            micro_block_list.append(micro_block_list_row)


        # print("len 1:",len(micro_block_idx_list))
        # print("len 2:", len(micro_block_idx_list[0]))
        # print(micro_block_idx_list)

        # 블럭 리스트하나마다
        # next 프레임에서의 모션 벡터를 찾는다.(search range)
        # 모션 벡터 계산은 각 픽셀에서 SAD값이 가장 작은 것? 모션벡터 저장
        motion_vector_list = []
        # print(len(micro_block_idx_list))
        # print(len(micro_block_idx_list[0]))
        for h in range(len(micro_block_idx_list)):
            motion_vector_list_row = []
            for w in range(len(micro_block_idx_list[0])):
                # search range 계산
                micro_x, micro_y = micro_block_idx_list[h][w]
                micro_block = micro_block_list[h][w]

                search_range = get_search_range(0, 0, width, height, micro_x, micro_y, MICRO_BLOCK_WIDTH, MICRO_BLOCK_HEIGHT)
                start_w, start_h, end_w, end_h = search_range

                mv = (0, 0)
                for y in range(start_h, end_h):
                    min_value = sys.maxsize
                    for x in range(start_w, end_w):
                        # search range 내에서 모션벡터 계산
                        window_block = next_frame[y:y + MICRO_BLOCK_HEIGHT, x:x + MICRO_BLOCK_WIDTH]
                        value = np.sum(np.abs(micro_block - window_block))
                        # 가장 작은 값 선택
                        if value < min_value:
                            mv = (x - micro_x, y - micro_y)
                            min_value = value

                motion_vector_list_row.append(mv)
            motion_vector_list.append(motion_vector_list_row)

        # print(motion_vector_list)
        # print("motion_vector_list:", len(motion_vector_list))
        # print("motion_vector_list[0] ", len(motion_vector_list[0]))
        frame_motion_vector_list.append(motion_vector_list)

    # print(frame_motion_vector_list)
    # print("frame_motion_vector_list:", len(frame_motion_vector_list))
    # print("frame_motion_vector_list[0]:",len(frame_motion_vector_list[0]))
    # print("frame_motion_vector_list[0][0]:",len(frame_motion_vector_list[0][0]))
    if flag is True:
        # estimated frame?에서 MV는 중간 값
        return get_mid_mv(frame_motion_vector_list)

    return np.array(frame_motion_vector_list)


def make_frame(prev_frame, next_frame, motion_vector):
    prev_frame = cv2.imread(prev_frame)
    next_frame = cv2.imread(next_frame)
    height, width, channel = prev_frame.shape

    prev_frame = prev_frame.astype(np.uint32)
    next_frame = next_frame.astype(np.uint32)

    motion_vector_frame = [[0 for j in range(width)] for i in range(height)]

    overlapped_range = [[[] for j in range(len(motion_vector[i]))]  for i in range(len(motion_vector))]
    overlapped_width = int((OVERLAPPED_WIDTH - MICRO_BLOCK_WIDTH) / 2)
    overlapped_height = int((OVERLAPPED_HEIGHT - MICRO_BLOCK_HEIGHT) / 2)

    # 각 픽셀값에 영항을 주는 모션벡터 저장
    overlapped_motion_vector = [[[] for j in range(width)] for i in range(height)]

    for h in range(0, int(height / MICRO_BLOCK_HEIGHT)):
        for w in range(0, int(width / MICRO_BLOCK_WIDTH)):
            temp_w = w * MICRO_BLOCK_WIDTH
            temp_h = h * MICRO_BLOCK_HEIGHT
            s_x = temp_w - overlapped_width if temp_w - overlapped_width >= 0 else temp_w
            s_y = temp_h - overlapped_height if temp_h - overlapped_height >= 0 else temp_h
            e_x = (w + 1) * MICRO_BLOCK_WIDTH
            e_x = e_x + overlapped_width if e_x + overlapped_width < width else e_x
            e_y = (h + 1) * MICRO_BLOCK_HEIGHT
            e_y = e_y + overlapped_height if e_y + overlapped_height < height else e_y
            overlapped_range[h][w] = (motion_vector[h][w], [[s_x, s_y], [e_x, e_y]])
            for y in range(s_y, e_y):
                for x in range(s_x, e_x):
                    # print("x, y", x, y, "  ", end="")
                    overlapped_motion_vector[y][x].append(motion_vector[h][w])
                # print("")

    # print overlapped mv
    for y in range(height):
        for x in range(width):
            #print("{", overlapped_motion_vector[y][x], "} ", end=" ")
            pass
        #print("\n")


    for h in range(0, int(height / MICRO_BLOCK_HEIGHT)):
        for w in range(0, int(width / MICRO_BLOCK_WIDTH)):
            mv_x, mv_y = motion_vector[h][w]

            for y in range(h * MICRO_BLOCK_HEIGHT, h * MICRO_BLOCK_HEIGHT + MICRO_BLOCK_HEIGHT):
                for x in range(w * MICRO_BLOCK_WIDTH, w * MICRO_BLOCK_WIDTH + MICRO_BLOCK_WIDTH):
                    motion_vector_frame[y][x] = [mv_x, mv_y]

    # print mv frame
    for y in range(height):
        for x in range(width):
            #print(motion_vector_frame[y][x] ,end=" ")
            pass
        #print("\n")

    # print overlap frame
    for h in range(0, int(height / MICRO_BLOCK_HEIGHT)):
        for w in range(0, int(width / MICRO_BLOCK_WIDTH)):
            #mv, xy = overlapped_range[h][w]
            #print("{", mv, xy, "}", end=" ")
            pass
        # print("\n")
        pass

    interpolated_frame = [[0 for j in range(width)] for i in range(height)]
    # print(interpolated_frame.shape)
    for y in range(height):
        for x in range(width):
            sum = 0
            for mv in overlapped_motion_vector[y][x]:
                # print("  mv :", mv[0], mv[1], end="")

                prev_y = y + mv[1]
                if prev_y >= height or prev_y < 0:
                    prev_y = 0 if prev_y < 0 else height - 1

                prev_x = x + mv[0]
                if prev_x >= width or prev_x < 0:
                    prev_x = 0 if prev_x < 0 else width - 1

                next_y = y - mv[1]
                if next_y >= height or next_y < 0:
                    next_y = 0 if next_y < 0 else height - 1

                next_x = x - mv[0]
                if next_x >= width or next_x < 0:
                    next_x = 0 if next_x < 0 else width - 1

                # print("  base_pixel :", prev_frame[prev_y][prev_x], end="")
                # print("  next_frame :", next_frame[next_y][next_x], end="")
                sum += prev_frame[prev_y][prev_x] + next_frame[next_y, next_x]
                # sum += cv2.add(prev_frame[prev_y][prev_x], next_frame[next_y, next_x])
            l = len(overlapped_motion_vector[y][x]) * 2
            res = sum / l
            res = np.array(res).T
            # print("  aft res :", res, " len :", len(overlapped_motion_vector[y][x]))

            # print()
            # interpolated_frame[y][x] = np.array(np.split(res, 3))
            interpolated_frame[y][x] = res.astype(np.uint8)

    in_np = np.array(interpolated_frame)
    # print(in_np)
    for y in range(height):
        for x in range(width):
            #print(in_np[y][x], " ", end="")
            pass
        #print("")

    # print(interpolated_frame)
    #for y in range(height):
        #for x in range(width):
            #print(interpolated_frame[y][x], end="")
        #print("")
    #print('\n'.join(map(''.join, motion_vector_frame)))
    # black_screen = np.zeros(curr_frame.shape)
    # cv2.imwrite("frame/black.jpg", black_screen)

    # 겹쳐지는 부분의 크기를 정합시다.
    return in_np


def get_mv_bidirectional_blockmatching(prev_img, next_img):
    prev_frame = cv2.imread(prev_img)
    next_frame = cv2.imread(next_img)
    height, width, channel = prev_frame.shape

    # micro block 크기 구하기...
    # 현재 이미지를 micro block으로 나눈다. 나누어진 블럭 리스트를 가져오기
    micro_block_list = []
    # 각 micro block 좌표
    micro_block_idx_list = []

    # 현재 이미지를 micro block으로 나눈다. 나누어진 블럭 리스트를 가져오기
    # 각 micro block 좌표
    for h_idx in range(0, height, MICRO_BLOCK_HEIGHT):
        micro_block_list_row = []
        micro_block_idx_list_row = []
        for w_idx in range(0, width, MICRO_BLOCK_WIDTH):
            micro_block_list_row.append(prev_frame[h_idx: h_idx + MICRO_BLOCK_HEIGHT, w_idx: w_idx + MICRO_BLOCK_WIDTH])
            micro_block_idx_list_row.append((w_idx, h_idx))
        micro_block_idx_list.append(micro_block_idx_list_row)
        micro_block_list.append(micro_block_list_row)

    # 블럭 리스트하나마다
    # next 프레임에서의 모션 벡터를 찾는다.(search range)
    # 모션 벡터 계산은 각 픽셀에서 SAD값이 가장 작은 것? 모션벡터 저장
    frame_motion_vector = [[0 for j in range(len(micro_block_idx_list[0]))] for i in range(len(micro_block_idx_list))]
    for h in range(len(micro_block_idx_list)):
        for w in range(len(micro_block_idx_list[0])):
            # search range 계산
            micro_x, micro_y = micro_block_idx_list[h][w]
            micro_block = micro_block_list[h][w]

            search_range = get_search_range(0, 0, width, height, micro_x, micro_y, MICRO_BLOCK_WIDTH,
                                            MICRO_BLOCK_HEIGHT)
            start_w, start_h, end_w, end_h = search_range

            mv = (0, 0)
            min_value = np.infty
            for y in range(start_h, end_h + 1):
                for x in range(start_w, end_w + 1):
                    # search range 내에서 모션벡터 계산
                    window_block = next_frame[y:y + MICRO_BLOCK_HEIGHT, x:x + MICRO_BLOCK_WIDTH]
                    value = np.sum(np.abs(micro_block - window_block))
                    # 가장 작은 값 선택
                    if value < min_value:
                        mv = (x - micro_x, y - micro_y)
                        min_value = value
            frame_motion_vector[h][w] = mv

    return frame_motion_vector


def get_psnr_from_img(image_path):
    img = cv2.imread(image_path)
    mse = (img ** 2).mean()
    psnr = 10 * math.log10((255 ** 2) / mse)
    return psnr


def get_psnr_from_np_arr(arr):
    mse = (arr ** 2).mean()
    psnr = 10 * math.log10((255 ** 2) / mse)
    return psnr


def save_img_from_np_arr(arr, save_path):
    cv2.imwrite(save_path, arr)


def draw_mv_vector(img, mv, save_path):
    frame = cv2.imread(img)

    height, width, channel = frame.shape

    for y in range(0, int(height / MICRO_BLOCK_HEIGHT) - 1):
        for x in range(0, int(width / MICRO_BLOCK_WIDTH) - 1):
            idx_x = x * MICRO_BLOCK_WIDTH
            idx_y = y * MICRO_BLOCK_HEIGHT
            mv_x, mv_y = mv[y][x]
            cv2.arrowedLine(frame, (idx_x, idx_y), (idx_x + mv_x, idx_y + mv_y), (0, 255, 0), 1)
    cv2.imwrite(save_path, frame)


def main():
    # path = save_frame_files("video/sample1.mp4", 20) # create frame images from video

    b = "sample/frame/frame0002.jpg"
    b_psnr = get_psnr_from_img(b)
    print('PSNR: %s dB' % b_psnr)

    prev = "sample/frame/frame0000.jpg"
    next = "sample/frame/frame0002.jpg"

    interpolated_frame_path = "sample/result/interpolated.jpg"
    motion_vector_field_path = "sample/result/mv_field.jpg"

    mv = get_mv_bidirectional_blockmatching(prev, next)
    interpolated_img = make_frame(prev, next, mv)
    save_img_from_np_arr(interpolated_img, interpolated_frame_path)
    res_psnr = get_psnr_from_np_arr(interpolated_img)

    draw_mv_vector(interpolated_frame_path, mv, motion_vector_field_path)
    print('PSNR: %s dB' % res_psnr)


if __name__ == "__main__":
    main()
