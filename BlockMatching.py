import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
from datetime import datetime
import sys


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
        micro_block_width   = 10
        micro_block_height  = 10

        # 현재 이미지를 micro block으로 나눈다. 나누어진 블럭 리스트를 가져오기
        micro_block_list = []
        # 각 micro block 좌표
        micro_block_idx_list = []

        for w_idx in range(0, width, micro_block_width):
            micro_block_list_row = []
            micro_block_idx_list_row = []
            for h_idx in range(0, height, micro_block_height):
                # print(w_idx, h_idx, w_idx + micro_block_width, h_idx + micro_block_height)
                # print(curr_frame[h_idx: h_idx + micro_block_height, w_idx: w_idx + micro_block_width].shape)
                micro_block_list_row.append(curr_frame[h_idx: h_idx + micro_block_height, w_idx: w_idx + micro_block_width])
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
        for w in range(len(micro_block_idx_list)):
            motion_vector_list_row = []
            for h in range(len(micro_block_idx_list[0])):
                # search range 계산
                micro_x, micro_y = micro_block_idx_list[w][h]
                micro_block = micro_block_list[w][h]

                search_range = get_search_range(0, 0, width, height, micro_x, micro_y, micro_block_width, micro_block_height)
                start_w, start_h, end_w, end_h = search_range

                mv = (0, 0)
                for x in range(start_w, end_w):
                    min_value = sys.maxsize
                    for y in range(start_h, end_h):
                        # search range 내에서 모션벡터 계산
                        window_block = next_frame[y:y + micro_block_height, x:x + micro_block_width]
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


if __name__ == "__main__":
    # path = save_front_frame("video/sample2.mp4", 20)
    pred = block_matching("frame/sample2_2019-10-06_21_17_10_20_estim", True)
    base = block_matching("frame/sample2_2019-10-06_21_17_10_20_base", False)
    print(base)
    print(pred)
    print(base.shape)
    print(pred.shape)
    print(base - pred)

    # TODO
    # 각 프레임 사이의 모션벡터에 대해서
    # 1. PSNR 계산
    # 2. Motion vector field 출력

