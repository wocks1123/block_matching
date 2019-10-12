import argparse

from BlockMatching import get_mv_bidirectional_blockmatching, get_psnr_from_img, get_psnr_from_np_arr, make_frame\
                            , draw_mv_vector, save_img_from_np_arr, MICRO_BLOCK_WIDTH, MICRO_BLOCK_HEIGHT\
                            , OVERLAPPED_WIDTH, OVERLAPPED_HEIGHT


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prev_img", required=True)
    parser.add_argument("--next_img", required=True)
    parser.add_argument("--real_frame", required=True)
    parser.add_argument("--micro_block_width", required=True)
    parser.add_argument("--micro_block_height", required=True)
    parser.add_argument("--overlapped_block_width", required=True)
    parser.add_argument("--overlapped_block_height", required=True)
    parser.add_argument("--interpolated_frame_path", required=True)
    parser.add_argument("--motion_vector_field_path", required=True)

    args = parser.parse_args()

    #TODO
    # 여기서 값을 할당해도 변경되지 않음...?
    MICRO_BLOCK_WIDTH   = args.micro_block_width
    MICRO_BLOCK_HEIGHT  = args.micro_block_height
    OVERLAPPED_WIDTH    = args.overlapped_block_width
    OVERLAPPED_HEIGHT   = args.overlapped_block_height

    interpolated_frame_path     = args.interpolated_frame_path
    motion_vector_field_path    = args.motion_vector_field_path

    b = args.real_frame
    b_psnr = get_psnr_from_img(b)
    print("real frame psnr : %s dB" % b_psnr)

    prev = args.prev_img
    next = args.next_img
    mv = get_mv_bidirectional_blockmatching(prev, next)
    interpolated_img = make_frame(prev, next, mv)
    save_img_from_np_arr(interpolated_img, interpolated_frame_path)

    res_psnr = get_psnr_from_np_arr(interpolated_img)

    draw_mv_vector(interpolated_frame_path, mv, motion_vector_field_path)
    print("interpolated frame psnr : %s dB" % res_psnr)


if __name__ == "__main__":
    main()
