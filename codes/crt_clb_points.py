import pickle


POINT_RATIO = 0.012
# parameters of get_clb_points function
POINTS_CLB_WIN_ROW = 3
POINTS_CLB_WIN_COL = 3
POINTS_N_SMP_IN_PNT = 20
# parameters of get_clb_lines function
LINE_CLB_WIN_ROW = 10
LINE_TIME = 5
LINE_FRAME_RATE = 30


def get_clb_points(clb_win_row, clb_win_col, n_smp_in_pnt):
    points = []
    dy = (1 - clb_win_row * POINT_RATIO) / (clb_win_row - 1)
    dx = (1 - clb_win_col * POINT_RATIO) / (clb_win_col - 1)

    for j in range(clb_win_row):
        p_y = j * (POINT_RATIO + dy) + POINT_RATIO / 2
        for i in range(clb_win_col):
            p_x = i * (POINT_RATIO + dx) + POINT_RATIO / 2
            smp_in_p = []
            for k in range(n_smp_in_pnt):
                smp_in_p.append([p_x, p_y])
            points.append(smp_in_p)

    with open(f"../files/calibration_points_{clb_win_row}x{clb_win_col}x{n_smp_in_pnt}.pickle", 'wb') as f:
        pickle.dump(points, f)
    return 0


def get_clb_lines(clb_win_row, line_time, frame_rate):
    clb_win_col = line_time * frame_rate
    points = []
    dy = (1 - clb_win_row * POINT_RATIO) / (clb_win_row - 1)
    dx = (1 - clb_win_col * POINT_RATIO) / (clb_win_col - 1)

    for j in range(clb_win_row):
        p_y = j * (POINT_RATIO + dy) + POINT_RATIO / 2
        smp_in_p = []
        for i in range(clb_win_col):
            p_x = i * (POINT_RATIO + dx) + POINT_RATIO / 2
            smp_in_p.append([p_x, p_y])
        points.append(smp_in_p)

    with open(f"../files/calibration_points_{clb_win_row}x{clb_win_col}x1.pickle", 'wb') as f:
        pickle.dump(points, f)
    return 0


get_clb_points(POINTS_CLB_WIN_ROW, POINTS_CLB_WIN_COL, POINTS_N_SMP_IN_PNT)
get_clb_lines(LINE_CLB_WIN_ROW, LINE_TIME, LINE_FRAME_RATE)
