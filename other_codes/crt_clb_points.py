import pickle


METHOD = 0
POINT_RATIO = 0.012


def get_clb_points():
    clb_win_col = 7
    clb_win_row = 5
    n_smp_in_pnt = 10
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


def get_clb_lines():
    line_time = 2
    frame_rate = 10
    clb_win_col = line_time * frame_rate
    clb_win_row = 3
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


if METHOD == 0:
    calibration_points = get_clb_points()
else:
    calibration_points = get_clb_lines()
