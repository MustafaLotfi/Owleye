import pickle


# parameters of get_clb_points function
points_clb_win_row = 3
points_clb_win_col = 3
smp_in_pnt = 20
# parameters of get_clb_lines function
line_clb_win_row = 10
line_time = 5
line_frame_rate = 30


def get_clb_points(clb_win_row, clb_win_col, n_smp_in_pnt):
    point_ratio = 0.012
    points = []
    dy = (1 - clb_win_row * point_ratio) / (clb_win_row - 1)
    dx = (1 - clb_win_col * point_ratio) / (clb_win_col - 1)

    for j in range(clb_win_row):
        p_y = j * (point_ratio + dy) + point_ratio / 2
        for i in range(clb_win_col):
            p_x = i * (point_ratio + dx) + point_ratio / 2
            smp_in_p = []
            for k in range(n_smp_in_pnt):
                smp_in_p.append([p_x, p_y])
            points.append(smp_in_p)

    with open(f"../files/{clb_win_row}x{clb_win_col}x{n_smp_in_pnt}.pickle", 'wb') as f:
        pickle.dump(points, f)


def get_clb_lines(clb_win_row, line_time, frame_rate):
    point_ratio = 0.012
    clb_win_col = line_time * frame_rate
    points = []
    dy = (1 - clb_win_row * point_ratio) / (clb_win_row - 1)
    dx = (1 - clb_win_col * point_ratio) / (clb_win_col - 1)

    for j in range(clb_win_row):
        p_y = j * (point_ratio + dy) + point_ratio / 2
        smp_in_p = []
        for i in range(clb_win_col):
            p_x = i * (point_ratio + dx) + point_ratio / 2
            smp_in_p.append([p_x, p_y])
        points.append(smp_in_p)

    with open(f"../files/{clb_win_row}x{clb_win_col}x1.pickle", 'wb') as f:
        pickle.dump(points, f)


get_clb_points(points_clb_win_row, points_clb_win_col, smp_in_pnt)
get_clb_lines(line_clb_win_row, line_time, line_frame_rate)
