import math
import numpy as np


def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if (abs(tZ) < 10e-6):
        print('tz = ', tZ)
    elif (norm_prev_pts.size == 0):
        print('no prev points')
    elif (norm_prev_pts.size == 0):
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(
            norm_prev_pts, norm_curr_pts, R, foe, tZ)
    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(np.array(curr_container.EM))
    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec


def normalize(pts, focal, pp):
    # transform pixels into normalized pixels using the focal length and principle point
    result = []
    for p in pts:
        result.append([(p[0] - pp[0])/focal, (p[1] - pp[1])/focal])
    return np.array(result)

def unnormalize(pts, focal, pp):
    # transform normalized pixels into pixels using the focal length and principle point
    result = []

    for p in pts:
        result.append([(p[0] * focal + pp[0]), (p[1] * focal + pp[1])])
    return np.array(result)




def decompose(EM):
    # extract R, foe and tZ from the Ego Motion
    R = EM[:3, :3]

    tx = EM[0, 3]
    ty = EM[1, 3]
    tz = EM[2, 3]

    foe = np.array([tx/tz, ty/tz])
    return R, foe, tz


def rotate(pts, R):
    # rotate the points - pts using R
    result = []
    for p in pts:
        temp = R.dot(np.array([p[0], p[1], 1]))
        result.append([temp[0]/temp[2], temp[1]/temp[2]])

    return result


# compute the epipolar line between p and foe
# run over all norm_pts_rot and find the one closest to the epipolar line
# return the closest point and its index
def find_corresponding_points(p, norm_pts_rot, foe):
    part_one = ( foe[1] - p[1] ) / ( foe[0] - p[0] )
    part_two = ( ( p[1] * foe[0] ) - ( foe[1] * p[0] ) ) / ( foe[0] - p[0] )

    mechane = math.sqrt(part_one*part_one + 1)
    closet_point = -1
    index = -1
    point = -1


    for i in range(len(norm_pts_rot)):
        distance = abs((part_one*norm_pts_rot[i][0] + part_two - norm_pts_rot[i][1]))/mechane
        if(closet_point == -1 or distance < closet_point):
            closet_point = distance
            index = i
            point = norm_pts_rot[i]
    return index, point


# calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
# calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
# combine the two estimations and return estimated Z
def calc_dist(p_curr, p_rot, foe, tZ):
    zx = (tZ * (foe[0] - p_rot[0]) )/(p_curr[0] - p_rot[0])
    zy = (tZ * (foe[1] - p_rot[1])) / (p_curr[1] - p_rot[1])

    return (zy+zx)/2