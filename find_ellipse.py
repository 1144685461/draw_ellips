#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import math
import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

########################################################
# Calculate the conic factor Find the Cone Coefficient
#  input param: five points
#
# Calculation formula
# # Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
# # Bxy + Cy^2 + Dx + Ey + F = -x^2
#
# return [A, B, C, D, E, F]
########################################################
def find_cone_coe(points):
    IS_MATCH = True
    a = np.array([[p[0] * p[1], p[1] * p[1], p[0], p[1], 1] for p in points])
    # print("a = \n", a)
    b = np.array([-p[0]*p[0] for p in points])
    # print("b = \n", b)
    coe = np.linalg.solve(a, b)

    B, C, D, E, F = coe[0], coe[1], coe[2], coe[3], coe[4]

    A = [(-(B * p[0] * p[1] + C * p[1] * p[1] + D * p[0] + E * p[1] + F) / (p[0] * p[0])) for p in points]

    for a in A:
        if A[0] != a:
            IS_MATCH = False
            print("A = ", A)

    result = [A[0], B, C, D, E, F]
    # print("coe: ", result)
    # #is_match = np.allclose(np.dot(a, X), b)
    # #print("is_match:", is_match)
    return IS_MATCH, result

########################################################
# Calculate Center
#  input param:
#  coes: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
#
#  Calculation formula
#  x = (BE-2CD)/(4AC-B^2)
#  y = (BD-2AE)/(4AC-B^2)
#
#  return [x, y] # center
########################################################
def calculate_ellipse_center(coes):
    A, B, C, D, E, F = coes[0], coes[1], coes[2], coes[3], coes[4], coes[5]

    if (4*A*C - B*B == 0):
        print("no find center, 4*A*C - B*B == 0")
        exit(-1)

    x = (B*E - 2*C*D) / (4*A*C - B*B)
    y = (B*D - 2*A*E) / (4*A*C - B*B)
    return [x,y]

######################################################################################
# Calculate major and minor axis
#  input param:
#      coes:   Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
#      center point:
#
#  Calculation formula
#  a^2 = (DXc + EYc + 2F)/(A + C - ((A-C)^2 + B^2)^(1/2))
#  b^2 = (DXc + EYc + 2F)/(A + C + ((A-C)^2 + B^2)^(1/2))
#  c^  = a^2 - b^2
#
#  return [a, b, c]
#######################################################################################
def calculate_ellipse_axis(coes, center):
    A, B, C, D, E, F = coes[0], coes[1], coes[2], coes[3], coes[4], coes[5]
    Xc = center[0]
    Yc = center[1]

    a2 = (-1)*(D*Xc + E*Yc + 2*F)/(A + C - ((A-C)**2 + B**2)**0.5)
    b2 = (-1)*(D*Xc + E*Yc + 2*F)/(A + C + ((A-C)**2 + B**2)**0.5)
    c2 = a2 - b2

    if (c2 < 0):
        print("no find focus, a2 - b2 < 0")
        exit(-1)
    #print("major_axis:", major_axis, "minor_axis:", minor_axis)
    ellipse_axis = [a2**0.5, b2**0.5, c2**0.5]

    #print("ellipse[a, b, c]:", ellipse_axis)
    return ellipse_axis

########################################################
# Major axis angle
#  input param:
#      coes:   Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
#  Calculation formula
#  angle = (1/2)*arctan(B/(A-C))
#
#   return angle
########################################################
def major_axis_angle(coes):
    A, B, C, D, E, F = coes[0], coes[1], coes[2], coes[3], coes[4], coes[5]
    if (A - C == 0):
        print("no find axis_angle, A - C == 0")
        exit(-1)
    return (1/2)*math.atan(B/(A-C))

########################################################
# Eccentricity
#  Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
#  e = c/a = (a^2 - b^2)^(1/2) / a
########################################################

########################################################
# root_formula
#  input param:
#      coes: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
#      x:    point x
#
# Calculation formula
#    a, b, c = C, B * x + E, A * x * x + D * x + F
#    y1 = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
#    y2 = (-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)
#
# return [A, B, C, D, E, F]
########################################################
def root_formula(coes, x):
    has_root = False
    A, B, C, D, E, F = coes[0], coes[1], coes[2], coes[3], coes[4], coes[5]
    a, b, c = C, B * x + E, A * x * x + D * x + F

    if (b * b < 4 * a * c):
        # print("x = %d has no match y" %(x))
        return has_root, 0, 0
    else:
        has_root = True

    y1 = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    y2 = (-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)
    return has_root, y1, y2

######################################################################################
# Calculate major and minor axis
#  input param:
#      coes:   Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
#      center point: [Xc, Yc]
#      axises: [a, b, c]
#      angle:  angle about x
#      points: input five point
#
#  Calculation formula
#  a^2 = (DXc + EYc + 2F)/(A + C - ((A-C)^2 + B^2)^(1/2))
#  b^2 = (DXc + EYc + 2F)/(A + C + ((A-C)^2 + B^2)^(1/2))
#  c^  = a^2 - b^2
#
#  print save ellipse png
#######################################################################################
def draw_ellipse_points(coes, center, axises, angle, points):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.title('draw ellips as 5 points')

    # draw five points
    points_x = [point[0] for point in points]
    points_y = [point[1] for point in points]
    ax.plot(points_x, points_y, 'ro')

    # draw center point
    Cx = center[0]
    Cy = center[1]
    ax.plot(Cx, Cy, 'ro')

    # draw ellipse as x,y
    a = axises[0]
    b = axises[1]

    # ellipse point
    x_l = np.linspace(Cx - a, Cx + a, int(2*a), dtype = float)
    #print(x_l)
    max = ((points[0][0] - Cx)**2 + (points[0][1] - Cy)**2)**(1/2)
    min = max
    point_max = [0, 0]
    point_min = [0, 0]

    for x in x_l:
        has_root, y1, y2 = root_formula(coes, x)
        if (has_root == False):
            continue
        ax.plot(x, y1, 'k,')
        ax.plot(x, y2, 'k,')

        r = ((x - Cx)**2 + (y1 - Cy)**2)**(1/2)

        if (max < r):
            max = r
            point_max = [x, y1]
        if (min > r):
            min = r
            point_min = [x, y1]

    print("real a = %f, b = %f" % (max, min))
    print("point_max:", point_max)
    print("point_min:", point_min)

    # draw a axis
    a_axis_x = [center[0], point_max[0]]
    a_axis_y = [center[1], point_max[1]]
    lable_a = "a=" + str(round(max,2))
    ax.plot(a_axis_x, a_axis_y, label = lable_a)
    
    # draw b axis
    b_axis_x = [center[0], point_min[0]]
    b_axis_y = [center[1], point_min[1]]
    lable_b = "b=" + str(round(min,2))
    ax.plot(b_axis_x, b_axis_y, label = lable_b)

    plt.legend(loc='upper left')
    plt.savefig("test.png")

######################################################################################
# find the farthest points from Points
#  input param:
#      points: Enter at least 5 dots
#
#  return two points co-ordinate
#######################################################################################
def find_farthest_points(points):
    nums = len(points)
    max, l,r = -1, 0, 0

    for i  in range(nums):
        for j  in range(i + 1, nums):
            distance = (points[j][0] - points[i][0])**2 + (points[j][1] - points[i][1])**2
            if (distance > max):
                max = distance
                l,r = i,j
    return [points[l], points[r]]

######################################################################################
# Calculate lines coes
#  input param:
#   two points
#
#  Calculation formula
#  y = (y1 - y0)/(x1 - x0)*x - x0*(y1 - y0)/(x1 - x0) + y0 
#
#  return [c, a, b] cy = ax + b
#######################################################################################
def calculate_line_coes(points):
    x0, y0 = points[0][0], points[0][1]
    x1, y1 = points[1][0], points[1][1]

    if (x0 == x1):
        return [0, 1, -x0]
    else:
        a = (y1 - y0)/(x1 - x0)
        return [1, a, y0 - a*x0]

######################################################################################
# Calculate line coes
#  input param:
#   angle: angle with horizontal direction
#   point: one point in line
#
#  Calculation formula
#  y = tan(angle)(x - x0) + y0 
#
#  return [c, a, b] cy = ax + b
#######################################################################################
def calculate_line_coes(angle, point):
    x0, y0 = point[0], point[1]

    if (angle == 90):
        return [0, 1, -x0]
    else:
        a = math.tan(angle)
        return [1, a, y0 - a*x0]

######################################################################################
# Calculate line coes
#  input param:
#   angle: angle with horizontal direction
#   point: one point in line
#
#  Calculation formula
#  y = tan(angle)(x - x0) + y0 
#
#  return [c, a, b] cy = ax + b
#######################################################################################
def calculate_vertical_line_coes(f_two_points, point):
    if (f_two_points[1][0] == f_two_points[0][0]):
        angle = 90
    else:
        angle_arc = (f_two_points[1][1] - f_two_points[0][1])/(f_two_points[1][0] - f_two_points[0][0])
        angle = math.atan(angle_arc)

    x0, y0 = point[0], point[1]

    if (angle == 0):
        return [0, 1, -x0]
    else:
        a = 1/math.tan(angle)
        return [1, a, y0 - a*x0]

######################################################################################
# calculate_ellipse_and_line_cross_points
#  input param:
#   ellipse_coes: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
#   line_coes: cy = ax + b
#
#  return IS_MATCH, Points
#######################################################################################
def calculate_ellipse_and_line_cross_points(coes, line_coes):
    A, B, C, D, E, F = coes[0], coes[1], coes[2], coes[3], coes[4], coes[5]
    c, a, b = line_coes[0], line_coes[1], line_coes[2]

    # print(c, a, b)

    if (c == 0):
        x = -b/a
        has_root, y1, y2 = root_formula(coes, x)
        return has_root, [[x, y1], [x, y2]]

    a1 = c*c*A + a*B*c + a*a*C
    b1 = B*b*c + 2*a*b*C + D*c*c + E*a*c
    c1 = C*b*b + E*b*c + F*c*c

    # print(a1, b1, c1)

    if (b1 * b1 < 4 * a1 * c1):
        print("has no match x")
        return False, [[0, 0]]
    else:
        has_root = True
        x1 = (-b1 + math.sqrt(b1 * b1 - 4 * a1 * c1)) / (2 * a1)
        x2 = (-b1 - math.sqrt(b1 * b1 - 4 * a1 * c1)) / (2 * a1)
        print("x1 = ", x1)
        print("x2 = ", x2)
        has_root_x1, y1, y2 = root_formula(coes, x1)
        print(has_root_x1, y1, y2)
        has_root_x2, y3, y4 = root_formula(coes, x2)
        print(has_root_x2, y3, y4)

        if (has_root_x1 == True and has_root_x2 == True):
            return True, [[x1, y1], [x1, y2], [x2, y3], [x2, y4]]
        elif (has_root_x1 == True):
            return True, [[x1, y1], [x1, y2]]
        elif (has_root_x2 == True):
            return True, [[x2, y3], [x2, y4]]
        else:
            return False, [[0, 0]]

######################################################################################
# draw_ellipse
#  input param:
#      coes:   Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
#      center point: [Xc, Yc]
#      axises: [a, b, c]
#      angle_x:  angle about x
#      points: input five point
#
#  draw：
#      draw five points
#      draw center point
#      draw ellipse which base on center, a,b angle
#      draw a axis
#      draw b axis
#
#  Calculation formula
#  a^2 = (DXc + EYc + 2F)/(A + C - ((A-C)^2 + B^2)^(1/2))
#  b^2 = (DXc + EYc + 2F)/(A + C + ((A-C)^2 + B^2)^(1/2))
#  c^  = a^2 - b^2
#
#  print save ellipse png
#######################################################################################
def draw_ellipse(coes, center, axises, angle_x, points):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.title('draw ellips as 5 points')

    # draw five points
    points_x = [point[0] for point in points]
    points_y = [point[1] for point in points]
    ax.plot(points_x, points_y, 'ro')

    # draw center point
    Cx = center[0]
    Cy = center[1]
    ax.plot(Cx, Cy, 'ro', label = "angle = " + str(round(angle_x, 2)))
    ax.text(Cx, Cy, '({}, {})'.format(round(Cx, 2), round(Cy, 2)))
    # use center, a,b angle to draw ellipse
    a = axises[0]
    b = axises[1]

    print("a = %f b = %f" % (a, b))
    ell1 = Ellipse(xy = (Cx,Cy), width = 2*a, height = 2*b, angle = angle_x)
    ax.add_patch(ell1)
    ell1.set(alpha=0.5,
            fc='w',    # facecolor, red
            ec='black',    # edgecolor, green
            lw=3,    # line width
            ls=':',    # line style
            #label= "angle = " + str(round(angle_x, 2))
            )

    # draw a axis
    lable_a = "a=" + str(round(a,2))
    offset = a*math.cos(angle_x)*1.1
    x_l = np.linspace(Cx - offset, Cx + offset, int(2*offset*100), dtype = float)

    if (angle_x == 90):
        y = 0*x_l
    else:
        y = (math.tan(angle_x))*(x_l - Cx) + Cy
    ax.plot(x_l, y, label = lable_a)

    # draw b axis
    lable_b = "b=" + str(round(b,2))
    if (angle_x == 0):
        x_l = [0, 0]
        y = [-b*1.1, b*1.1]
    else:
        offset = b*math.sin(angle_x)*1.1
        x_l  = np.linspace(Cx - offset , Cx + offset, int(2*offset*10), dtype = float)
        y  = (1/math.tan(angle_x))*(x_l - Cx) + Cy
    ax.plot(x_l, y, label = lable_b)

    plt.legend(loc='upper left')
    plt.savefig("test.png")

######################################################################################
# draw_ellipse
#  input param:
#      points: input five point
#
#  draw：
#      draw five points
#      draw center point
#      draw ellipse which base on center, a,b angle
#      draw a axis
#      draw b axis
#
#  print save ellipse png
#######################################################################################
def draw_ellipse(points):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.title('draw ellips as 5 points')
    # draw five points
    points_x = [point[0] for point in points]
    points_y = [point[1] for point in points]
    ax.plot(points_x, points_y, 'ro')

    # find cone coes
    IS_MATCH, coes = find_cone_coe(points)
    print("IS_MATCH:", IS_MATCH)
    if (IS_MATCH == False):
        print("输入的5个点无法构建出椭圆")
        exit(-1)

    print("coes:", coes)
    center = calculate_ellipse_center(coes)
    print("center:", center)
    axises = calculate_ellipse_axis(coes, center)
    print("axises:", axises)
    angle_x = major_axis_angle(coes)
    print("angle_x:", angle_x)
    f_two_points = find_farthest_points(points)
    print("f_two_points:", f_two_points)

    # draw center point
    Cx = center[0]
    Cy = center[1]
    ax.plot(Cx, Cy, 'ro', label = "angle = " + str(round(angle_x, 2)))
    ax.text(Cx, Cy, '({}, {})'.format(round(Cx, 2), round(Cy, 2)))
    # use center, a,b angle to draw ellipse
    a = axises[0]
    b = axises[1]

    print("a = %f b = %f" % (a, b))
    ell1 = Ellipse(xy = (Cx,Cy), width = 2*a, height = 2*b, angle = angle_x)
    ax.add_patch(ell1)
    ell1.set(alpha=0.5,
            fc='w',    # facecolor, red
            ec='black',    # edgecolor, green
            lw=3,    # line width
            ls=':',    # line style
            #label= "angle = " + str(round(angle_x, 2))
            )

    # draw a axis
    lable_a = "a=" + str(round(a,2))
    offset = a*math.cos(angle_x)*1.1
    x_l = np.linspace(Cx - offset, Cx + offset, int(2*offset*100), dtype = float)

    # lcy = lax + lb
    lc, la, lb = calculate_line_coes(angle_x, center)
    if (lc == 0):
        y = 0*x_l
    else:
        y = la*x_l + lb
    ax.plot(x_l, y, label = lable_a)

    # draw b axis
    lable_b = "b=" + str(round(b,2))
    # angle_y = (90+angle_x) if angle_x < 90 else (angle_x-90)
    # lc, la, lb = calculate_line_coes(angle_y, center)
    # offset = abs(b*1.1*math.sin(angle_y))
    # y = np.linspace(Cy - offset, Cy+ offset, int(2*offset*100), dtype = float)

    # if (la == 0):
    #      x_l = 0*y_l
    # else:
    #     x_l = (lc/la)*y - lb/la
    if (angle_x == 0):
        x_l = [0, 0]
        y = [-b*1.1, b*1.1]
    else:
        offset = b*math.sin(angle_x)*1.1
        x_l  = np.linspace(Cx - offset , Cx + offset, int(2*offset*10), dtype = float)
        y  = (1/math.tan(angle_x))*(x_l - Cx) + Cy

    ax.plot(x_l, y, label = lable_b)

    # draw farthest points lines
    lable_f = "farthest"
    x = [f_two_points[0][0], f_two_points[1][0]]
    y = [f_two_points[0][1], f_two_points[1][1]]
    ax.plot(x, y, label = lable_f)

    # draw farthest points vertical lines
    Cx = (f_two_points[0][0] + f_two_points[1][0])/2
    Cy = (f_two_points[0][1] + f_two_points[1][1])/2
    ax.plot(Cx, Cy, 'ro')
    ax.text(Cx, Cy, '({}, {})'.format(round(Cx, 2), round(Cy, 2)))

    # calculate vertical line cy = ax + b
    lc, la, lb = calculate_vertical_line_coes(f_two_points, [Cx, Cy])
    print(lc, la, lb)

    # 临时测试用
    offset = 0.2
    x_l = np.linspace(Cx - offset, Cx + offset, int(2*offset*100), dtype = float)
    y = la*x_l + lb
    ax.plot(x_l, y)

    # 计算直线与椭圆交叉的
    ret, points = calculate_ellipse_and_line_cross_points(coes, [lc, la, lb])

    # 绘制交叉点
    for point in points:
        x = point[0]
        y = point[1]
        ax.plot(x, y, 'ro')
        ax.text(x, y, '({}, {})'.format(round(x, 2), round(y, 2)))

    plt.legend(loc='upper left')
    plt.savefig("test.png")

if __name__ == "__main__":
    print("find ellipse")
    points = [[181, 250], [309, 296], [455, 305], [642, 269], [669, 251]]
    draw_ellipse(points)