#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import math
import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def draw_points(points):
    print(points)
    points_x = [point[0] for point in points]
    points_y = [point[1] for point in points]

    plt.xlim(xmax=1000, xmin=0)
    plt.ylim(ymax=1000, ymin=0)

    #plt.title("任意五个点")
    plt.plot(points_x, points_y, 'ro')
    # #plt.show()
    plt.savefig("test.png")

# Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
def draw_ellipse(center, axises):
    print(center)
    print(axises)
    # img = data.chelsea()
    # #rr,cc = draw.ellipse(center[1], center[0], axises[1], axises[0])
    # rr,cc = draw.ellipse(300, 300, 60, 120)
    # print("rr:", rr)
    # print("cc", cc)
    # draw.set_color(img, [rr, cc], [0, 255, 255])
    # plt.imshow(img, plt.cm.gray)
    # plt.show()
    # plt.savefig("test.png")
    # 创建画布
    # fig = plt.figure(figsize=(12, 8), facecolor='beige')
    # axes = fig.subplots(nrows=1, ncols=1, subplot_kw={'aspect': 'equal'})
    # ax = axes[0, 0]
    # ellipse = Ellipse(xy=(center[0],center[1]), width=axises[0], height=axises[1], angle=0)
    # plt.plot(p=ellipse)
    # ellipse.set(alpha=0.4, color='lightskyblue')
    # plt.savefig("test.png")

def test_solve():
    A = np.array([[1, 1, 1], [0, 2, 5], [2, 5, -1]])
    print("系数矩阵:\n", A)
    b = np.array([[6], [-4], [27]])
    print("矩阵b:\n", b)
    X = lg.solve(A, b)
    print("X=\n", X)
    is_match = np.allclose(np.dot(A, X), b)
    print("is_match:", is_match)

########################################################
# # Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
# # Bxy + Cy^2 + Dx + Ey + F = -x^2

########################################################
def find_ellipse(p1, p2, p3, p4, p5):
    x1, y1, x2, y2, x3, y3, x4, y4, x5, y5 = p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1], p5[0], p5[1]

    a = np.array([
        [x1 * y1, y1 * y1, x1, y1, 1],
        [x2 * y2, y2 * y2, x2, y2, 1],
        [x3 * y3, y3 * y3, x3, y3, 1],
        [x4 * y4, y4 * y4, x4, y4, 1],
        [x5 * y5, y5 * y5, x5, y5, 1],
    ])

    print("a = \n", a)
    b = np.array([-x1*x1, -x2*x2, -x3*x3, -x4*x4, -x5*x5])
    print("b = \n", b)

    x = np.linalg.solve(a, b)

    B, C, D, E, F = x[0], x[1], x[2], x[3], x[4]

    A = -(B * x1 * y1 + C * y1 * y1 + D * x1 + E * y1 + F) / (x1 * x1)

    print("coe = ", A)
    print(" xx :", A, B, C, D, E, F)

    #is_match = np.allclose(np.dot(a, X), b)
    #print("is_match:", is_match)

    return [A, B, C, D, E, F]

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

if __name__ == "__main__":
    print("find ellipse")
    #test_solve()
    points = [[181, 250], [309, 296], [455, 305], [642, 269], [669, 251]]
    IS_MATCH, coes = find_cone_coe(points)
    print("IS_MATCH:", IS_MATCH)
    if (IS_MATCH == False):
        print("输入的5个点无法构建出椭圆")
        exit(-1)
    print("coes:", coes)
    center = calculate_ellipse_center(coes)
    print("center:", center)
    axis = calculate_ellipse_axis(coes, center)
    print("axis:", axis)
    x_angle = major_axis_angle(coes)
    print("x_angle:", x_angle)
    draw_ellipse(coes, center, axis, x_angle, points)
    # draw_ellipse_points(coes, center, axis, x_angle, points)
