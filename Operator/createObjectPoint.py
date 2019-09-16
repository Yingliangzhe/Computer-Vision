import numpy as np

'''
the cooredinate of image plane is defined in u-v 
'''

'''
  Initialization of 3D world chessboard coordinate, because the range of 
  detected corners are in 
  [ 48 ... 12 6 0
    49 ... 13 7 1
    50 ... 14 8 2
    51 ... 15 9 3
    52 ... 16 10 4
    53 ... 17 11 5
  ]
  the first point pair is located at (8,5,0), second is (8,4,0)
  '''

def createObjectPoint(corners):
    first = 0
    fifth = 4
    threshold = 140
    last = len(corners) - 1
    u_axis = 9
    v_axis = 6


    if corners[first][0] < corners[last][0] and corners[first][1] > corners[last][1]:
        # because the corners coordinate has been reshaped to (54,2)
        if abs(corners[fifth][1] - corners[first][1]) > threshold:
            '''
            pass
           [5  11  17  23  29  35  41  47  53
            4  10  16  22  28  34  40  46  52
            3   9  15  21  27  33  39  45  51
            2   8  14  20  26  32  38  44  50
            1   7  13  19  25  31  37  43  49
            0   6  12  18  24  30  36  42  48]
            '''

            '''
            我们现在假定在像平面的 U－V　坐标，z 坐标就认为它是0，
            megrid函数把行向量[0:u_axis]复制了v_axis次，并且把[0:v_axis]复制了u_axis次。
            这里希望说明一下，这个横纵坐标，网上的教程就是数横向有几个点，纵向有几个点，我这里直接定义成u和v，
            这样更能显示出像平面特有的坐标特性。
            '''
            x, y = np.meshgrid(range(0, u_axis, 1), range(0, v_axis, 1))
            x = x.T
            y = y.T
            object_points = np.hstack((x.reshape(54, 1), y.reshape(54, 1), np.zeros((54, 1)))).astype(np.float32)
            object_points_image = object_points * 20
        else:
            '''
            pass
             [45  46  47  48  49  50  51  52  53
              36  37  38  39  40  41  42  43  44
              27  28  29  30  31  32  33  34  35
              18  19  20  21  22  23  24  25  26
               9  10  11  12  13  14  15  16  17
               0   1   2   3   4   5   6   7   8]
             '''
            x, y = np.meshgrid(range(8, -1, -1), range(5, -1, -1))
            object_points = np.hstack((x.reshape(54, 1), y.reshape(54, 1), np.zeros((54, 1)))).astype(np.float32)
            # 参见书签里python tutorial的用法，完成了按第一列排序的要求
            object_points_image = object_points[np.lexsort(-object_points[:, ::-1].T)]
            object_points_image = object_points_image[::-1]
            object_points_image = object_points_image * 20



    elif corners[first][0] > corners[last][0] and corners[first][1] > corners[last][1]:
        if abs(corners[fifth][1] - corners[first][1]) > threshold:
            '''
            pass
            [53  47  41  35  29  23  17  11  5
             52  46  40  34  28  22  16  10  4
             51  45  39  33  27  21  15   9  3
             50  44  38  32  26  20  14   8  2
             49  43  37  31  25  19  13   7  1
             48  42  36  30  24  18  12   6  0]
            '''
            object_points = np.zeros((u_axis * v_axis, 3), np.float32)
            # 用这两个指令生成一个网格，来模拟x y的坐标。生成之后的坐标如下：
            '''
            x:
            [[0 1 2 3 4 5 6]
             [0 1 2 3 4 5 6]
             [0 1 2 3 4 5 6]
             [0 1 2 3 4 5 6]
             [0 1 2 3 4 5 6]
             [0 1 2 3 4 5 6]]
            y:
            [[0 0 0 0 0 0 0]
             [1 1 1 1 1 1 1]
             [2 2 2 2 2 2 2]
             [3 3 3 3 3 3 3]
             [4 4 4 4 4 4 4]
             [5 5 5 5 5 5 5]]
            '''
            x, y = np.meshgrid(range(u_axis-1, -1, -1), range(0, v_axis, 1))
            #但是我们想让生成的y坐标也像x一样能够在reshape之后是从左到右的排序，我们引入了一个z坐标。
            #通过这个操作我们就可以得到一个像x坐标一样排列的坐标，同时又保持和原有的坐标数量一致。
            y, z = np.meshgrid(range(0,v_axis,1),range(0,u_axis,1))
            #现在我们需要把x按照行排个序
            x = x.T
            #现在我们得到了一个 8 8 8 ... 7 7 7的数组
            #
            object_points = np.hstack((x.reshape(54,1),y.reshape(54,1),np.zeros((54,1)))).astype(np.float32)
            #通过上面的指令我们完成了一个把数组给按照深度的堆叠，现在的坐标顺序满足了，opencv找到的角点的顺序
            object_points_image = object_points*20

        else:
            '''
            pass
           [53  52  51  50  49  48  47  46  45
            44  43  42  41  40  39  38  37  36
            35  34  33  32  31  30  29  28  27
            26  25  24  23  22  21  20  19  18
            17  16  15  14  13  12  11  10   9
             8   7   6   5   4   3   2   1   0]
            '''
            x, y = np.meshgrid(range(u_axis-1, -1, -1), range(0, v_axis, 1))
            object_points = np.hstack((x.reshape(54, 1), y.reshape(54, 1),np.zeros((54,1)))).astype(np.float32)
            object_points_image = object_points * 20

    elif corners[first][0] < corners[last][0] and corners[first][1] < corners[last][1]:
        if abs(corners[fifth][1] - corners[first][1]) < threshold:
            '''
            pass
            [ 0  1  2  3  4  5  6  7  8
              9 10 11 12 13 14 15 16 17
             18 19 20 21 22 23 24 25 26
             27 28 29 30 31 32 33 34 35
             36 37 38 39 40 41 42 43 44
             45 46 47 48 49 50 51 52 53 
             ]
             '''


            x, y = np.meshgrid(range(0, u_axis, 1), range(v_axis - 1, -1, -1))
            object_points = np.hstack((x.reshape(54, 1), y.reshape(54, 1), np.zeros((54, 1)))).astype(np.float32)
            object_points_image = object_points * 20

        else:
            '''
            pass
            [ 0  6 12 18 24 30 36 42 48
              1  7 13 19 25 31 37 43 49
              2  8 14 20 26 32 38 44 50
              3  9 15 21 27 33 39 45 51
              4 10 16 22 28 34 40 46 52
              5 11 17 23 29 35 41 47 53 
             ]
             '''
            x, y = np.meshgrid(range(0, u_axis, 1), range(v_axis - 1, -1, -1))
            x = x.T
            y = y.T
            object_points = np.hstack((x.reshape(54, 1), y.reshape(54, 1), np.zeros((54, 1)))).astype(np.float32)
            object_points_image = object_points * 20

    else:
        if abs(corners[fifth][1] - corners[first][1]) < threshold:
            '''
            pass
            [ 8  7  6  5  4  3  2  1  0
             17 16 15 14 13 12 11 10  9
             26 25 24 23 22 21 20 19 18
             35 34 33 32 31 30 29 28 27
             44 43 42 41 40 39 38 37 36
             53 52 51 50 49 48 47 46 45
             ]
            '''
            x, y = np.meshgrid(range(0, u_axis, 1), range(0, v_axis, 1))
            object_points = np.hstack((x.reshape(54, 1), y.reshape(54, 1))).astype(np.float32)
            object_points = object_points[::-1]
            object_points_image = object_points * 20
        else:
            '''
            pass
            [ 48 42 36 30 24 18 12  6  0
              49 43 37 31 25 19 13  7  1
              50 44 38 32 26 20 14  8  2
              51 45 39 33 27 21 15  9  3
              52 46 40 34 28 22 16 10  4
              53 47 41 35 29 23 17 11  5 
             ]
            '''
            x, y = np.meshgrid(range(0, u_axis, 1), range(v_axis - 1, -1, -1))
            object_points = np.hstack((x.reshape(54, 1), y.reshape(54, 1), np.zeros((54, 1)))).astype(np.float32)
            object_points_image = object_points[np.lexsort(-object_points[:, ::-1].T)]
            object_points_image = object_points_image * 20

    return object_points_image