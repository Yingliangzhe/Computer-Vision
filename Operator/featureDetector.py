import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


affine_folder = 'corner_template_affine/'
affine_foto = [affine_folder+'BM_0.bmp',affine_folder+'BM_1.bmp',affine_folder+'LM_0.bmp',affine_folder+'LM_1.bmp',
               affine_folder+'RM_0.bmp',affine_folder+'RM_1.bmp',affine_folder+'TM_0.bmp',affine_folder+'TM_1.bmp']
affine = 1

dark_folder = 'corner_template_dark/'
dark_foto = [dark_folder+'BM_0.bmp',dark_folder+'BM_1.bmp',dark_folder+'LM_0.bmp',dark_folder+'LM_1.bmp',
             dark_folder+'RM_0.bmp',dark_folder+'RM_1.bmp',dark_folder+'TM_0.bmp',dark_folder+'TM_1.bmp']
dark = 2

reflec_folder = 'corner_template_reflec/'
reflec_foto = [reflec_folder+'BM_0.bmp',reflec_folder+'BM_1.bmp',reflec_folder+'LM_0.bmp',reflec_folder+'LM_1.bmp',
               reflec_folder+'RM_0.bmp',reflec_folder+'RM_1.bmp',reflec_folder+'TM_0.bmp',reflec_folder+'TM_1.bmp']
reflec = 3

rotation_folder = 'corner_template_rotation/'
rotation_foto = [rotation_folder+'BM_0.bmp',rotation_folder+'BM_1.bmp',rotation_folder+'LM_0.bmp',rotation_folder+'LM_1.bmp',
                 rotation_folder+'RM_0.bmp',rotation_folder+'RM_1.bmp',rotation_folder+'TM_0.bmp',rotation_folder+'TM_1.bmp']
rotation = 4

scale_folder = 'corner_template_scale/'
scale_foto = [scale_folder+'BM_0.bmp',scale_folder+'BM_1.bmp',scale_folder+'LM_0.bmp',scale_folder+'LM_1.bmp',
              scale_folder+'RM_0.bmp',scale_folder+'RM_1.bmp',scale_folder+'TM_0.bmp',scale_folder+'TM_1.bmp']
scale = 5

translation_folder = 'corner_template_translation/'
translation_foto = [translation_folder+'BM_0.bmp',translation_folder+'BM_1.bmp',translation_folder+'LM_0.bmp',translation_folder+'LM_1.bmp',
                    translation_folder+'RM_0.bmp',translation_folder+'RM_1.bmp',translation_folder+'TM_0.bmp',translation_folder+'TM_1.bmp']
translation = 6

viewpoint_folder = 'corner_template_viewpoint/'
viewpoint_foto = [viewpoint_folder+'BM_0.bmp',viewpoint_folder+'BM_1.bmp',viewpoint_folder+'LM_0.bmp',viewpoint_folder+'LM_1.bmp',
                  viewpoint_folder+'RM_0.bmp',viewpoint_folder+'RM_1.bmp',viewpoint_folder+'TM_0.bmp',viewpoint_folder+'TM_1.bmp']
viewpoint = 7

all_fotos = {'affine':affine_foto,'dark':dark_foto,'reflec':reflec_foto,'rotation':rotation_foto,'scale':scale_foto,'translation':translation_foto,'viewpoint':viewpoint_foto}

pin_folder = 'Loadport_template/'
pin_foto = [pin_folder+'BM_pin.png',pin_folder+'BM_pin_1.png',pin_folder+'TL_pin.png',pin_folder+'TL_pin_1.png',
            pin_folder+'TR_pin.png',pin_folder+'TR_pin_1.png']

all_pin_fotos = {'affine':pin_foto}
'''
LT = cv2.imread('corner_template/LT_template.bmp')
LT_gray = cv2.cvtColor(LT,cv2.COLOR_BGR2GRAY)
print(LT_gray.shape)

'''

def fotoIterator(transformation,**all_fotos):
    img_gray = []
    img_name = []
    for image_name in all_fotos[transformation]:
        img = cv2.imread(image_name)
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_gray.append(img)
        img_name.append(image_name.split('/')[-1].split('.')[-2].split('_')[-2])
    return img_gray,img_name


'''
for image in affine_foto:
    img_0 = cv2.imread(image)
    img_gray_0 = img_gray_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
    image = index(image)
    image = image + 1

    img_1 = cv2.imread(image)
    img_gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
'''

'''

img_0 = cv2.imread(affine_foto[0])
img_gray_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)

img_1 = cv2.imread(affine_foto[1])
img_gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
'''

def BRISK_descriptor(img_0, img_1, keypoints_0, keypoints_1, transformation,descriptor):
    descriptor_brisk = cv2.BRISK_create()
    _, descriptor_0 = descriptor_brisk.compute(img_0, keypoints_0)

    _, descriptor_1 = descriptor_brisk.compute(img_1, keypoints_1)

    bf = cv2.BFMatcher()
    matches = bf.match(descriptor_0, descriptor_1)

    matches = sorted(matches, key=lambda x: x.distance)

    img_match = cv2.drawMatches(img_0, keypoints_0, img_1, keypoints_1, matches[:10], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure()
    plt.imshow(img_match)
    plt.title(transformation + ':' + descriptor +'matching')
    plt.show()

def FREAK_descriptor(img_0, img_1, keypoint_0,keypoint_1,transformation,descriptor):
    descriptor_freak = cv2.xfeatures2d_FREAK.create()
    _, descriptor_0 = descriptor_freak.compute(img_0,keypoints_0)
    _, descriptor_1 = descriptor_freak.compute(img_1,keypoints_1)


    bf = cv2.BFMatcher()
    if (descriptor_0 is None or descriptor_1 is None) == False :
        matches = bf.match(descriptor_0,descriptor_1)
        matches = sorted(matches,key=lambda x: x.distance)
        img_match = cv2.drawMatches(img_0, keypoints_0, img_1, keypoints_1, matches[:10], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    else:
        img_match = np.hstack((img_0,img_1))

    plt.figure()
    plt.imshow(img_match)
    plt.title(transformation + ':' +descriptor+ 'matching')
    plt.show()

def ORB_descriptor(img_0, img_1, keypoint_0,keypoint_1,transformation,descriptor):
    descriptor_orb = cv2.ORB_create()
    _, descriptor_0 = descriptor_orb.compute(img_0,keypoints_0)
    _, descriptor_1 = descriptor_orb.compute(img_1,keypoints_1)

    bf = cv2.BFMatcher()
    matches = bf.match(descriptor_0,descriptor_1)

    matches = sorted(matches, key=lambda x: x.distance)
    img_match = cv2.drawMatches(img_0, keypoints_0, img_1, keypoints_1, matches[:10], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure()
    plt.imshow(img_match)
    plt.title(transformation + ':' + descriptor + 'matching')
    plt.show()


#TM_img_edge = cv2.imread('TM_edge.bmp')
#TM_img_edge_gray = cv2.cvtColor(TM_img_edge,cv2.COLOR_BGR2GRAY)
detector_str = input()

if detector_str == 'Harris':
    # --------------------------------------------------------------------
    # ------------------------- Harris corner ----------------------------
    # --------------------------------------------------------------------
    transformation = input()
    img,img_name = fotoIterator(transformation,**all_fotos)
    descriptor = input()

    for img_num in range(len(img)):
        if (img_num%2) == 0:
            img_0 = img[img_num]
            img_1 = img[img_num+1]

            img_0_gray = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
            img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)

            img_0_gray = np.float32(img_0_gray)
            img_1_gray = np.float32(img_1_gray)

            HL_detector = cv2.xfeatures2d_HarrisLaplaceFeatureDetector.create()

            keypoints_0 = HL_detector.detect(img_0)
            keypoints_1 = HL_detector.detect(img_1)

            img_0 = cv2.drawKeypoints(img_0, keypoints_0, (255, 0, 0),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            img_1 = cv2.drawKeypoints(img_1, keypoints_1, (255, 0, 0),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            #dst_0 = cv2.cornerHarris(img_0_gray, 3, 5, 0.04)
            #dst_1 = cv2.cornerHarris(img_1_gray, 3, 5, 0.04)

            #dst_0 = cv2.dilate(dst_0, None)
            #dst_1 = cv2.dilate(dst_1, None)

            #img_0[dst_0 > 0.01*dst_0.max()] = [255, 0, 0]
            #img_1[dst_1 > 0.01*dst_1.max()] = [255, 0, 0]


            #print(img_gray.index(img))
            plt.figure()
            #print(transformation+':'+img_name[img_gray.index(img)])
            plt.subplot(121)
            plt.imshow(img_0,cmap='gray')
            plt.title(transformation + ':' + img_name[img_num] + '0')
            plt.subplot(122)
            plt.imshow(img_1,cmap='gray')
            plt.title(transformation + ':' + img_name[img_num] + '1')
            plt.show()

            # HarrisLaplace + BRISK

            if descriptor == 'BRISK':
                BRISK_descriptor(img_0, img_1, keypoints_0, keypoints_1, transformation, descriptor)

            # HarrisLaplace + FREAK
            elif descriptor == 'FREAK':
                FREAK_descriptor(img_0, img_1, keypoints_0, keypoints_1, transformation, descriptor)

            # HarrisLaplace + ORB
            elif descriptor == 'ORB':
                ORB_descriptor(img_0, img_1, keypoints_0, keypoints_1, transformation, descriptor)

            else:
                pass




    # LT_img_gray_t = np.float32(LT_gray)
    # dst_t = cv2.cornerHarris(LT_img_gray_t,2,9,0.04)


    # LT[dst_t>0.01*dst_t.max()] = [0,0,255]





elif detector_str == 'ORB':
    # --------------------------------------------------------------------
    # ------------------------- ORB ----------------------------
    # --------------------------------------------------------------------
    transformation = input()
    img,img_name = fotoIterator(transformation,**all_fotos)
    descriptor = input()

    for img_num in range(len(img)):
        if (img_num%2) == 0:
            img_0 = img[img_num]
            img_1 = img[img_num+1]

            orb_detector = cv2.ORB_create(nfeatures=50,scaleFactor=1.1,nlevels=3,scoreType=0)
            #orb_detector = cv2.ORB_create()

            keypoints_0 = orb_detector.detect(img_0, None)
            print(len(keypoints_0))

            keypoints_1 = orb_detector.detect(img_1, None)
            print(len(keypoints_1))

            img_0 = cv2.drawKeypoints(img_0, keypoints_0, (255, 255, 0),
                                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            img_1 = cv2.drawKeypoints(img_1, keypoints_1, (255, 255, 0),
                                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


            #print(img_gray.index(img))
            plt.figure()
            #print(transformation+':'+img_name[img_gray.index(img)])
            plt.subplot(121)
            plt.imshow(img_0,cmap='gray')
            plt.title(transformation + ':' + img_name[img_num] + '0')
            plt.subplot(122)
            plt.imshow(img_1,cmap='gray')
            plt.title(transformation + ':' + img_name[img_num] + '1')
            plt.show()




elif detector_str == 'SURF':
    # --------------------------------------------------------------------
    # ------------------------- SURF ----------------------------
    # --------------------------------------------------------------------

    transformation = input()
    img,img_name = fotoIterator(transformation,**all_pin_fotos)
    descriptor = input()

    for img_num in range(len(img)):
        if (img_num%2) == 0:
            img_0 = img[img_num]
            img_1 = img[img_num+1]

            surf = cv2.xfeatures2d.SURF_create(300)

            keypoints_0 = surf.detect(img_0, None)
            print(len(keypoints_0))

            keypoints_1 = surf.detect(img_1, None)
            print(len(keypoints_1))


            img_0 = cv2.drawKeypoints(img_0, keypoints_0, None, (255, 0, 0),
                                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            img_1 = cv2.drawKeypoints(img_1, keypoints_1, None, (255, 0, 0),
                                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


            #print(img_gray.index(img))
            plt.figure()
            #print(transformation+':'+img_name[img_gray.index(img)])
            plt.subplot(121)
            plt.imshow(img_0,cmap='gray')
            plt.title(transformation + ':' + img_name[img_num] + '0')
            plt.subplot(122)
            plt.imshow(img_1,cmap='gray')
            plt.title(transformation + ':' + img_name[img_num] + '1')
            plt.show()
            #descriptor = input()

            # SURF + BRISK

            if descriptor == 'BRISK':
                BRISK_descriptor(img_0, img_1, keypoints_0,keypoints_1,transformation,descriptor)

            # SURF + FREAK
            elif descriptor == 'FREAK':
                FREAK_descriptor(img_0, img_1, keypoints_0,keypoints_1,transformation,descriptor)

            # SURF + ORB
            elif descriptor == 'ORB':
                ORB_descriptor(img_0, img_1, keypoints_0,keypoints_1,transformation,descriptor)

            else:
                pass

elif detector_str == 'SIFT':
    transformation = input()
    img, img_name = fotoIterator(transformation, **all_fotos)
    descriptor = input()

    for img_num in range(len(img)):
        if (img_num % 2) == 0:
            img_0 = img[img_num]
            img_1 = img[img_num + 1]

            sift = cv2.xfeatures2d.SIFT_create()

            keypoints_0 = sift.detect(img_0, None)
            print(len(keypoints_0))

            keypoints_1 = sift.detect(img_1, None)
            print(len(keypoints_1))

            img_0 = cv2.drawKeypoints(img_0, keypoints_0, None, (255, 0, 0),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            img_1 = cv2.drawKeypoints(img_1, keypoints_1, None, (255, 0, 0),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # print(img_gray.index(img))
            plt.figure()
            # print(transformation+':'+img_name[img_gray.index(img)])
            plt.subplot(121)
            plt.imshow(img_0, cmap='gray')
            plt.title(transformation + ':' + img_name[img_num] + '0')
            plt.subplot(122)
            plt.imshow(img_1, cmap='gray')
            plt.title(transformation + ':' + img_name[img_num] + '1')
            plt.show()

'''
# Hough Transformation Probabilistic
min_line_length = 100
max_line_gap = 5
lines = cv2.HoughLinesP(TM_img_edge_gray,1,np.pi/180,60,min_line_length,max_line_gap)
print(lines.shape)

spec_line = []
for i in range(lines.shape[0]):
	for x1, y1, x2, y2 in lines[i]:
		k = (y1 - y2)/(x1 - x2)
		angle = abs(math.atan(k)*180/math.pi)
		#if angle > 40 and angle < 50:
			#spec_line = spec_line.append(lines[i])
		cv2.line(TM_img_edge, (x1, y1), (x2, y2), (0, 255, 0), 2)

'''
# LSD line detector




while(1):
    cv2.imshow('img_0',img_0_kp)
    cv2.imshow('img_1',img_1_kp)
    #cv2.imshow('LT',LT)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # hit escape to quit
        break