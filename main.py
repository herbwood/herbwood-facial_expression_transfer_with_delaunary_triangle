import cv2
import numpy as np
import random
import time 
import argparse

def draw_point(img, p) :
    cv2.circle( img, p, 2, (0, 0, 255), cv2.FILLED, cv2.LINE_AA, 0 )

# Check if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

# Draw delaunay triangles
def get_delaunay(img, subdiv, original_color) :
    
    total_tri_list = []

    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])
    
    total_tri_list = [[] for _ in range(len(triangleList))]
    
    for i, t in enumerate(triangleList):
        print(i, t)
            
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        
        total_tri_list[i].append(np.float32([[list(map(int, pt1)), list(map(int, pt2)), list(map(int, pt3))]]))
        
    return total_tri_list

def draw_delaunay(img, triangleList, delaunay_color ) :
    
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList :
        
        pt1 = (t[0][0][0], t[0][0][1])
        pt2 = (t[0][1][0], t[0][1][1])
        pt3 = (t[0][2][0], t[0][2][1])
        
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
        
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)

def warpTriangle(img1, img2, tri1, tri2):
    
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(tri1) # (x, y, w, h)
    r2 = cv2.boundingRect(tri2)
    
    # Offset points by left top corner of the respective rectangles
    tri1Cropped = []
    tri2Cropped = []
    
    for i in range(0, 3):
        tri1Cropped.append(((tri1[0][i][0] - r1[0]),(tri1[0][i][1] - r1[1])))
        tri2Cropped.append(((tri2[0][i][0] - r2[0]),(tri2[0][i][1] - r2[1])))

    # Crop input image
    img1Cropped = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(tri1Cropped), np.float32(tri2Cropped) )
    
    # Apply the Affine Transform just found to the src image
    img2Cropped = cv2.warpAffine( img1Cropped, warpMat, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0);

    img2Cropped = img2Cropped * mask
    
    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
    
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Cropped

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--emotion", default='sad', type=str,
                        help="target emotion to transfer. choies=[happy, sad]")
    
    args = parser.parse_args()
    
    
    # Define colors for drawing.
    delaunay_color = (255,255,255)
    points_color = (0, 0, 255)

    # Read in the image.
    img = cv2.imread("obama.jpg");
    
    h, w, _ = img.shape
    print(h, w)
    
    # Keep a copy around
    img_orig = img.copy();
    
    # Rectangle to be used with Subdiv2D
    size = img.shape
    rect = (0, 0, size[1], size[0])
    
    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect);

    # Create an array of points.
    points = [];
    
    # Read in the points from a text file
    with open("obama.txt") as file :
        for line in file :
            x, y = line.split()
            points.append((int(x), int(y)))

    # add boundary points 
    points = points + [(0,0), (int(w/2),0), (int(w-1),0), (w-1, int(h/2)), ( w-1, h-1 ), (int(w/2), h-1 ), (0, h-1), (0, int(h/2))]
    
    # Insert points into subdiv
    for p in points :
        subdiv.insert(p)
    
    total_tri_list = get_delaunay( img, subdiv, original_color=delaunay_color);
    
    
    
    
    if args.emotion == "sad":
        # left eyebrow 
        # le = -17 # H / 41
        le = -1 * int(h / 41)
        
        # right eyebrow 
        # re = -17 # H / 41
        # re2 = -16 # H / 44 
        re = -1 * int(h / 41)
        re2 = -1 * int(h / 44)
        
        # left lip 
        # ll = 16 # H / 44
        # ll2 = 13 # H / 55
        ll = int(h / 44)
        ll2 = int(h / 55)
        
        # right lip  
        # rl = 14 # H / 50 
        rl = int(h / 50)
        
        # tri[0][1][1] += le 
        target_dict = {
                        # left eyebrow
                    70 : np.float32([[[256, 0], [270, 219+le], [250, 219]]]), # 70 [256.   0. 270. 219. 250. 219.]
                    73 : np.float32([[[250, 219], [270, 219+le],[266, 239]]]), # 73 [250. 219. 270. 219. 266. 239.]
                    76 : np.float32([[[270, 219+le], [289, 223], [266, 239]]]), # 76 270. 219. 289. 223. 266. 239.]
                    
                        # right eyebrow
                    69 : np.float32([[[270, 219+re], [256, 0], [339, 216+re2]]]), # 69 [270. 219. 256.   0. 339. 216.]
                    81 : np.float32([[[339, 216+re2], [256, 0], [358, 215]]]), # 81 [339. 216. 256.   0. 358. 215.]
                    80 : np.float32([[[320, 222], [339, 216+re2], [343, 236]]]),# 80 [320. 222. 339. 216. 343. 236.]
                    82 : np.float32([[[343, 236], [339, 216+re2], [358, 215]]]),# 82 [343. 236. 339. 216. 358. 215.]
                    77 : np.float32([[[289, 223], [270, 219+re], [339, 216+re2]]]), # 77 [289. 223. 270. 219. 339. 216.]
                    72 : np.float32([[[320, 222], [289, 223], [339, 216+re2]]]), # 72 [320. 222. 289. 223. 339. 216.]
                    
                    # left lip 
                    17 : np.float32([[[279, 356], [243, 373], [263, 346+ll]]]), # 17 [279. 356. 243. 373. 263. 346.] 
                    24 : np.float32([[[243, 373], [229, 349], [263, 346+ll]]]), # 24 [243. 373. 229. 349. 263. 346.]
                    115 : np.float32([[[263, 346+ll], [278, 341], [270, 347+ll2]]]), # 115 [263. 346. 278. 341. 270. 347.]
                    133 : np.float32([[[279, 356], [263, 346+ll], [270, 347+ll2]]]), # 133 [279. 356. 263. 346. 270. 347.]
                    131 : np.float32([[[278, 341], [279, 356], [270, 347+ll2]]]), # 131 [278. 341. 279. 356. 270. 347.]
                    20 : np.float32([[[229, 349], [220, 322], [263, 346+ll]]]), # 20 [229. 349. 220. 322. 263. 346.]
                    99 : np.float32([[[263, 346+ll], [220, 322], [281, 311]]]), # 99 [263. 346. 220. 322. 281. 311.]
                    114 : np.float32([[[263, 346+ll], [281, 311], [278, 341]]]), # 114 [263. 346. 281. 311. 278. 341.]
                    
                    # right lip 
                    123 : np.float32([[[332, 353], [348, 342+rl], [374, 367]]]), # 123 [332. 353. 348. 342. 374. 367.]
                    49 : np.float32([[[388, 344], [374, 367], [348, 342+rl]]]), # 49 [388. 344. 374. 367. 348. 342.]
                    53 : np.float32([[[397, 319], [388, 344], [348, 342+rl]]]), # 53 [397. 319. 388. 344. 348. 342.]
                    91 : np.float32([[[326, 307], [397, 319], [348, 342+rl]]]), # 91 [326. 307. 397. 319. 348. 342.]
                    124 : np.float32([[[348, 342+rl], [332, 353], [342, 343]]]), # 124 [348. 342. 332. 353. 342. 343.]
                    125 : np.float32([[[331, 338], [348, 342+rl], [342, 343]]]), # 125 [331. 338. 348. 342. 342. 343.]
                    126 : np.float32([[[332, 353], [331, 338], [342, 343]]]), # 126 [332. 353. 331. 338. 342. 343.]
                    22 : np.float32([[[331, 338], [326, 307], [348, 342+rl]]]), # 22 [331. 338. 326. 307. 348. 342.]
                    
                    }
        
    elif args.emotion == "happy":
        # left eyebrow 
        # le = -1 * int(h / 41)
        le = -1 * 10
        
        # right eyebrow 
        re = -10 # H / 41
        re2 = -16 # H / 44 
        # re = -1 * int(h / 41)
        # re2 = -1 * int(h / 44)
        
        # left eye 
        leye = -8
        leye2 = -8
        
        # right eye 
        reye = -8
        reye2 = -8
        
        # left lip 
        ll = -13 # H / 44
        ll2 = -10 # H / 55
        # ll = int(h / 44)
        # ll2 = int(h / 55)
        
        # right lip  
        rl = -14 # H / 50 
        rl2 = -10 # H / 50 
        # rl = int(h / 50)
        
        # tri[0][1][1] += le 
        target_dict = {
                        # left eyebrow
                    66 : np.float32([[[232, 225], [256, 0], [250, 219+le]]]), # 66 [232. 225. 256.   0. 250. 219.]
                    70 : np.float32([[[256,  0], [270, 219], [250, 219+le]]]), # 70 [256.   0. 270. 219. 250. 219.]
                    0 : np.float32([[[250, 219+le], [254, 240], [232, 225]]]), # 0 [250. 219. 254. 240. 232. 225.]
                    1 : np.float32([[[254, 240], [250, 219+le], [266, 239]]]), # 1 [254. 240. 250. 219. 266. 239.]
                    73 : np.float32([[[250, 219+le], [270, 219], [266, 239]]]), # 73 [250. 219. 270. 219. 266. 239.]
                    
                    # right eyebrow
                    81 : np.float32([[[339, 216], [256,  0], [358, 215+re]]]), # 81 [339. 216. 256.   0. 358. 215.]
                    83 : np.float32([[[358, 215+re], [256,  0], [511, 0]]]), # 83 [358. 215. 256.   0. 511.   0.]
                    86 : np.float32([[[375, 220], [358, 215+re], [511, 0]]]), # 86 [375. 220. 358. 215. 511.   0.]
                    82 : np.float32([[[343, 236], [339, 216], [358, 215+re]]]), # 82 [343. 236. 339. 216. 358. 215.]
                    111 : np.float32([[[358, 215+re], [356, 236], [343, 236]]]), # 111 [358. 215. 356. 236. 343. 236.]
                    85 : np.float32([[[375, 220], [356, 236], [358, 215+re]]]), # 85 [375. 220. 356. 236. 358. 215.]
                    
                    # left eye 
                    107 : np.float32([[[243, 247], [254, 240], [254, 249+leye]]]), # 107 [243. 247. 254. 240. 254. 249.]
                    106 : np.float32([[[254, 240], [266, 247+leye2], [254, 249+leye]]]), # 106 [254. 240. 266. 247. 254. 249.]
                    41 : np.float32([[[254, 249+leye], [214, 297], [243, 247]]]), # 41 [254. 249. 214. 297. 243. 247.]
                    109 : np.float32([[[266, 247+leye2], [281, 311], [254, 249+leye]]]), # 109 [266. 247. 281. 311. 254. 249.]
                    104 : np.float32([[[266, 239], [266, 247+leye2], [254, 240]]]), # 104 [266. 239. 266. 247. 254. 240.]
                    105 : np.float32([[[266, 247+leye2], [266, 239], [276, 245]]]), # 105 [266. 247. 266. 239. 276. 245.]
                    42 : np.float32([[[214, 297], [254, 249+leye], [281, 311]]]), # 42 [214. 297. 254. 249. 281. 311.]
                    57 : np.float32([[[304, 277], [266, 247+leye2], [276, 245]]]), # 57 [304. 277. 266. 247. 276. 245.]
                    25 : np.float32([[[304, 277], [281, 311], [266, 247+leye2]]]), # 25 [304. 277. 281. 311. 266. 247.]
                    
                    # right eye 
                    112 : np.float32([[[332, 243], [343, 236], [344, 245+reye]]]), # 112 [332. 243. 343. 236. 344. 245.]
                    65 : np.float32([[[356, 236], [344, 245+reye], [343, 236]]]), # 65 [356. 236. 344. 245. 343. 236.]
                    61 : np.float32([[[356, 236], [356, 245+reye2], [344, 245+reye]]]), # 61 [356. 236. 356. 245. 344. 245.]
                    62 : np.float32([[[356, 245+reye2], [356, 236], [367, 242]]]), # 62 [356. 245. 356. 236. 367. 242.]
                    68 : np.float32([[[344, 245+reye], [304, 277], [332, 243]]]), # 68 [344. 245. 304. 277. 332. 243.]
                    67 : np.float32([[[304, 277], [344, 245+reye], [326, 307]]]), # 67 [304. 277. 344. 245. 326. 307.]
                    113 : np.float32([[[356, 245+reye2], [326, 307], [344, 245+reye]]]), # 113 [356. 245. 326. 307. 344. 245.]
                    103 : np.float32([[[403, 291], [326, 307], [356, 245+reye2]]]), # 103 [403. 291. 326. 307. 356. 245.]
                    110 : np.float32([[[403, 291], [356, 245+reye2], [367, 242]]]), # 110 [403. 291. 356. 245. 367. 242.]
                    
                    # left lip 
                    99 : np.float32([[[263, 346+ll], [220, 322], [281, 311]]]), # 99 [263. 346. 220. 322. 281. 311.]
                    20 : np.float32([[[229, 349], [220, 322], [263, 346+ll]]]), # 20 [229. 349. 220. 322. 263. 346.]
                    24 : np.float32([[[243, 373], [229, 349], [263, 346+ll]]]), # 24 [243. 373. 229. 349. 263. 346.]
                    17 : np.float32([[[279, 356], [243, 373], [263, 346+ll]]]), # 17 [279. 356. 243. 373. 263. 346.]
                    115 : np.float32([[[263, 346+ll], [278, 341], [270, 347+ll2]]]), # 115 [263. 346. 278. 341. 270. 347.]
                    133 : np.float32([[[279, 356], [263, 346+ll], [270, 347+ll2]]]), # 133 [279. 356. 263. 346. 270. 347.]
                    131 : np.float32([[[278, 341], [279, 356], [270, 347+ll2]]]), # 131 [278. 341. 279. 356. 270. 347.]
                    114 : np.float32([[[263, 346+ll], [281, 311], [278, 341]]]), # 114 [263. 346. 281. 311. 278. 341.]
                    
                    # right lip 
                    91 : np.float32([[[326, 307], [397, 319], [348, 342+rl]]]), # 91 [326. 307. 397. 319. 348. 342.]
                    53 : np.float32([[[397, 319], [388, 344], [348, 342+rl]]]), # 53 [397. 319. 388. 344. 348. 342.]
                    49 : np.float32([[[388, 344], [374, 367], [348, 342+rl]]]), # 49 [388. 344. 374. 367. 348. 342.]
                    123 : np.float32([[[332, 353], [348, 342+rl], [374, 367]]]), # 123 [332. 353. 348. 342. 374. 367.]
                    124 : np.float32([[[348, 342+rl], [332, 353], [342, 343+rl2]]]), # 124 [348. 342. 332. 353. 342. 343.]
                    125 : np.float32([[[331, 338], [348, 342+rl], [342, 343+rl2]]]), # 125 [331. 338. 348. 342. 342. 343.]
                    126 : np.float32([[[332, 353], [331, 338], [342, 343+rl2]]]), # 126 [332. 353. 331. 338. 342. 343.]
                    22 : np.float32([[[331, 338], [326, 307], [348, 342+rl]]]), # 22 [331. 338. 326. 307. 348. 342.]
                    }
        
    else:
        print("Choose the emotion choices in [happy, sad]")
    
    
    # imgOut = 255 * np.ones(img.shape, dtype = img.dtype)
    imgOut = 0 * np.ones(img.shape, dtype = img.dtype)
    
    postImp_list = []
    drawList = []
    
    # 697 512
    for i, tri in enumerate(total_tri_list):
        
        if i in list(target_dict.keys()):
            warpTriangle(img, imgOut, tri[0], target_dict[i])
            drawList.append(target_dict[i])
            
        else:
            warpTriangle(img, imgOut, tri[0], tri[0])
            drawList.append(tri[0])
            
    triOutput = imgOut.copy()
            
    draw_delaunay(triOutput, drawList, (255, 255, 255))
            
    cv2.imshow("Input", img)
    cv2.imshow("Triangle Output", triOutput)
    cv2.imshow(f"Output : {args.emotion}", imgOut)
    # cv2.imwrite(f"sad_{time.time()}.jpg", imgOut)
    cv2.waitKey(0)
            