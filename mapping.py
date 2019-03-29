import cv2
import scipy.io
import numpy as np


def mapping():
    for j in range(1, 19):
        image_data_path = '/media/newhd/Kinect Project/data/sub' + str(j) + '/'
        for i in range(1, 130):
            w = 10000 + i
            st = str(w)
            file_name = image_data_path + 'ColorFrame' + st[1:] + '.jpg'
            mat_file = image_data_path + 'DepthFrame' + st[1:] + '.mat'
            img = scipy.io.loadmat(mat_file)
            string = 'Dep' + st[1:] + '_'
            image = img[string]%256
            # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            # print(image)
            #
            (h, w) = image.shape[:2]

            center = (w / 2, h / 2)

            angle90 = -90
            scale = 1.0

            M = cv2.getRotationMatrix2D(center, angle90, scale)

            rotated90 = cv2.warpAffine(image, M, (h, w))
            if j < 9:
                rotated90 = rotated90[200:424, 100:324]
            else:
                rotated90 = rotated90[155:379, 100:324]
            n = st[1:]
            mapped_path = '/media/newhd/Kinect Project/data_mapped/sub%d/MappedFrame%s.jpg'%(j, n)
            cv2.imwrite(mapped_path, rotated90)


def main():
    mapping()


if __name__ == "__main__":
    main()
