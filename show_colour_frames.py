import cv2

def show_image():
    for j in range(1, 19):
        image_data_path = '/media/newhd/Kinect Project_new/data/sub'+ str(j) + '/'
        for i in range(1, 130):
            w = 10000 + i
            st = str(w)

            file_name = image_data_path + 'ColorFrame' + st[1:] + '.jpg'
            print(file_name)
            img = cv2.imread(file_name)
            if img is None:
                continue
            else:
                crop_img = img[450:800, 850:1500]
                cv2.imshow("Cropped", crop_img)
                cv2.waitKey(10)
        cv2.destroyAllWindows()

def main():
    show_image()

if __name__ == '__main__':
    main()