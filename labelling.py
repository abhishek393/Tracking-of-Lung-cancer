import scipy.io
import numpy as np
import csv

def labelling():
    dict = []
    for j in range(1, 19):
        depth_path = '/home/akshit/Desktop/KinectProject_new/data/sub' + str(j) + '/DepthFrame'
        d1 = depth_path + '0004.mat'
        d2 = depth_path + '0035.mat'

        depth_1 = scipy.io.loadmat(d1)
        depth_val_1 = depth_1['Dep0004_']
        depth_val_1 = depth_val_1[325:332, 212:219]
        depth_2 = scipy.io.loadmat(d2)
        depth_val_2 = depth_2['Dep0035_']
        depth_val_2 = depth_val_2[325:332, 212:219]

        difference_matrix = []
        for i, k in np.nditer([depth_val_1 , depth_val_2]):
            if abs(i-k) > 60000:
                     difference_matrix.append(65536 - abs(i-k))
            else:
                difference_matrix.append(abs(i-k))

        count_heavy = 0
        count_normal = 0
        count_low = 0

        for m in difference_matrix:
            if m >= 10:
                count_heavy += 1
            if m in range(5, 10):
                count_normal +=1
            if m < 5:
                count_low += 1

        if count_heavy >= count_normal and count_heavy > count_low:
                label = 2
        elif count_normal > count_heavy and count_normal > count_low:
                label = 1
        elif count_heavy == count_low:
                label = 1
        else:
                label = 0

        dict.append({"subject": 'sub' + str(j), "label" : label})




        #print(difference_matrix)
    with open('/home/akshit/Desktop/KinectProject_new/data_mapped/labels.csv', 'w') as file:
        fields = ['subject', 'label']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dict)
    print(dict)

def main():
    main()

if __name__ == "__main__":
    main()