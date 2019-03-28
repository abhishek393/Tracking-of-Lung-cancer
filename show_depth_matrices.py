import scipy.io

def show_depth_matrix():
    for j in range(1, 19):
        image_data_path = '/home/akshit/Desktop/KinectProject_new/data/sub' + str(j) + '/'
        print("\n\nPrinting Depth values for Subject " + str(j))
        for i in range(1, 130):
            w = 10000 + i
            st = str(w)
            mat_file = image_data_path + 'DepthFrame' + st[1:] + '.mat'
            mat = scipy.io.loadmat(mat_file)
            string = 'Dep' + st[1:] + '_'
            mat1 = mat[string]
            print(mat1)

def main():
    show_depth_matrix()

if __name__ == '__main__':
    main()
