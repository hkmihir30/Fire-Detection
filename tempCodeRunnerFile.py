for img in os.listdir(non_fire_images_path):
    img_array = cv2.imread(os.path.join(non_fire_images_path,img))
    plt.imshow(img_array)
    plt.show()
    break