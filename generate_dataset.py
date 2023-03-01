import cv2
import os
import shutil
import numpy as np
from definition import *


# create source images with rotation and flips
def rot_flip(image, mask, name, temp_im_path, temp_mk_path):
    count = 0
    cv2.imwrite(temp_im_path + name + '_' + str(count) + '.png', image)
    cv2.imwrite(temp_mk_path + name + '_' + str(count) + '.png', mask)
    count += 1

    image_flip0 = cv2.flip(image, 0)
    mask_flip0 = cv2.flip(mask, 0)
    cv2.imwrite(temp_im_path + name + '_' + str(count) + '.png', image_flip0)
    cv2.imwrite(temp_mk_path + name + '_' + str(count) + '.png', mask_flip0)
    count += 1

    image_flip1 = cv2.flip(image, 1)
    mask_flip1 = cv2.flip(mask, 1)
    cv2.imwrite(temp_im_path + name + '_' + str(count) + '.png', image_flip1)
    cv2.imwrite(temp_mk_path + name + '_' + str(count) + '.png', mask_flip1)
    count += 1

    image_flip01 = cv2.flip(image_flip0, 1)
    mask_flip01 = cv2.flip(mask_flip0, 1)
    cv2.imwrite(temp_im_path + name + '_' + str(count) + '.png', image_flip01)
    cv2.imwrite(temp_mk_path + name + '_' + str(count) + '.png', mask_flip01)
    count += 1

    image_rotate90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    mask_rotate90 = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(temp_im_path + name + '_' + str(count) + '.png', image_rotate90)
    cv2.imwrite(temp_mk_path + name + '_' + str(count) + '.png', mask_rotate90)
    count += 1

    image_r_flip0 = cv2.flip(image_rotate90, 0)
    mask_r_flip0 = cv2.flip(mask_rotate90, 0)
    cv2.imwrite(temp_im_path + name + '_' + str(count) + '.png', image_r_flip0)
    cv2.imwrite(temp_mk_path + name + '_' + str(count) + '.png', mask_r_flip0)
    count += 1

    image_r_flip1 = cv2.flip(image_rotate90, 1)
    mask_r_flip1 = cv2.flip(mask_rotate90, 1)
    cv2.imwrite(temp_im_path + name + '_' + str(count) + '.png', image_r_flip1)
    cv2.imwrite(temp_mk_path + name + '_' + str(count) + '.png', mask_r_flip1)
    count += 1

    image_r_flip01 = cv2.flip(image_r_flip0, 1)
    mask_r_flip01 = cv2.flip(mask_r_flip0, 1)
    cv2.imwrite(temp_im_path + name + '_' + str(count) + '.png', image_r_flip01)
    cv2.imwrite(temp_mk_path + name + '_' + str(count) + '.png', mask_r_flip01)


def check_edges(img):
    height, width = img.shape

    cropped_ok = False
    if np.count_nonzero(img[0, :]) == 0 and np.count_nonzero(img[-1, :]) == 0 and \
            np.count_nonzero(img[:, 0]) == 0 and np.count_nonzero(img[:, -1]) == 0:

        if np.count_nonzero(img) > 0:
            return True
    return cropped_ok


def r_pixel():
    return np.random.randint(0, 255)


def get_pts(image):
    rows, cols, ch = image.shape
    pts1 = np.float32(
        [[cols * .25, rows * .95],
         [cols * .90, rows * .95],
         [cols * .10, 0]]
    )
    pts2 = np.float32(
        [[cols * .25 + cols * r_pixel() / 255 * 0.2, rows * .95 - rows * r_pixel() / 255 * 0.2],
         [cols * .90 - cols * r_pixel() / 255 * 0.2, rows * .95 - rows * r_pixel() / 255 * 0.2],
         [cols * .10 + cols * r_pixel() / 255 * 0.2, 0 + rows * r_pixel() / 255 * 0.2]]
    )
    return pts1, pts2


if __name__=="__main__":

    if os.path.exists('temp_img_folder'):
        shutil.rmtree('temp_img_folder')
    if os.path.exists('temp_msk_folder'):
        shutil.rmtree('temp_msk_folder')
    if os.path.exists('image_data'):
        shutil.rmtree('image_data')
    if os.path.exists('mask_data'):
        shutil.rmtree('mask_data')

    os.makedirs('temp_img_folder', exist_ok=True)
    os.makedirs('temp_msk_folder', exist_ok=True)
    os.makedirs('image_data', exist_ok=True)
    os.makedirs('mask_data', exist_ok=True)

    temp_im_path = 'temp_img_folder/'
    temp_mk_path = 'temp_msk_folder/'
    store_img_path = 'image_data/'
    store_msk_path = 'mask_data/'

    c = 0
    for im_path, msk_path in zip(images_paths, masks_paths):
        image = cv2.imread(im_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        mask = cv2.imread(msk_path, 0)
        rot_flip(image, mask, 'orig' + str(c), temp_im_path, temp_mk_path)
        c += 1


    alpha = 0.5
    beta = 1 - alpha
    orig_files = os.listdir(temp_im_path)
    len_orig_files = len(orig_files)
    file_num = 0

    print('Image generation in progress............')

    for i in range(0, 6000):
        fil_sel = np.random.randint(0, len_orig_files)
        image = cv2.imread(temp_im_path + orig_files[fil_sel])
        #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mask = cv2.imread(temp_mk_path + orig_files[fil_sel], 0)

        if np.random.randint(0, 100) < 50:
            skew = True
        else:
            skew = False

        if skew:
            pts1, pts2 = get_pts(image)
            M = cv2.getAffineTransform(pts1, pts2)
            rows, cols, ch = image.shape

            img_skew = cv2.warpAffine(image, M, (cols, rows))
            mask = cv2.warpAffine(mask, M, (cols, rows))

            background_noise = np.random.randn(img_skew.shape[0], img_skew.shape[1], 3)
            background_noise = (background_noise / background_noise.max() * 255).astype(np.uint8)

            image = cv2.addWeighted(background_noise, alpha, img_skew, beta, 0)

        max_h, max_w = mask.shape
        while_loop = True
        while (True):
            if not while_loop:
                break

            x = np.array((np.random.randint(0, max_h), np.random.randint(0, max_w)))
            height = np.random.randint(1, max_h - x[0] + 1)
            width = np.random.randint(1, max_w - x[1] + 1)

            if height / width > 0.5 and width / height > 0.5:
                cropped_mask = mask[x[0]:x[0] + height, x[1]:x[1] + width]

                good = check_edges(cropped_mask)

                if good:
                    cropped_image = image[x[0]:x[0] + height, x[1]:x[1] + width]
                    while_loop = False

        cv2.imwrite(store_img_path + orig_files[fil_sel].replace('.png', str(file_num) + '.png'), cropped_image)
        cv2.imwrite(store_msk_path + orig_files[fil_sel].replace('.png', str(file_num) + '.png'), cropped_mask)
        file_num += 1

    if os.path.exists('temp_img_folder'):
        shutil.rmtree('temp_img_folder')
    if os.path.exists('temp_msk_folder'):
        shutil.rmtree('temp_msk_folder')
