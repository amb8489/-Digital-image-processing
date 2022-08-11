'''
1) break img into blocks the size of the text
2) put blocks of the img into a list
3) for each block, for each filter find out what char it is and remove that block from the list


possible optimization:

multi threaded: one thread does one half of the alphabet


find and section off only the lines


every time there is a hit go for ward char width

bottom pixel of first row as reference for mesures

'''
import ipcv
import numpy as np
import time

CHAR_WIDTH = 17
CHAR_HEIGHT = 22
import cv2


def angle_between(v1, v2):
    unit_vector1 = v1 / np.linalg.norm(v1)
    unit_vector2 = v2 / np.linalg.norm(v2)
    return np.dot(unit_vector1, unit_vector2)


'''
code  Vowels - (letters-vowels) - 0-9 to ' , - . b/c more likly to get a letter first
'''


def make_ascii_map(masks, mask_ascii_codes):
    m = {}
    b = 0
    for code in mask_ascii_codes:
        m[code] = masks[b]
        b += 1

    vowels = [ord("A"), ord("E"), ord("I"), ord("O"), ord("U")]

    letters = mask_ascii_codes[14:]

    for v in vowels:
        index = np.argwhere(letters == v)
        letters = np.delete(letters, index)

    ascii_codes = np.append(np.asarray(vowels), letters)

    ascii_codes = np.append(ascii_codes, mask_ascii_codes[:14])

    map = {}
    histo = {}

    for i in ascii_codes:
        map[i] = m[i]
        histo[i] = 0

    return map, histo


def character_recognition_ang(src, templates, codes, threshold):
    text = ""
    ascii_map, histo = make_ascii_map(templates, codes)
    codes = ascii_map.keys()

    zeros = np.array(np.where(src == 0))
    PAD = (min(zeros[0]), min(zeros[1]))
    shift = PAD[1]

    for y in range(PAD[0], src.shape[0], CHAR_HEIGHT + 10):

        row_img = src[y:y + CHAR_HEIGHT, 0:src.shape[1]]
        scale = 1

        while len(np.where(row_img[0:2, 0:src.shape[1]] == 0)[0]) == 0:
            row_img = src[y + scale: y + scale + CHAR_HEIGHT, 0:src.shape[1]]
            scale += 1
            print(scale)
            if y + scale + CHAR_HEIGHT > src.shape[0]:
                break

        if row_img.shape[0] != CHAR_HEIGHT:
            continue
        # start = time.time()
        for x in range(0, src.shape[1]):
            block = row_img[0:y + CHAR_HEIGHT, shift: CHAR_WIDTH + shift]
            if block.shape[1] != 17:
                shift = PAD[1]
                break
            for b in codes:
                mask = ascii_map[b]
                zeros = np.array(np.where(mask == 0))
                min_Y = zeros[0][0]
                PAD = (min_Y, zeros[1, 0])
                zeros_Y_block = np.array(np.where(block == 0))
                if (len(zeros_Y_block[0]) > 0):
                    min_Y_block = zeros_Y_block[0][0]
                    PAD_block = (min_Y_block, zeros_Y_block[1, 0])
                    dif_x = PAD_block[1] - PAD[1]
                    bl = row_img[0:y + CHAR_HEIGHT, dif_x + shift:shift + CHAR_WIDTH + dif_x]

                    correctness = angle_between(bl.flatten(), mask.flatten())

                    found = 0
                    # ipcv.showA(cv2.addWeighted(bl, .5, mask, .2, 0.0),"mask",150)

                    if correctness >= threshold:
                        # print("found:",chr(b))
                        ch = chr(b)
                        text += ch
                        shift += CHAR_WIDTH + dif_x - 1
                        found = 1
                        histo[b] += 1
                        # ipcv.showA(cv2.addWeighted(bl, .7, mask, .5, 0.0), "found", 350)
                        break
                else:
                    text += " "
                    shift += CHAR_WIDTH
                    break
            if found == 0:
                shift += 1
        text += "\n"

    return text, histo
