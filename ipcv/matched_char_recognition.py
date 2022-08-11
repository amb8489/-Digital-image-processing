'''
author aaron see angle_char_recognition.py for doc they are the same for both
'''
import ipcv
import numpy as np
import time
import cv2


CHAR_WIDTH = 17
CHAR_HEIGHT = 22
'''
    zeros = np.where(v1 == 0 )
    zeros = np.column_stack(zeros)

'''
def invert_0_and_255(v1):
    return ((v1 - 255)*255)





def closeness(v1,v2):
    # print()
    # print(len(np.where(v1 != 0)[0]))
    # print(np.sum(v1 * v2))
    # print("-------------------")
    return len(np.where(v1!=0)[0])*np.sum(v1*v2)



def normalize(v1):
    v1 = np.divide(v1, sum(v1.flatten()))
    return v1


'''
code  Vowels - (letters-vowels) - 0-9 to ' , - . b/c more likly to get a letter first
'''


def make_ascii_map(masks, mask_ascii_codes):
    m = {}
    b = 0
    for i in mask_ascii_codes:
        m[i] = masks[b]
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


def character_recognition_mat(src, templates, codes, threshold):
    print("running")
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
                    correctness = closeness(normalize(invert_0_and_255(bl)), normalize(invert_0_and_255(mask)))
                    found = 0
                    if correctness >= threshold:
                        ch = chr(b)
                        text += ch
                        shift += CHAR_WIDTH + dif_x - 1
                        found = 1
                        histo[b] += 1
                        break
                else:
                    text += " "
                    shift += CHAR_WIDTH
                    break
            if found == 0:
                shift += 1
        text += "\n"

    return text, histo
