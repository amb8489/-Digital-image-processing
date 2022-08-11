'''
author aaron berghash


'''
import ipcv
import numpy as np
import time

CHAR_WIDTH = 17
CHAR_HEIGHT = 22
import cv2

'''
gets angle between two vects

'''
def angle_between(v1, v2):
    unit_vector1 = v1 / np.linalg.norm(v1)
    unit_vector2 = v2 / np.linalg.norm(v2)
    return np.dot(unit_vector1, unit_vector2)


'''
code  Vowels - (letters-vowels) - 0-9 to ' , - . b/c more likly to get a letter first

maps ascii to filter 
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


'''
OPTIMIZATION:


1) first it finds the very first black pixel, and goes two white rows up from it.


2) the row is checked that the first two rows have at least 1 black pixel in the first two rows, else the row is shifted down one till.

3) when it cuts out a row from the img that is 22 high by the img width long


4) next from the row slice , from left to right a block is formed by taking 17 pixel wide chunck from the start of row to
17 out, while this chunck is all white ie just white space padding on ether end of the page, the block this shifted a char width
over..

5) once the block has some black pixels in it the first mask is over layed on that block the upper left most black pixel is
calulated for each fliter and block and their difference in X is found, next using that difference the block is shifted by that amount
essentally lining up block and the letter fillter based off of their first black pixel and then the matching is done (ether cos or match)

this is done for each filter till a match is fouund or not, if found the block shifts over 17 more, if not it shifts by 1, this would mean a space or
a non named char.  this happens till the block reaaches the end of the row.

the row is moved down till it finds a new row ie finds a new black pixel lowwer then the lowwest of the previous row and all is repeated till end of
the img is found 


'''
def character_recognition_ang(src, templates, codes, threshold):
    text = ""

    # mapping filter to ascii code, maping empty histogram

    ascii_map, histo = make_ascii_map(templates, codes)
    codes = ascii_map.keys()


    # 1) from doc
    zeros = np.array(np.where(src == 0))
    PAD = (min(zeros[0]), min(zeros[1]))
    shift = PAD[1]
    #---------------
    for y in range(PAD[0], src.shape[0], CHAR_HEIGHT + 10):
        # 2) from doc
        row_img = src[y:y + CHAR_HEIGHT, 0:src.shape[1]]
        scale = 1

        while len(np.where(row_img[0:2, 0:src.shape[1]] == 0)[0]) == 0:
            row_img = src[y + scale: y + scale + CHAR_HEIGHT, 0:src.shape[1]]
            scale += 1
            if y + scale + CHAR_HEIGHT > src.shape[0]:
                break

        if row_img.shape[0] != CHAR_HEIGHT:
            continue
        #-------------------
        for x in range(0, src.shape[1]):

            #  3)
            block = row_img[0:y + CHAR_HEIGHT, shift: CHAR_WIDTH + shift]
            if block.shape[1] != 17:
                shift = PAD[1]
                break
            #--------------
            for b in codes:
                # 4)
                mask = ascii_map[b]
                zeros = np.array(np.where(mask == 0))
                min_Y = zeros[0][0]
                PAD = (min_Y, zeros[1, 0])
                zeros_Y_block = np.array(np.where(block == 0))
                if (len(zeros_Y_block[0]) > 0):
                    min_Y_block = zeros_Y_block[0][0]
                    PAD_block = (min_Y_block, zeros_Y_block[1, 0])

                    #5)
                    dif_x = PAD_block[1] - PAD[1]
                    bl = row_img[0:y + CHAR_HEIGHT, dif_x + shift:shift + CHAR_WIDTH + dif_x]

                    # see this to animate
                    # ipcv.show(cv2.addWeighted(bl, .5, mask, .2, 0.0),"mask")


                    # checking
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
