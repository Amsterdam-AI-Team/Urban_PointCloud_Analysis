#!/usr/bin/env python

import cv2
import pathlib
import os
import sys
import pandas as pd

in_folder = 'pole_pics_all/'
out_file = 'pole_data.csv'


cv2.namedWindow("Pole Tagger")

# All files ending with .png
in_files = list(pathlib.Path(in_folder).glob('*.png'))

if len(in_files) == 0:
    print('No files found.')
    sys.exit()

in_codes = [int(f.stem) for f in in_files]

if os.path.isfile(out_file):
    df = pd.read_csv(out_file, index_col=0)
else:
    df = (pd.DataFrame(columns=['code', 'street_light', 'location', 'angle',
                                'height', 'fp', 'unknown'])
          .set_index('code'))
n_cols = df.shape[1]

done_codes = set(df.index.values)
todo_codes = [c for c in in_codes if c not in done_codes]
n_todo = len(todo_codes)

print(f'Found {len(in_codes)} files, {n_todo} remaining.')

i = 0

code = todo_codes[i]
df.loc[code] = [0]*n_cols
img = cv2.imread(f'{in_folder}{code}.png')  # load image
print(f'{i+1}/{n_todo}: tagging {code}.png...')

while True:
    if i >= len(todo_codes):
        print("Done!")
        break

    cv2.imshow("Pole Tagger", img)

    # The function waitKey waits for a key event infinitely (when delay<=0)
    k = cv2.waitKey(33)
    if k == 13 or k == 32:  # [space] or [enter]: next image
        if (df.loc[code] == 0).all():
            print('No data entered yet.')
            continue

        # First save previous data
        df.to_csv('_tmp.csv')

        # Prepare next image
        i += 1
        code = todo_codes[i]
        df.loc[code] = [0]*n_cols
        img = cv2.imread(f'{in_folder}{code}.png')  # load image
        print(f'{i+1}/{n_todo}: tagging {code}.png...')
        continue
    elif k == 99:  # 'c' for correction
        print('Correction...')
        df.loc[code] = [0]*n_cols
        continue
    elif k == 27:  # escape key
        # Discard last entry
        df.drop(index=code, inplace=True)
        break
    elif k == -1:  # normally -1 returned,so don't print it
        continue
    elif k == 48:  # 0: false positive
        print('FP')
        df.loc[code, 'fp'] = 1
    elif k == 49:  # 1: correct
        print('Street light')
        df.loc[code, 'street_light'] = 1
    elif k == 50:  # 2: wrong location
        print('Incorrect location')
        df.loc[code, 'location'] = 1
    elif k == 51:  # 3: wrong angle
        print('Incorrect angle')
        df.loc[code, 'angle'] = 1
    elif k == 52:  # 4: wrong height
        print('Incorrect height')
        df.loc[code, 'height'] = 1
    elif k == 53:  # 5: unknown
        print('Unknown')
        df.loc[code, 'unknown'] = 1
    else:
        print(k)  # else print its value

print(f'Saving tags to {out_file}...')
df.to_csv(out_file)
