'''
create_damaged_words.py
damage word images in a dataset
'''

import os
import numpy as np
import imageio
import util
import pdb


images_path = '../pz-data/data/word_images_normalized/'
output_images_path = '../pz-data/data/word_images_damaged/'


def main():
    # initialization
    samples, height, width = util.collectSamples(images_path, binarize=False)
    file_names = os.listdir(images_path)
    file_names.sort()

    damaged_samples = np.zeros(samples.shape)
    # damage loop
    damaged_count = 0
    total_count = len(samples)
    print("Damaging words...")
    for i in range(total_count):
        if damaged_count > total_count / 2 or np.random.randint(0,2) == 0:
            damaged_samples[i] = samples[i]
        else:
            # damage!
            damaged_count +=1
            damage_index = np.random.randint(1,6)
            damaged_samples[i] = damageSample(samples[i], damage_index)

    # save loop
    print("Saving damaged words...")
    if not (os.path.isdir(output_images_path)):
        os.mkdir(output_images_path)
    for i in range(total_count):
        inv = util.invertImage(damaged_samples[i])
        imageio.imwrite('{}/{}.jpg'.format(output_images_path, file_names[i]), inv)



def damageSample(sample, index):
    to_return = sample
    for _ in range(index):
        candidates = np.where(to_return > 0)
        if candidates[0].shape[0] == 0:
            return sample
        dam_width = np.random.randint(6,17)
        dam_height = np.random.randint(6,17)
        dam_location = np.random.randint(0,candidates[0].shape[0])
        dam_row = candidates[0][dam_location]
        dam_col = candidates[1][dam_location]
        to_return[dam_row:dam_row+dam_height, dam_col:dam_col+dam_height] = 0

    return to_return



if __name__ == '__main__':
    main()
