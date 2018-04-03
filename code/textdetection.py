#!/Users/jackbandy/anaconda3/bin/python


# Standard imports
import cv2
import numpy as np
import util
import tensorflow as tf

# convolution kernel filled with 1. This means that convolution with just sums up value under kernel.
# Kernel dimensions should be [inputs_height, inputs_width, channels_in, channels_out].
RLSA_THRESHOLD = 128
RLSA_conv_weights = tf.Variable(tf.constant(1.0, shape=[1, 2 * RLSA_THRESHOLD + 1, 1, 1]),
                                trainable=False,
                                name='RLSA_conv_weights')
def main():
    # Read image
    im = cv2.imread("../data/aligned_sample_pages/Wycliffe-005-001r-2010-col1.jpg",cv2.IMREAD_GRAYSCALE)

    # erode
    im = cv2.medianBlur(im,5)
    kernel = np.ones((5,5), np.uint8)
    im = cv2.erode(im, kernel, iterations=1)

    # binarize
    im = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)
    params = cv2.SimpleBlobDetector_Params()
    
    t = RLSA(np.array(im))
    print("T: {}".format(t.shap))
     
     



# Input should have following dimensions [number_of_inputs, inputs_height, inputs_width, 1 (or for example number of heatmaps)].
def non_maxima_suppression(input):
    # You can remove comment below and reshape input within this method.
    # input = tf.reshape(input, shape=[dim1_size, dim2_size, dim3_size, dim4_size])

    # Finding maxmimum value in [1, 1, 3, 1] window using max pooling. Size should be adjusted to fit your problem.
    max = tf.nn.max_pool(input, ksize=[1, 1, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
    # If value at position max(dim1_size, y, x, dim4_size) equels to input(dim1_size, y, x, dim4_size) then the value was
    # infact maximum in given window and we want to preserve that value, otherwise we will set that value to zero.
    result = tf.select(tf.equal(max, input), max, negative)
    # In case that max(dim1_size, y, x, dim4_size) was not maximum but distance between max(dim1_size, y, x, dim4_size)
    # and actual maximum value is less then threshold, then we will keep max(dim1_size, y, x, dim4_size) as well.
    # result = tf.select(tf.less(max - input, THRESHOLD), max, negative)
    return result



def RLSA(inp):
    # RLSA works with binary data, therefore we will set all data below threshold to 0 and everything else to 1.
    #threshold = tf.where(inp > RLSA_THRESHOLD, positive, negative)
    threshold = inp
    # Convolution with given kernel will sum up elements in window (we are using only row convolution).
    RLSA_conv = tf.nn.conv2d(threshold, RLSA_conv_weights, strides=[1, 1, 1, 1], padding='SAME')
    # If the sum of single convolution is equal or bigger than RLSA_THRESHOLD then we can set value to 1 and 0 otherwise.
    #RLSA_output = tf.where(RLSA_conv >= RLSA_THRESHOLD, positive, negative)
    return RLSA_conv



if __name__ == "__main__":
    main()



