import cv2 as cv
import numpy as np
import math

def get_gradients(image):
    img = np.array(image, dtype=float)
    gx = np.array(image, dtype=float)
    gy = np.array(image, dtype=float)

    for row in range(128):
        for col in range(64):
            if col == 0:
                gx[row][col] = img[row][col + 1] - 0
            elif col == 63:
                gx[row][col] = 0 - img[row][col - 1]
            else:
                gx[row][col] = img[row][col + 1] - img[row][col - 1]

    for row in range(128):
        for col in range(64):
            if row == 0:
                gy[row][col] = 0 - img[row + 1][col]
            elif row == 127:
                gy[row][col] = img[row - 1][col] - 0
            else:
                gy[row][col] = img[row - 1][col] - img[row + 1][col]

    return gx , gy

def get_mag_phase(gx,gy):
    magn = np.array(image, dtype=float)
    phase = np.array(image, dtype=float)
    for row in range(128):
        for col in range(64):
            magn[row][col] = math.sqrt(pow(gx[row][col], 2) + pow(gy[row][col], 2))
            if gx[row][col] != 0:
                phase[row][col] = math.degrees(math.atan(gy[row][col] / gx[row][col])) % 180
            else:
                phase[row][col] = math.degrees(0.0)
    return magn, phase

def calulate_histogram(mag,phase):
    temp_mag=np.zeros((8,8))
    temp_phase = np.zeros((8, 8))
    blocks_mag=np.zeros((128,8,8))
    blocks_phase = np.zeros((128, 8, 8))
    n=0
    for i in range(0,128,8):
        for j in range(0,64,8):
            for k in range(8):
                for f in range(8):
                    temp_mag [f][k] = mag[i+k][j+f]
                    temp_phase[f][k]=phase[i+k][j+f]

            blocks_mag[n]=temp_mag
            blocks_phase[n] = temp_phase
            n=n+1
    bins=np.zeros((128,9))
    temp=0
    for i in range(128):
        for j in range(8):
            for k in range(8):
                temp=blocks_phase[i][j][k]
                for x in range(9):
                    if (x*2*10)==temp:
                        bins[i][x]+=blocks_mag[i][j][k]
                        break
                    elif temp>160:
                        bins[i][8] += ((temp - 160) / 20) * blocks_mag[i][j][k]
                        bins[i][0] += ((180 - temp) / 20) * blocks_mag[i][j][k]
                        break
                    elif (x*2*10)>temp:
                        bins[i][x-1]+=((temp-((x-1)*2*10))/20)*blocks_mag[i][j][k]
                        bins[i][x]+=(((x*2*10)-temp)/20)*blocks_mag[i][j][k]
                        break

    return bins

def normalization(bins):
    normalized = np.zeros((105, 36))
    n = 0
    bin = np.zeros((16, 8, 9))
    for j in range(16):
        for k in range(8):
            bin[j][k] = bins[k+(j*8)]

    for j in range(15):
        for k in range(7):
            normalized[n]= np.concatenate(( bin[j][k] , bin[j][ k+1 ] ,bin[ j+1 ][k] , bin[j+1][k+1]))
            n=n+1
    k=0
    for i in range(105):
        for j in range(36):
            k+=pow(normalized[i][j],2)
        k=math.sqrt(k)
        for j in range(36):
            normalized[i][j]=normalized[i][j]/k
    return normalized

# Step 1: Preprocess the image (64 x 128)
image = cv.imread('image.jpg',0)
image = cv.resize(image, dsize=[64, 128])
# Step 2: Calculating Gradients (direction x and y)
gx,gy = get_gradients(image)
# Step 3: Calculate the Magnitude and Orientation
magn,phase = get_mag_phase(gx,gy)
# Step 4: Calculate Histogram of Gradients in 8×8 cells (9×1)
bins = calulate_histogram(magn,phase)
# Step 5: Normalize gradients in 16×16 cell (36×1)
normalized = normalization(bins)
# Step 6: Features for the complete image
normalized_matrix=normalized.flatten()
print("total features for the image : ",normalized_matrix.shape[0])



