import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import xlsxwriter

# np.set_printoptions(threshold=sys.maxsize) #cek hasil output e ga kepotong
np.seterr(divide='ignore', invalid='ignore') #gae ignore notif
# import gambar
rgbImage = cv2.imread('oli.jpg'); #import gbr
print('Original Dimensions : ',rgbImage.shape)

# Resized
# scale_percent = 60  # percent of original size
# width = int(rgbImage.shape[1] * scale_percent / 100)
# height = int(rgbImage.shape[0] * scale_percent / 100)
# dim = (width, height)

# Resize 2
width = 200
height = 200
dim = (width, height)
# resize image
resized = cv2.resize(rgbImage, dim, interpolation=cv2.INTER_AREA)

print('Resized Dimensions : ', resized.shape)

# cv2.imshow("Resized image", resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# misahin chanel
B, G, R = resized[:, :, 0], resized[:, :, 1], resized[:, :, 2] # For RGB image
# pix_val_flat_B = [x for sets in B for x in sets] #cek iso flat
# pix_val_flat_G = [x for sets in G for x in sets] #cek iso flat
# pix_val_flat_R = [x for sets in R for x in sets] #cek iso flat
# print (pix_val_flat_B)
# print(B)
# print (pix_val_flat_G)
# print(G)
# print (pix_val_flat_R)
# print(R)

# membuat value RGB menjadi 0 sampai 1
B = cv2.normalize(B.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) #cek dadi range 0 - 1
nB = [x for sets in B for x in sets] #cek iso flat
#print("Nilai nB", nB)
G = cv2.normalize(G.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) #cek dadi range 0 - 1
nG = [x for sets in G for x in sets] #cek iso flat
#print("Nilai nG", nG)
R = cv2.normalize(R.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) #cek dadi range 0 - 1
nR = [x for sets in R for x in sets] #cek iso flat
#print("Nilai nR", nR)
print("nilai R", R)
print("nilai G", G)
print("nilai B", B)

#Mencari nilai V
V = np.maximum(np.array(B), np.array(G), np.array(R))
print("NILAI V", V)


#Mencari Nilai S
X = np.minimum(np.array(B), np.array(G), np.array(R))
print("NILAI X", X)

vx=np.array(V)-np.array(X)
print("Nilai vx", vx)
if np.any(V) == 0:
    S = 0
else:
    S = np.array(vx) / np.array(V)
# S=(np.array(V)-np.array(X))/np.array(V)
print("Nilai S", S)

#Mencari Nilai H
r = (np.array(V) - np.array(R)) / np.array(vx)
if np.isnan(np.any(r)): #isnan buat ngecek apakah angka atau bukan
    r = 0
g=(np.array(V)-np.array(G))/np.array(vx)
if np.isnan(np.any(g)):
    g = 0
b=(np.array(V)-np.array(B))/np.array(vx)
if np.isnan(np.any(b)):
    b = 0
# print("nilai r", r)
if np.any(R) == 0 and np.any(G) == 0 and np.any(B) == 0:
    H = 0
elif np.any(R) == np.any(V):
    if np.any(G) == np.any(X):
        H = 5 + np.array(b)
    else:
        H = 1 - np.array(g)
elif np.any(G) == np.any(V):
    if np.any(B) == np.any(X):
        H = 1 + np.array(r)
    else:
        H = 3 - np.array(b)
else:
    if np.any(R) == np.any(X):
        H = 3 + np.array(g)
    else:
        H = 5 - np.array(r)

H=H/6
print("nilai H", H)


# cv2.imshow("nilai R", H)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


cv2.imshow("HSV", cv2.merge([np.array(V), np.array(S), np.array(H)]))
cv2.waitKey(0)
# cv2.imshow("Hue", H)
# cv2.waitKey(0)
# cv2.imshow("Saturation", S)
# cv2.waitKey(0)
# cv2.imshow("Value", V)
# cv2.waitKey(0)
# cv2.imshow("H", cv2.merge([H, V, S]))
# cv2.waitKey(0)
#
# cv2.imshow("S", cv2.merge([S, H, V]))
# cv2.waitKey(0)
#
# cv2.imshow("S", cv2.merge([S, V, H]))
# cv2.waitKey(0)
#
# cv2.imshow("V", cv2.merge([V, S, H]))
# cv2.waitKey(0)
#
# cv2.imshow("V", cv2.merge([V, H, S]))
# cv2.waitKey(0)

cv2.destroyAllWindows()

# np.savetxt("foo.csv", R, delimiter=",") #gae ngesave csv
# np.savetxt("foo.csv", G, delimiter=",")
# np.savetxt("foo.csv", B, delimiter=",")
# workbook = xlsxwriter.Workbook('contoh2.xlsx')
# worksheet = workbook.add_worksheet()
#
# for i in range(len(R)):
#     for j in range(len(R[i])):
#         worksheet.write(i,j,R)
#
# workbook.close()

#----------K-K---N---N---N---N--------------------------------------------------------------#
#----------KK----N-N-N---N-N-N----------------------------KNN-------------------------------#
#----------K-K---N---N---N---N--------------------------------------------------------------#

