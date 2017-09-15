from PIL import Image
import sys

im = Image.open(sys.argv[1])

x,y = im.size

for i in range(x):
    for j in range(y):
        r,g,b = im.getpixel((i,j))
        im.putpixel((i,j), (int(r/2),int(g/2),int(b/2)))

im.save("Q2.png", "png")











