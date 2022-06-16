from PIL import Image
import os

data_dir = '/home/njuciairs/zmy/code/PSNRL/output'
# path = data_dir + '/target.png'
def crop(path, dirOutput):
    # dirOutput = data_dir + '/target'
    try:
        os.makedirs(dirOutput)
    except OSError:
        pass
    img = Image.open(path)
    for i in range(0, 8):
        for j in range(0, 8):
            if i == 0:
                left = 2
            else:
                left = 2*(i+1) + 64*i
            right = left + 64
            if j == 0:
                up = 2
            else:
                up = 2*(j+1) + 64*j
            down = up + 64
            out = img.crop((left, up, right, down))
            suffix = str(i) + '_' + str(j) + '.png'
            out.save(os.path.join(dirOutput, suffix))

files = ['/target.png', '/source.png', '/gene.png']
for file in files:
    crop(data_dir + file, data_dir + file[:-4])
