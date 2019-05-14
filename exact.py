import subprocess as sp
import numpy as np
import time
def extract_frames(file_dir):
    img_size = 224
    args = [
                'ffmpeg',
                '-loglevel', 'quiet',
                '-y',
                '-discard', 'nokey',
                '-i', file_dir,
                '-vframes', '3',
                '-vf', 'scale=\'max(iw*{ss}/ih\\,{ss})\':\'max(ih*{ss}/iw\\,{ss})\',crop={cs}:{cs}'.format(
                    ss=256,
                    cs=img_size
                ),
                '-vsync', 'vfr',
                '-f', 'rawvideo',
                '-pix_fmt', 'rgb24',
                '-an',
                '-']

    pipe = sp.Popen(args, stdout=sp.PIPE, bufsize=10 ** 4)
    images = []
    while True:
        raw_image = pipe.stdout.read(img_size * img_size * 3)
        if not raw_image:
            break
        image = np.fromstring(raw_image, dtype='uint8')
        image = image.reshape((img_size, img_size, 3))
        images.append(image)
        pipe.stdout.flush()
start = time.time() 
for i in range(100):
    extract_frames('MTSVRC/test/video/903502792.mp4')
    print((time.time()-start)/(i+1))
