
from PIL import Image




def img_read(path):
    img=Image.open(path)
    iw,ih=img.size
    w,h=531,520
    scale=min(w/iw,h/ih)
    nw,nh=int(iw*scale),int(ih*scale)
    image = img.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w,h), (240,240,240))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    x=new_image.save(r"../predict/res/cache/temp.jpg")
