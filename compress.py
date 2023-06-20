from PIL import Image

# Images have to be made smaller otherwise frame interpolation exceeds memory allocation

def crop(i):
    img = Image.open('./data/{}.png'.format(i))
    basewidth = 300
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.Resampling.LANCZOS)
    #w, h = img.size
    #img = img.crop((70, 32, w-70, h-32))
    #img.resize((160, 160))
    img.save('./data/resized/{}.png'.format(i))


for i in range(57, 91):
    crop(i)

for i in range(265, 322):
    crop(i)

for i in range(332, 343):
    crop(i)

