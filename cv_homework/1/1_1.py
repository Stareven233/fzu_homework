from PIL import Image
 
image = Image.open("Fzu_shutong.jpg")
image.show()
image_L = image.convert('L')
image_L.save('Fzu_shutong_L.jpg')
