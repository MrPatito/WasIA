from PIL import Image, ImageDraw

# Create a new image with a red background
img = Image.new('RGB', (224, 224), color='red')

# Add some text to make it more distinctive
d = ImageDraw.Draw(img)
d.text((10,10), "Test Image", fill='white')

# Save the image
img.save('test_data/test_image.png')
