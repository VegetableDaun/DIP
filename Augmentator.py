# Create augmentator for data
import albumentations

Aug = albumentations.Compose([
    albumentations.HorizontalFlip(),  # horizontal flips
    albumentations.Affine(rotate=(-20, 20), p=0.50),  # rotation
    albumentations.RandomBrightnessContrast(p=0.20)  # random brightness
])
