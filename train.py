from barcode_model import train_model

train_dir = r'/Projects/Chat/dataset/train'
validation_dir = r'/Projects/Chat/dataset/validation'

model = train_model(train_dir, validation_dir)
