from data import DIV2K

train_loader = DIV2K(scale=4,  # 2, 3, 4 or 8
                     downgrade='bicubic',  # 'bicubic', 'unknown', 'mild' or 'difficult'
                     subset='train')  # Training dataset are images 001 - 800

# Create a tf.data.Dataset
train_ds = train_loader.dataset(batch_size=16,  # batch size as described in the EDSR and WDSR papers
                                random_transform=True,  # random crop, flip, rotate as described in the EDSR paper
                                repeat_count=None)  # repeat iterating over training images indefinitely
