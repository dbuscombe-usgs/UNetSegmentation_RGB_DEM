

from imports import *

from imageio import imwrite, imread
from skimage.filters.rank import median
from skimage.morphology import disk
import matplotlib.pyplot as plt



#############################################################
imdir = 'data/images'

dem_path = 'data/dems'

lab_path = 'data/labels'

## write 0 in label where image is 0

images = tf.io.gfile.glob(imdir+os.sep+'samples'+os.sep+'*.jpg')

nb_images=len(images)
print(nb_images)

# BAD = []
for i in images:
    l = i.replace('images', 'labels').replace('.jpg','_label.jpg')
    im = imread(i)
    lab = imread(l)
    lab[im[:,:,0]==0] = 0
    imwrite(l, lab)
    #plt.imshow(im); plt.imshow(lab, alpha=0.5); plt.show()
    # good = input('good?')
    # if good is not 'y':
    #     BAD.append(i)


#==========================================================

NY = 608
NX = 608

data_gen_args = dict(featurewise_center=False,
                     featurewise_std_normalization=False,
                     rotation_range=5,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.1)
image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
dem_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)


L = []
i=1
for k in range(2): #create 2 more sets ....

    #set a different seed each time to get a new batch of 100
    seed = int(np.random.randint(0,100,size=1))
    img_generator = image_datagen.flow_from_directory(
            imdir,
            target_size=(NX, NY),
            batch_size=100, ## .... of 100 images ...
            class_mode=None, seed=seed, shuffle=True)

    dem_generator = image_datagen.flow_from_directory(
            dem_path,
            target_size=(NX, NY),
            batch_size=100, ## .... of 100 images ...
            class_mode=None, seed=seed, shuffle=True)

    #the seed must be the same as for the training set to get the same images
    mask_generator = mask_datagen.flow_from_directory(
            lab_path,
            target_size=(NX, NY),
            batch_size=100,  ## .... and of 10 labels
            class_mode=None, seed=seed, shuffle=True)

    #The following merges the 3 generators (and their flows) together:
    train_generator = (pair for pair in zip(img_generator, dem_generator, mask_generator))

    #grab a batch of images, dems, and label images
    x, y, z = next(train_generator)

    # write them to file and increment the counter
    for im,dem,lab in zip(x,y,z):
        imwrite(imdir+os.sep+'augimage_000'+str(i)+'.jpg', im)
        imwrite(dem_path+os.sep+'augdem_000'+str(i)+'.jpg', dem)
        lab += 1 #avoid averaging over zero
        lab = median(lab[:,:,0]/255.0, disk(3)).astype(np.uint8)
        lab[im[:,:,0]==0]=0
        lab[lab>11] = 11

        print(np.unique(lab.flatten()))

        L.append(np.unique(lab.flatten()))
        imwrite(lab_path+os.sep+'auglabel_000'+str(i)+'_label.jpg',(lab).astype(np.uint8))
        i += 1

    #save memory
    del x, y, im, dem, lab
    #get a new batch

print(np.round(np.unique(np.hstack(L))))

# [ 0  1  2  3  4  5  6  7  8  9 10 11]


im1 = imread('data/images/augimage_0001.jpg')
im2 = imread('data/dems/augdem_0001.jpg')[:,:,0]
merged = np.dstack((im1, im2))

plt.subplot(131)
plt.imshow(im1); plt.axis('off'); plt.title('RGB')
plt.subplot(132)
plt.imshow(im2); plt.axis('off'); plt.title('DEM')
plt.subplot(133)
plt.imshow(merged); plt.axis('off'); plt.title('4D')
plt.savefig('4d.png', dpi=200, bbox_inches='tight'); plt.close('all')


###############################################################
## VARIABLES
###############################################################


tfrecord_dir = os.getcwd()+os.sep+'tfrecords'

images = tf.io.gfile.glob(imdir+os.sep+'*.jpg')

nb_images=len(images)
print(nb_images)

ims_per_shard = 10

SHARDS = int(nb_images / ims_per_shard) + (1 if nb_images % ims_per_shard != 0 else 0)

shared_size = int(np.ceil(1.0 * nb_images / SHARDS))

flag='3d'
dataset = tf.data.Dataset.list_files(imdir+os.sep+'*.jpg', seed=10000) # This also shuffles the images
dataset = dataset.map(read_seg_image_and_label_dunes)
dataset = dataset.map(recompress_seg_image, num_parallel_calls=AUTO)
dataset = dataset.batch(shared_size)

for shard, (image, label) in enumerate(dataset):
  shard_size = image.numpy().shape[0]
  filename = tfrecord_dir+os.sep+"dunes3d-" + "{:02d}-{}.tfrec".format(shard, shard_size)

  with tf.io.TFRecordWriter(filename) as out_file:
    for i in range(shard_size):
      example = to_seg_tfrecord(image.numpy()[i],label.numpy()[i])
      out_file.write(example.SerializeToString())
    print("Wrote file {} containing {} records".format(filename, shard_size))


flag='4d'
dataset = tf.data.Dataset.list_files(imdir+os.sep+'*.jpg', seed=10000) # This also shuffles the images
dataset = dataset.map(read_seg_image_and_label_dunes)
dataset = dataset.map(recompress_seg_image4d, num_parallel_calls=AUTO)
dataset = dataset.batch(shared_size)

for shard, (image, label) in enumerate(dataset):
  shard_size = image.numpy().shape[0]
  filename = tfrecord_dir+os.sep+"dunes4d-" + "{:02d}-{}.tfrec".format(shard, shard_size)

  with tf.io.TFRecordWriter(filename) as out_file:
    for i in range(shard_size):
      example = to_seg_tfrecord(image.numpy()[i],label.numpy()[i])
      out_file.write(example.SerializeToString())
    print("Wrote file {} containing {} records".format(filename, shard_size))
