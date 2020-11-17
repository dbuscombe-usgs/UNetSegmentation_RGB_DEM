


from imports import *


def get_batched_dataset(filenames):

    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = True

    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTO)
    dataset = dataset.map(read_seg_tfrecord_dunes, num_parallel_calls=AUTO)

    dataset = dataset.cache() # This dataset fits in RAM
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) # drop_remainder will be needed on TPU
    dataset = dataset.prefetch(AUTO) #

    return dataset

def get_training_dataset():
    return get_batched_dataset(training_filenames)

def get_validation_dataset():
    return get_batched_dataset(validation_filenames)

@tf.autograph.experimental.do_not_convert
def read_seg_tfrecord_dunes(example):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "label": tf.io.FixedLenFeature([], tf.string),   # shape [] means scalar
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32)/ 255.0

    label = tf.image.decode_jpeg(example['label'], channels=1)
    label = tf.cast(label, tf.uint8)

    cond = tf.greater(label, tf.ones(tf.shape(label),dtype=tf.uint8)*8)
    label = tf.where(cond, tf.ones(tf.shape(label),dtype=tf.uint8)*0, label)

    label = tf.one_hot(tf.cast(label, tf.uint8), 9)
    label = tf.squeeze(label)

    return image, label





data_path= os.getcwd()+os.sep+"data/dunes"

filepath = os.getcwd()+os.sep+'results/dunes_8class_best_weights_model.h5'

hist_fig = os.getcwd()+os.sep+'results/dunes_8class_model.png'

patience = 20

ims_per_shard = 12

VALIDATION_SPLIT = 0.5

BATCH_SIZE = 4

filenames = sorted(tf.io.gfile.glob(data_path+os.sep+'dunes3d*.tfrec'))

nb_images = ims_per_shard * len(filenames)
print(nb_images)

split = int(len(filenames) * VALIDATION_SPLIT)

training_filenames = filenames[split:]
validation_filenames = filenames[:split]

validation_steps = int(nb_images // len(filenames) * len(validation_filenames)) // BATCH_SIZE
steps_per_epoch = int(nb_images // len(filenames) * len(training_filenames)) // BATCH_SIZE


train_ds = get_training_dataset()


L = []
for k in range(12):
  plt.figure(figsize=(16,16))
  for imgs,lbls in train_ds.take(1):
    #print(lbls)
    for count,(im,lab) in enumerate(zip(imgs, lbls)):
       plt.subplot(int(BATCH_SIZE/2),int(BATCH_SIZE/2),count+1)
       plt.imshow(im)
       plt.imshow(np.argmax(lab,-1), cmap=plt.cm.bwr, alpha=0.5)#, vmin=0, vmax=7)
       #plt.imshow(lab, cmap=plt.cm.bwr, alpha=0.5, vmin=0, vmax=9)
       plt.axis('off')
       L.append(np.unique(np.argmax(lab,-1)))

  plt.show()

val_ds = get_validation_dataset()


### Model training


nclasses=9
TARGET_SIZE = 608
model = res_unet((TARGET_SIZE, TARGET_SIZE, 3), BATCH_SIZE, 'multiclass', nclasses)
model.compile(optimizer = 'adam', loss = tf.keras.losses.CategoricalHinge(), metrics = [mean_iou])

earlystop = EarlyStopping(monitor="val_loss",
                              mode="min", patience=patience)

model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss',
                                verbose=0, save_best_only=True, mode='min',
                                save_weights_only = True)

lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=True)

callbacks = [model_checkpoint, earlystop, lr_callback]


history = model.fit(train_ds, steps_per_epoch=steps_per_epoch, epochs=MAX_EPOCHS,
                      validation_data=val_ds, validation_steps=validation_steps,
                      callbacks=callbacks)

plot_seg_history_iou(history, hist_fig)

plt.close('all')
K.clear_session()


### Model evaluation
scores = model.evaluate(val_ds, steps=validation_steps)

print('loss={loss:0.4f}, Mean IOU={iou:0.4f}'.format(loss=scores[0], iou=scores[1]))

sample_data_path = os.getcwd()+os.sep+'data/dunes/images/files'
test_samples_fig = os.getcwd()+os.sep+'dunes_sample_16class_est16samples.png'

sample_filenames = sorted(tf.io.gfile.glob(sample_data_path+os.sep+'*.jpg'))


from matplotlib.colors import ListedColormap
cmap = ListedColormap(["#000000", "#3366CC", "#DC3912", "#FF9900", "#109618", "#990099", "#0099C6", "#DD4477", "#66AA00"])

plt.figure(figsize=(24,24))

for counter,f in enumerate(sample_filenames):
    image = seg_file2tensor(f)/255
    est_label = model.predict(tf.expand_dims(image, 0) , batch_size=1).squeeze()

    est_label = tf.argmax(est_label, axis=-1)

    plt.subplot(4,4,counter+1)
    name = sample_filenames[counter].split(os.sep)[-1].split('.jpg')[0]
    plt.title(name, fontsize=10)
    plt.imshow(image)

    plt.imshow(est_label, alpha=0.5, cmap=cmap, vmin=0, vmax=8)

    plt.axis('off')

# plt.show()
plt.savefig(test_samples_fig,
            dpi=200, bbox_inches='tight')
plt.close('all')
