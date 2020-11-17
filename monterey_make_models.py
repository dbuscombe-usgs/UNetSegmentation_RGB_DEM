

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

#
# water
# marsh
# ag
# beach
# foredune
# dense_dune
# sparse_dune
# bare_dune
# iceplant
# pavement_path
# other



nclasses=12
TARGET_SIZE = 608
ims_per_shard = 10

patience = 20

VALIDATION_SPLIT = 0.6

BATCH_SIZE = 6
#====================================================================
## initial 3d model

# data_path= os.getcwd()+os.sep+"data"

filepath = os.getcwd()+os.sep+'results/dunes_11class_best_weights_model.h5'

hist_fig = os.getcwd()+os.sep+'results/dunes_11class_model.png'


filenames = sorted(tf.io.gfile.glob(os.getcwd()+os.sep+'tfrecords/threeD/dunes3d*.tfrec'))

nb_images = ims_per_shard * len(filenames)
print(nb_images)

split = int(len(filenames) * VALIDATION_SPLIT)

training_filenames = filenames[split:]
validation_filenames = filenames[:split]

validation_steps = int(nb_images // len(filenames) * len(validation_filenames)) // BATCH_SIZE
steps_per_epoch = int(nb_images // len(filenames) * len(training_filenames)) // BATCH_SIZE

train_ds = get_training_dataset()

L = []
for k in range(3):
  plt.figure(figsize=(16,16))
  for imgs,lbls in train_ds.take(1):
    #print(lbls)
    for count,(im,lab) in enumerate(zip(imgs, lbls)):
       plt.subplot(int(BATCH_SIZE/2),int(BATCH_SIZE/2),count+1)
       plt.imshow(im)
       plt.imshow(np.argmax(lab,-1), cmap=plt.cm.bwr, alpha=0.5, vmin=0, vmax=11)
       #plt.imshow(lab, cmap=plt.cm.bwr, alpha=0.5, vmin=0, vmax=9)
       plt.axis('off')
       L.append(np.unique(np.argmax(lab,-1)))

  plt.show()

print(np.round(np.unique(np.hstack(L))))

val_ds = get_validation_dataset()


# model building and training

model = res_unet((TARGET_SIZE, TARGET_SIZE, 3), BATCH_SIZE, 'multiclass', nclasses)
# model.compile(optimizer = 'adam', loss = tf.keras.losses.CategoricalHinge(), metrics = [mean_iou])
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [mean_iou, dice_coef])

earlystop = EarlyStopping(monitor="val_loss",
                              mode="min", patience=patience)

model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss',
                                verbose=0, save_best_only=True, mode='min',
                                save_weights_only = True)

lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=True)

callbacks = [model_checkpoint, earlystop, lr_callback]


# model.fit(train_ds, steps_per_epoch=steps_per_epoch, epochs=MAX_EPOCHS,
#                       validation_data=val_ds, validation_steps=validation_steps,
#                       callbacks=callbacks)

history = model.fit(train_ds, steps_per_epoch=steps_per_epoch, epochs=MAX_EPOCHS,
                      validation_data=val_ds, validation_steps=validation_steps,
                      callbacks=callbacks)

plot_seg_history_iou(history, hist_fig)

plt.close('all')
K.clear_session()


scores = model.evaluate(val_ds, steps=validation_steps)

print('loss={loss:0.4f}, Mean IOU={iou:0.4f}'.format(loss=scores[0], iou=scores[1]))

# loss=0.8154, Mean IOU=0.9856




#====================================================================
## revised 3d model

# VALIDATION_SPLIT = 0.6
BATCH_SIZE = 8

filepath = os.getcwd()+os.sep+'results/dunes_11class_best_weights_3dmodel_revA.h5'

hist_fig = os.getcwd()+os.sep+'results/dunes_11class_3dmodel_revA.png'


filenames = sorted(tf.io.gfile.glob(os.getcwd()+os.sep+'tfrecords/threeD/dunes3d*.tfrec'))

nb_images = ims_per_shard * len(filenames)
print(nb_images)

split = int(len(filenames) * VALIDATION_SPLIT)

training_filenames = filenames[split:]
validation_filenames = filenames[:split]

validation_steps = int(nb_images // len(filenames) * len(validation_filenames)) // BATCH_SIZE
steps_per_epoch = int(nb_images // len(filenames) * len(training_filenames)) // BATCH_SIZE


train_ds = get_training_dataset()

val_ds = get_validation_dataset()
K.clear_session()


## model building and training
model = res_unet((TARGET_SIZE, TARGET_SIZE, 3), BATCH_SIZE, 'multiclass', nclasses)
# model.compile(optimizer = 'adam', loss = tf.keras.losses.CategoricalHinge(), metrics = [mean_iou, dice_coef])
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [mean_iou, dice_coef])

earlystop = EarlyStopping(monitor="val_loss",
                              mode="min", patience=patience)

model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss',
                                verbose=0, save_best_only=True, mode='min',
                                save_weights_only = True)

lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=True)

callbacks = [model_checkpoint, earlystop, lr_callback]

K.clear_session()

#warm-up
# model.fit(train_ds, steps_per_epoch=steps_per_epoch, epochs=MAX_EPOCHS,
#                       validation_data=val_ds, validation_steps=validation_steps,
#                       callbacks=callbacks)

history = model.fit(train_ds, steps_per_epoch=steps_per_epoch, epochs=MAX_EPOCHS,
                      validation_data=val_ds, validation_steps=validation_steps,
                      callbacks=callbacks)

plot_seg_history_iou(history, hist_fig)

plt.close('all')
K.clear_session()

history = model.fit(train_ds, steps_per_epoch=steps_per_epoch, epochs=MAX_EPOCHS,
                      validation_data=val_ds, validation_steps=validation_steps,
                      callbacks=callbacks)

plot_seg_history_iou(history, hist_fig)

plt.close('all')
K.clear_session()


scores = model.evaluate(val_ds, steps=validation_steps)

print('loss={loss:0.4f}, Mean IOU={iou:0.4f}'.format(loss=scores[0], iou=scores[1]))

# loss=0., Mean IOU=0.




#====================================================================
## revised 3d model

BATCH_SIZE = 9

filepath = os.getcwd()+os.sep+'results/dunes_11class_best_weights_3dmodel_revB.h5'

hist_fig = os.getcwd()+os.sep+'results/dunes_11class_3dmodel_revB.png'


filenames = sorted(tf.io.gfile.glob(os.getcwd()+os.sep+'tfrecords/threeD/dunes3d*.tfrec'))

nb_images = ims_per_shard * len(filenames)
print(nb_images)

split = int(len(filenames) * VALIDATION_SPLIT)

training_filenames = filenames[split:]
validation_filenames = filenames[:split]

validation_steps = int(nb_images // len(filenames) * len(validation_filenames)) // BATCH_SIZE
steps_per_epoch = int(nb_images // len(filenames) * len(training_filenames)) // BATCH_SIZE


train_ds = get_training_dataset()

val_ds = get_validation_dataset()
K.clear_session()


## model building and training
model = res_unet((TARGET_SIZE, TARGET_SIZE, 3), BATCH_SIZE, 'multiclass', nclasses)
# model.compile(optimizer = 'adam', loss = tf.keras.losses.CategoricalHinge(), metrics = [mean_iou, dice_coef])
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [mean_iou, dice_coef])

earlystop = EarlyStopping(monitor="val_loss",
                              mode="min", patience=patience)

model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss',
                                verbose=0, save_best_only=True, mode='min',
                                save_weights_only = True)

lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=True)

callbacks = [model_checkpoint, earlystop, lr_callback]

K.clear_session()

#warm-up
model.fit(train_ds, steps_per_epoch=steps_per_epoch, epochs=MAX_EPOCHS,
                      validation_data=val_ds, validation_steps=validation_steps,
                      callbacks=callbacks)

history = model.fit(train_ds, steps_per_epoch=steps_per_epoch, epochs=MAX_EPOCHS,
                      validation_data=val_ds, validation_steps=validation_steps,
                      callbacks=callbacks)

plot_seg_history_iou(history, hist_fig)

plt.close('all')
K.clear_session()

history = model.fit(train_ds, steps_per_epoch=steps_per_epoch, epochs=MAX_EPOCHS,
                      validation_data=val_ds, validation_steps=validation_steps,
                      callbacks=callbacks)

plot_seg_history_iou(history, hist_fig)

plt.close('all')
K.clear_session()


scores = model.evaluate(val_ds, steps=validation_steps)

print('loss={loss:0.4f}, Mean IOU={iou:0.4f}'.format(loss=scores[0], iou=scores[1]))

# loss=0., Mean IOU=0.






# sample_data_path = os.getcwd()+os.sep+'data/dunes/images/files'
# test_samples_fig = os.getcwd()+os.sep+'dunes_sample_16class_est16samples.png'
# sample_label_data_path = os.getcwd()+os.sep+'data/dunes/labels/files'
#
# sample_filenames = sorted(tf.io.gfile.glob(sample_data_path+os.sep+'*.jpg'))
# sample_label_filenames = sorted(tf.io.gfile.glob(sample_label_data_path+os.sep+'*.jpg'))
#




sample_data_path = os.getcwd()+os.sep+'data/image_sample'

# test_samples_fig = os.getcwd()+os.sep+'dunes2d_sample_16class_est16samples.png'

sample_filenames = sorted(tf.io.gfile.glob(sample_data_path+os.sep+'*.jpg'))


from matplotlib.colors import ListedColormap
cmap = ListedColormap(["#000000", "#3366CC", "#DC3912", "#FF9900", "#109618", "#990099", "#0099C6", "#DD4477", "#66AA00"])

plt.figure(figsize=(24,24))

for counter,f in enumerate(sample_filenames):
    # image = seg_file2tensor(f)/255

    dem = seg_file2tensor(f.replace('images','dems').replace('ortho','dem'))/255

    # merged = np.dstack((image.numpy(), dem.numpy()[:,:,0]))

    est_label = model.predict(tf.expand_dims(dem, 0) , batch_size=1).squeeze()

    est_label = tf.argmax(est_label, axis=-1)

    plt.subplot(4,4,counter+1)
    name = sample_filenames[counter].split(os.sep)[-1].split('.jpg')[0]
    plt.title(name, fontsize=10)
    plt.imshow(dem, cmap=plt.cm.gray)

    plt.imshow(est_label, alpha=0.5, cmap=cmap, vmin=0, vmax=8)

    plt.axis('off')

# plt.show()
plt.savefig(test_samples_fig,
            dpi=200, bbox_inches='tight')
plt.close('all')
