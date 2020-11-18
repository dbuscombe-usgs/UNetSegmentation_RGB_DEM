

from imports import *

def get_batched_dataset(filenames):

    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = True

    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTO)
    dataset = dataset.map(read_seg_tfrecord_4ddunes, num_parallel_calls=AUTO)

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

BATCH_SIZE = 8
#====================================================================
## initial 4d model

# data_path= os.getcwd()+os.sep+"data"

filepath = os.getcwd()+os.sep+'results/dunes_11class_best_weights_4dmodel.h5'

hist_fig = os.getcwd()+os.sep+'results/dunes_11class_4dmodel.png'


filenames = sorted(tf.io.gfile.glob(os.getcwd()+os.sep+'tfrecords/fourD/dunes4d*.tfrec'))

nb_images = ims_per_shard * len(filenames)
print(nb_images)

split = int(len(filenames) * VALIDATION_SPLIT)

training_filenames = filenames[split:]
validation_filenames = filenames[:split]

validation_steps = int(nb_images // len(filenames) * len(validation_filenames)) // BATCH_SIZE
steps_per_epoch = int(nb_images // len(filenames) * len(training_filenames)) // BATCH_SIZE

train_ds = get_training_dataset()


val_ds = get_validation_dataset()



model = res_unet((TARGET_SIZE, TARGET_SIZE, 4), BATCH_SIZE, 'multiclass', nclasses)
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

# loss=0.7667, Mean IOU=0.9844




filepath = os.getcwd()+os.sep+'results/dunes_11class_best_weights_4dmodel-revA.h5'

hist_fig = os.getcwd()+os.sep+'results/dunes_11class_4dmodel-revA.png'

BATCH_SIZE = 9

model = res_unet((TARGET_SIZE, TARGET_SIZE, 4), BATCH_SIZE, 'multiclass', nclasses)
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

# loss=0.7667, Mean IOU=0.9844
