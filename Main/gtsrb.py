gtsrb_ds = tf.keras.utils.image_dataset_from_directory(
    gtsrb_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle=True,
    seed=SEED
)

gtsrb_class_names = gtsrb_ds.class_names
num_classes = len(gtsrb_class_names)


gtsrb_ds.take(1)


# gtsrb_features

gtsrb_features = []
gtsrb_targets = []

for images, lbls in gtsrb_ds:
    feat = feature_extractor(images)
    gtsrb_features.append(feat.numpy())
    gtsrb_targets.append(lbls.numpy())

gtsrb_X = np.concatenate(gtsrb_features, axis=0)
gtsrb_y = np.concatenate(gtsrb_targets, axis=0)

# Train/Test Split
gtsrb_X_train, gtsrb_X_val, gtsrb_y_train, gtsrb_y_val = train_test_split(
    gtsrb_X,
    gtsrb_y, 
    test_size=0.15, 
    random_state=SEED
)



optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate,
    # decay=0.004,
    # use_ema=True,
    epsilon=1e-07
)

classifier.compile(
    optimizer=optimizer,
    loss=criterion,
    metrics=[
        "accuracy", # tf.keras.metrics.Accuracy(name='accuracy'),
        # tf.keras.metrics.AUC(name='AUC', multi_label=True),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tfa.metrics.F1Score(43, name='f1', ),
        # tfa.metrics.CohenKappa(num_classes=len(class_names))
    ]
)
gtsrb_history = classifier.fit(
    gtsrb_X_train, gtsrb_y_train,
    validation_data=(gtsrb_X_val, gtsrb_y_val),
    batch_size=batch_size,
    epochs=30,
    # callbacks=callbacks
)


gtsrb_y_train[4]


