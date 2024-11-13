import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import numpy as np

# GPUメモリを動的に確保
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def create_model(num_classes, input_shape=(128, 128, 1)):
    """
    VGG19ベースの転移学習モデルを作成
    """
    # VGG19をベースモデルとして読み込み（全結合層は除く）
    base_model = VGG19(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # ベースモデルの重みを凍結
    base_model.trainable = False
    
    # 新しいモデルを構築
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def prepare_data(train_dir, valid_dir, batch_size=32, img_size=224):
    """
    データ拡張とデータジェネレータの設定
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, valid_generator

def train_model(model, train_generator, valid_generator, epochs=20):
    """
    モデルのコンパイルと訓練
    """
    # モデルをコンパイル
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Early Stoppingの設定
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # モデルの訓練
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=valid_generator,
        callbacks=[early_stopping]
    )

    return history

def evaluate_model(model, test_generator):
    """
    モデルの評価
    """
    results = model.evaluate(test_generator)
    print(f'Test Loss: {results[0]:.4f}')
    print(f'Test Accuracy: {results[1]:.4f}')

# 使用例
if __name__ == "__main__":
    # パラメータの設定
    NUM_CLASSES = 5  # クラス数を指定
    BATCH_SIZE = 32
    IMG_SIZE = 128
    EPOCHS = 20

    # データディレクトリのパスを指定
    TRAIN_DIR = 'path/to/train'
    VALID_DIR = 'path/to/validation'
    TEST_DIR = 'path/to/test'

    # データの準備
    train_generator, valid_generator = prepare_data(
        TRAIN_DIR,
        VALID_DIR,
        BATCH_SIZE,
        IMG_SIZE
    )

    # テストデータの準備
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # モデルの作成
    model = create_model(NUM_CLASSES)

    # モデルの訓練
    history = train_model(model, train_generator, valid_generator, EPOCHS)

    # モデルの評価
    evaluate_model(model, test_generator)

    # モデルの保存
    model.save('vgg19_multiclass_model.h5')