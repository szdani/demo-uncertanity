import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


MODEL_LOC = '../../data/models/dropout'

def save_model(model):
    model.save(MODEL_LOC)
    
def load_model():
    return tf.keras.models.load_model(MODEL_LOC)

def eval(model, dataset, nb_of_evals=25, gradcam_target_layer=None):
    df = pd.DataFrame(columns=['gold', 'predicted', 'mean', 'std', 'img', 'gradcam_img'])
    ds = dataset.as_numpy_iterator()
    for idx, item in enumerate(ds):
        img = item[0].reshape((1,) + item[0].shape)
        multiple_imgs = np.repeat(img, nb_of_evals, axis=0)
        
        heatmap = None
        if gradcam_target_layer is not None:
            heatmap, raw_results = grad_cam(model, 
                                            multiple_imgs, item[1], 
                                            layer_name=gradcam_target_layer, 
                                            nb_of_evals=20)
        else:
            raw_results = model.predict(multiple_imgs)
        
        mean = np.mean(raw_results, axis=0)
        std = np.std(raw_results, axis=0)
        max_idx_on_mean = np.argmax(mean)
        
        df = df.append({
            'img': img,
            'gold': item[1],
            'predicted': max_idx_on_mean,
            'mean': mean,
            'std': std,
            'heatmap': heatmap
        }, ignore_index=True)
        
    return df  

def grad_cam(model, img, class_index, layer_name, nb_of_evals=20):
    img = np.repeat(img, nb_of_evals, axis=0)
    
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as t:
        conv_outputs, predictions = grad_model(img.astype(np.float32))
        loss = predictions[:,class_index]
        
     # Filter and grads
    output = conv_outputs[0]
    grads = t.gradient(loss, conv_outputs)[0]  
    
    # Guided backprop
    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads
    
    # Average grads spacially
    weights = tf.reduce_mean(grads, axis=(0, 1))
    
    # Placeholder for aggregation
    cam = np.ones(output.shape[0:2], dtype=np.float32)
    for index, w in enumerate(weights):
        cam += w * output[:, :, index]
    cam = cv2.resize(cam.numpy(), (28, 28))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())
    
    return heatmap, predictions
        
def visualize_example(df, target_class=None, nb_of_pos_examples=2, nb_of_neg_examples=2):
    '''Plotting results from pandas dataframe.
    Args:
    - df: padnas dataframe columns=['gold', 'predicted', 'mean', 'std', 'img', 'gradcam_img']
    - target_class: If not none then only this class will be plotted.
    - nb_of_pos_examples: Integer, number of good predictions to show
    - nb_of_neg_examples: Integer, number of failed predictions to show
    '''
    def _vis(df):
        '''Inner function responsible for plotting results.
        If heatmap is available (it's made during evaulation, and it's not a nan array) then gradCam
        visualization will be also plotted.
        3 subplots:
        - 1.: Original image.
        - 2.: (OPTIONAL) GradCam Heatmap if present.
        - 3.: Class predictions with standard deviations.
        '''
        for idx, item in df.iterrows():    
            print('______________________________________________')
            print("Gold/Predicted value: {} : {}".format(item['gold'], item['predicted']))
            print("Std on prediction: {}".format(item['std'][item['predicted']]))
            
            # If model is not none - calc gradcam
            if item['heatmap'] is not None and ~np.isnan(item['heatmap']).all():
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3,
                                                    figsize=(15,3),
                                                    gridspec_kw={'width_ratios': [1, 1, 3]})
                fig.tight_layout()
                mappable = ax2.imshow(item['heatmap'], cmap='jet', vmin=0, vmax=1)
                fig.colorbar(mappable, ax=ax2, shrink=0.95)

            else: # Else plot without gradcam
                fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(15,2))
            
            mappable = ax1.imshow(item['img'][0,:,:,0], cmap='gray', vmin=0, vmax=1)
            fig.colorbar(mappable, ax=ax1, shrink=0.95)
            
            ax3.scatter(range(0,10),item['mean'], s=80, marker='o')
            ax3.scatter(item['gold'], 0.0, c='red', marker='x', s=100)
            ax3.errorbar(range(0,10), item['mean'] ,yerr=item['std']/2, linestyle="None", c='midnightblue')
            ax3.set_title('Model predictions\nPredicted score with stdev with blue, gold class with red cross.')
            ax3.set_xlabel("Classes")
            ax3.set_ylabel("Predicted class scores")
            
            plt.tight_layout()
            plt.show()
    
    if target_class is not None:
        df = df[df['gold']==target_class]
    
    pos = df['gold'] == df['predicted']
    neg = ~pos
    
    pos = df[pos].sample(nb_of_pos_examples)
    neg = df[neg].sample(nb_of_neg_examples)
    _vis(pos)
    _vis(neg)

if __name__ == "__main__":
    # Dataset

    TRAIN_VAL_SPLIT_PERCENTAGE = 80 # 20
    BATCH = 25
    SHUFFLE_BUFFER_SIZE = 512

    dataset_builder = tfds.builder(name="mnist")
    dataset_builder.download_and_prepare()
    info = dataset_builder.info
    nb_train_examples = info.splits['train'].num_examples * (TRAIN_VAL_SPLIT_PERCENTAGE/100)
    nb_val_examples = info.splits['train'].num_examples * ((100-TRAIN_VAL_SPLIT_PERCENTAGE)/100)
    nb_test_examples = info.splits['test'].num_examples

    # as_supervised is needed otherwise we need to work with dictionaries
    ds_train = dataset_builder.as_dataset(split="train[:{}%]".format(TRAIN_VAL_SPLIT_PERCENTAGE),
                                          shuffle_files=True,
                                          batch_size=BATCH,
                                          as_supervised=True)
    ds_train = ds_train.repeat()
    # Normalize
    ds_train = ds_train.map(lambda img, label: (tf.cast(img,tf.float32) / 255.0, label))
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_val = dataset_builder.as_dataset(split="train[{}%:]".format(TRAIN_VAL_SPLIT_PERCENTAGE),
                                       as_supervised=True,
                                       batch_size=BATCH,)
    # Normalize
    ds_val = ds_val.map(lambda img, label: (tf.cast(img,tf.float32) / 255.0, label))
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = dataset_builder.as_dataset(split="test", as_supervised=True)
    # Normalize
    ds_test = ds_test.map(lambda img, label: (tf.cast(img,tf.float32) / 255.0, label))

    #viz = dataset_builder.as_dataset(split="test")
    #fig = tfds.show_examples(info, viz)

    # Model
    _input = tf.keras.layers.Input((28,28,1))
    x = tf.keras.layers.Conv2D(8, (3,3))(_input)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x)
    x = tf.keras.layers.Conv2D(16, (3,3))(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x)
    x = tf.keras.layers.Conv2D(32, (3,3))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.Dropout(0.3)(x, training=True)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.Dropout(0.3)(x, training=True)
    output = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=_input, outputs=output)

    opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['acc'])


    model.fit(ds_train, epochs=15, steps_per_epoch=nb_train_examples/BATCH,
             validation_data=ds_val, verbose=2)


    print('Conv layers for GradCam: (Use it for "visualize_example" call.)')
    for layer in model.layers:
        if str.startswith(layer.name, 'conv'):
            print(layer.name)

    # Eval on test set with gradCam
    df = eval(model, ds_test, gradcam_target_layer='conv2d')

    acc = df['gold'] == df['predicted']
    acc = acc.sum() / len(acc)
    print("Accuracy on test set {}".format(acc))

    # Plot example results
    visualize_example(df, target_class=None, nb_of_pos_examples=1, nb_of_neg_examples=2)


    # # MNIST-Corrupted

    def eval_on_mnist_corrupted_subset(model, subset, gradcam_target_layer='conv2d'):
        ds_test = tfds.load("mnist_corrupted/{}".format(subset), split='test', as_supervised=True)
        ds_test = ds_test.map(lambda img, label: (tf.cast(img,tf.float32) / 255.0, label))
        ds_test = ds_test.take(100)

        # Eval
        df_results = eval(model, ds_test, gradcam_target_layer=gradcam_target_layer)
        acc = df_results['gold'] == df_results['predicted']
        acc = acc.sum() / len(acc)
        
        return df_results, acc


    # ## Spatter noise
    df_spatter, acc_spatter = eval_on_mnist_corrupted_subset(model, 'spatter', gradcam_target_layer='conv2d')
    print("Accuracy on MNIST Corrupted (Spatter) test set {}".format(acc_spatter))
    visualize_example(df_spatter, target_class=None, nb_of_pos_examples=1, nb_of_neg_examples=2)

    # ## Fog noise
    df_fog, acc_fog = eval_on_mnist_corrupted_subset(model, 'fog', gradcam_target_layer='conv2d')
    print("Accuracy on MNIST Corrupted (Spatter) test set {}".format(acc_fog))
    visualize_example(df_fog, target_class=None, nb_of_pos_examples=1, nb_of_neg_examples=2)

    # ## Canny Edges
    df_canny, acc_canny = eval_on_mnist_corrupted_subset(model, 'canny_edges', gradcam_target_layer='conv2d')
    print("Accuracy on MNIST Corrupted (Spatter) test set {}".format(acc_canny))
    visualize_example(df_canny, target_class=None, nb_of_pos_examples=1, nb_of_neg_examples=2)


    # ## Brightness
    df_brightness, acc_rightness = eval_on_mnist_corrupted_subset(model, 'brightness', gradcam_target_layer='conv2d')
    print("Accuracy on MNIST Corrupted (Spatter) test set {}".format(acc_rightness))
    visualize_example(df_brightness, target_class=None, nb_of_pos_examples=1, nb_of_neg_examples=2)

