import tensorflow as tf
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

from tensorflow.examples.tutorials.mnist import input_data

class VAE(object):
    def __init__(self, n_batch=100, n_input=784, n_latent=2, n_hidden=500, learning_rate=0.001,
                 stddev_init=0.1, weight_decay_factor=0, n_epoch=100, seed=0, activation=tf.tanh,
                 checkpoint_path='model.ckpt'):
        self.n_batch = n_batch
        self.n_input = n_input
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.stddev_init = stddev_init
        self.weight_decay_factor = weight_decay_factor
        self.n_epoch = n_epoch
        self.seed = seed
        self.checkpoint_path = checkpoint_path
        self.activation = activation

    def _create_graph(self):
        with tf.Graph().as_default() as graph:
            tf.set_random_seed(self.seed)
            self.loss = self._create_model()
            self.optimizer = self._create_optimizer(self.loss, self.learning_rate)
            self.initializer = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            graph.finalize()
        return graph

    def _create_weights(self, shape):
        W = tf.Variable(tf.random_normal(shape, stddev=self.stddev_init))
        b = tf.Variable(0.1*np.ones(shape[1]).astype(np.float32))
        return W, b

    def _create_weight_decay(self):
        if self.weight_decay_factor > 0:
            return self.weight_decay_factor / 2.0 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        else:
            return 0

    def _create_encoder(self, x):
        with tf.variable_scope('encoder'):
            W3, b3 = self._create_weights([self.n_input, self.n_hidden])
            W4, b4 = self._create_weights([self.n_hidden, self.n_latent])
            W5, b5 = self._create_weights([self.n_hidden, self.n_latent])

            h = self.activation(x @ W3 + b3)
            mu = h @ W4 + b4
            log_sigma_squared = h @ W5 + b5
            sigma_squared = tf.exp(log_sigma_squared)
            sigma = tf.sqrt(sigma_squared)
        return mu, log_sigma_squared, sigma_squared, sigma

    def _create_decoder(self, z):
        with tf.variable_scope('decoder'):
            W1, b1 = self._create_weights([self.n_latent, self.n_hidden])
            W2, b2 = self._create_weights([self.n_hidden, self.n_input])

            y_logit = self.activation(z @ W1 + b1) @ W2 + b2
            y = tf.sigmoid(y_logit)
        return y_logit, y

    def _create_model(self):
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.mu, log_sigma_squared, sigma_squared, sigma = self._create_encoder(self.x)

        self.epsilon = tf.placeholder(tf.float32, self.n_latent)
        self.z = self.mu + sigma * self.epsilon
        y_logit, self.y = self._create_decoder(self.z)

        regularizer = -0.5 * tf.reduce_sum(1 + log_sigma_squared - tf.square(self.mu) - sigma_squared, 1)
        recon_error = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=self.x), 1)
        weight_decay = self._create_weight_decay()
        loss = tf.reduce_mean(regularizer + recon_error) + weight_decay
        return loss

    def _create_optimizer(self, loss, learning_rate):
        self.current_step = tf.Variable(0, trainable=False)
        #optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=batch)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=self.current_step)
        return optimizer

    def _compute_loss(self, session, batch, optimize=True):
        epsilon = np.random.randn(self.n_latent).astype(np.float32)
        ops = [self.loss, self.optimizer] if optimize else [self.loss]
        loss = session.run(ops, {self.x: batch[0], self.epsilon: epsilon})[0]
        return loss

    def fit(self, train, validation, refit=True):
        graph = self._create_graph()
        session = tf.Session(graph=graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(session, coord)
        session.run(self.initializer) if refit else self.saver.restore(session, self.checkpoint_path)
        initial_step = session.run(self.current_step)
        n_step = train.num_examples // self.n_batch * self.n_epoch
        start = time.time()
        np.random.seed(self.seed)

        try:
            self.learning_curve = {'train': [], 'val': []}
            loss_train = 0.
            loss_val = 0.
            for step in range(initial_step, n_step):
                loss_train += self._compute_loss(session, train.next_batch(self.n_batch))
                loss_val += self._compute_loss(session, validation.next_batch(self.n_batch), optimize=False)

                if (step * self.n_batch) % train.num_examples == 0 and step > 0:
                    train_error = self.n_batch / train.num_examples * loss_train
                    val_error = self.n_batch / train.num_examples * loss_val
                    self.learning_curve['train'] += [train_error]
                    self.learning_curve['val'] += [val_error]
                    loss_train = 0.
                    loss_val = 0.
                    print('epoch: %d, step: %d, training error: %1.4f, validation error: %1.4f, time elapsed %ds' % (
                    train.epochs_completed, step, train_error, val_error, time.time()-start))
        except KeyboardInterrupt:
            print('ending training')
        finally:
            self.saver.save(session, self.checkpoint_path)
            session.close()
            coord.request_stop()
            coord.join(threads)
            print('finished training')

    def _sample_2d_grid(self, grid_width):
        epsilon = norm.ppf(np.linspace(0, 1, grid_width + 2)[1:-1])
        epsilon_2d = np.dstack(np.meshgrid(epsilon, -epsilon)).reshape(-1, 2)
        return epsilon_2d

    def decode(self, z):
        graph = self._create_graph()
        with tf.Session(graph=graph) as session:
            self.saver.restore(session, self.checkpoint_path)
            y = session.run(self.y, {self.z: z})
        return y

    def encode(self, x):
        graph = self._create_graph()
        with tf.Session(graph=graph) as session:
            self.saver.restore(session, self.checkpoint_path)
            y = session.run(self.mu, {self.x: x})
        return y

    def mosaic(self, grid_width=20):
        z = self._sample_2d_grid(grid_width)
        image_width = int(np.sqrt(self.n_input))
        images = np.reshape(self.decode(z), [grid_width, grid_width, image_width, image_width])
        mos = 1 - np.bmat(images.tolist())
        return mos

if __name__ == '__main__':
    data = input_data.read_data_sets('MNIST data')
    vae = VAE(activation=tf.nn.relu)
    vae.fit(data.train, data.test, refit=True)
    latent_space = vae.encode(data.test.images)
    mosaic = vae.mosaic()

    sns.set_color_codes()
    sns.set_style('white')
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.imshow(mosaic, cmap='gray')
    ax1.set_title('Learned MNIST manifold')
    ax1.axis('off')
    sc = ax2.scatter(latent_space[:, 0], latent_space[:, 1], c=data.test.labels, marker='.', cmap='tab10')
    ax2.set_title('Latent representation')
    ax2.axis('equal')
    f.colorbar(sc, ax=ax2)
    sns.despine()
    plt.show()

    plt.figure()
    epochs = range(len(vae.learning_curve['train']))
    plt.plot(epochs, vae.learning_curve['train'], 'r-', label='training loss')
    plt.plot(epochs, vae.learning_curve['val'], 'b-', label='validation loss')
    plt.title('Learning curve')
    plt.ylabel('negative ELBO')
    plt.xlabel('epoch')
    sns.despine()
    plt.show()