import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import norm

from tensorflow.examples.tutorials.mnist import input_data


class VAE(object):
    def __init__(self, n_batch=100, n_input=784, n_latent=2, n_hidden=500, learning_rate=0.01,
                 stddev_init=0.1, weight_decay_factor=1e-4, n_step=10**6, seed=0, checkpoint_path='model.ckpt'):
        self.n_batch = n_batch
        self.n_input = n_input
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.stddev_init = stddev_init
        self.weight_decay_factor = weight_decay_factor
        self.n_step = n_step
        self.seed = seed
        self.checkpoint_path = checkpoint_path

    def _create_graph(self, mode):
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        with tf.Graph().as_default() as graph:
            self.loss = self._create_model(mode)
            self.optimizer = self._create_optimizer(self.loss, self.learning_rate)
            self.initializer = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            graph.finalize()
        return graph

    def _create_weights(self, shape):
        W = tf.Variable(tf.random_normal(shape, stddev=self.stddev_init))
        b = tf.Variable(np.zeros([shape[0], shape[1], 1]).astype(np.float32))
        return W, b

    def _create_weight_decay(self):
        return self.weight_decay_factor / 2.0 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

    def _create_encoder(self, x):
        with tf.variable_scope('encoder'):
            W3, b3 = self._create_weights([self.n_batch, self.n_hidden, self.n_input])
            W4, b4 = self._create_weights([self.n_batch, self.n_latent, self.n_hidden])
            W5, b5 = self._create_weights([self.n_batch, self.n_latent, self.n_hidden])

            h = tf.tanh(W3 @ x + b3)
            mu = W4 @ h + b4
            log_sigma_squared = W5 @ h + b5
            sigma_squared = tf.exp(log_sigma_squared)
            sigma = tf.sqrt(sigma_squared)
        return mu, log_sigma_squared, sigma_squared, sigma

    def _create_decoder(self, z):
        with tf.variable_scope('decoder'):
            W1, b1 = self._create_weights([self.n_batch, self.n_hidden, self.n_latent])
            W2, b2 = self._create_weights([self.n_batch, self.n_input, self.n_hidden])

            y_logit = W2 @ tf.tanh(W1 @ z + b1) + b2
            y = tf.sigmoid(y_logit)
        return y_logit, y

    def _create_model(self, mode):
        self.x = tf.placeholder(tf.float32, [self.n_batch, self.n_input])
        x = tf.reshape(self.x, [self.n_batch, self.n_input, 1])
        mu, log_sigma_squared, sigma_squared, sigma = self._create_encoder(x)

        if mode == 'fit':
            self.epsilon = tf.placeholder(tf.float32, self.n_latent)
            epsilon = tf.reshape(self.epsilon, [1, self.n_latent, 1])
            z = mu + sigma * epsilon
        else:
            self.z = tf.placeholder(tf.float32, [self.n_batch, self.n_latent])
            z = tf.reshape(self.z, [self.n_batch, self.n_latent, 1])
        y_logit, self.y = self._create_decoder(z)

        regularizer = -0.5 * tf.reduce_sum(1 + log_sigma_squared - tf.square(mu) - sigma_squared, 1)
        recon_error = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=x), 1)
        weight_decay = self._create_weight_decay()
        loss = tf.reduce_mean(regularizer) + tf.reduce_mean(recon_error) + weight_decay
        return loss

    def _create_optimizer(self, loss, learning_rate):
        batch = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=batch)
        return optimizer

    def fit(self, X):
        graph = self._create_graph(mode='fit')
        session = tf.Session(graph=graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(session, coord)

        try:
            session.run(self.initializer)
            start = time.time()
            for step in range(self.n_step):
                x = X.next_batch(self.n_batch)[0]
                epsilon = np.random.randn(self.n_latent).astype(np.float32)
                loss, _ = session.run([self.loss, self.optimizer], {self.x: x, self.epsilon: epsilon})

                if step % 100 == 0:
                    print('step: %d, mini-batch error: %1.4f, took %ds' % (step, loss, (time.time()-start)))
                    start = time.time()

        except KeyboardInterrupt:
            print('ending training')
        finally:
            self.saver.save(session, self.checkpoint_path)
            session.close()
            coord.request_stop()
            coord.join(threads)
            print('finished training')

    def _sample_2d_grid(self, grid_width):
        epsilon_x = norm.ppf(np.linspace(0, 1, grid_width + 2)[1:-1])
        epsilon_y = epsilon_x.copy()
        epsilon = np.dstack(np.meshgrid(epsilon_x, epsilon_y)).reshape(-1, 2)
        return epsilon

    def decode(self, z, n_batch=None):
        graph = self._create_graph(mode='decode')
        with tf.Session(graph=graph) as session:
            self.saver.restore(session, self.checkpoint_path)
            n_iter = n_batch // self.n_batch if n_batch is not None else 1
            y = [session.run(self.y, {self.z: z[i*self.n_batch:(i+1)*self.n_batch]}) for i in range(n_iter)]
        return y

    def mosaic(self, grid_width=20):
        z = self._sample_2d_grid(grid_width)
        w = int(np.sqrt(self.n_input))
        mos = np.bmat(np.reshape(self.decode(z, n_batch=grid_width ** 2), [grid_width, grid_width, w, w]).tolist())
        return mos

if __name__ == '__main__':
    data = input_data.read_data_sets('MNIST data')
    vae = VAE()
    vae.fit(data.train)
    mosaic = vae.mosaic()
    plt.imshow(mosaic, cmap='gray')
    plt.axis('off')
    plt.show()