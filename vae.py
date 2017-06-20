import tensorflow as tf
import numpy as np
import time
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm

from tensorflow.examples.tutorials.mnist import input_data


class VAE(object):
    def __init__(self, n_batch=100, n_input=784, n_latent=2, n_hidden=500, learning_rate=0.001, stddev_init=0.1,
                 bias_init=0.1, n_epoch=100, seed=0, checkpoint_path='checkpoints/model.ckpt'):
        self.n_batch = n_batch
        self.n_input = n_input
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.stddev_init = stddev_init
        self.bias_init = bias_init
        self.n_epoch = n_epoch
        self.seed = seed
        self.checkpoint_path = checkpoint_path

        self.graph = self._create_graph()
        self.learning_curve = {'train': [], 'val': []}

    def _create_graph(self):
        with tf.Graph().as_default() as graph:
            tf.set_random_seed(self.seed)
            self.loss = self._create_model()
            self.optimizer = self._create_optimizer(self.loss, self.learning_rate)
            self.initializer = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            graph.finalize()
        return graph

    def _restore_model(self, session):
        self.saver.restore(session, self.checkpoint_path)

    def _create_weights(self, shape):
        W = tf.Variable(tf.random_normal(shape, stddev=self.stddev_init))
        b = tf.Variable(tf.constant(self.bias_init, shape=[shape[1]], dtype=tf.float32))
        return W, b

    def _create_encoder(self, x):
        with tf.variable_scope('encoder'):
            W3, b3 = self._create_weights([self.n_input, self.n_hidden])
            W4, b4 = self._create_weights([self.n_hidden, self.n_latent])
            W5, b5 = self._create_weights([self.n_hidden, self.n_latent])

            h = tf.nn.relu(x @ W3 + b3)
            mu = h @ W4 + b4
            log_sigma_squared = h @ W5 + b5
            sigma_squared = tf.exp(log_sigma_squared)
            sigma = tf.sqrt(sigma_squared)
        return mu, log_sigma_squared, sigma_squared, sigma

    def _create_decoder(self, z):
        with tf.variable_scope('decoder'):
            W1, b1 = self._create_weights([self.n_latent, self.n_hidden])
            W2, b2 = self._create_weights([self.n_hidden, self.n_input])

            y_logit = tf.nn.relu(z @ W1 + b1) @ W2 + b2
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
        loss = tf.reduce_mean(regularizer + recon_error)
        return loss

    def _create_optimizer(self, loss, learning_rate):
        batch = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch)
        return optimizer

    def _compute_loss(self, session, batch, optimize=True):
        epsilon = np.random.randn(self.n_latent).astype(np.float32)
        ops = [self.loss, self.optimizer] if optimize else [self.loss]
        loss = session.run(ops, {self.x: batch[0], self.epsilon: epsilon})[0]
        return loss

    def fit(self, train, validation):
        session = tf.Session(graph=self.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(session, coord)
        session.run(self.initializer)
        n_step = train.num_examples // self.n_batch * self.n_epoch
        start = time.time()
        np.random.seed(self.seed)

        try:
            self.learning_curve['train'].clear()
            self.learning_curve['val'].clear()
            loss_train = 0.
            loss_val = 0.
            for step in range(n_step):
                loss_train += self._compute_loss(session, train.next_batch(self.n_batch))
                loss_val += self._compute_loss(session, validation.next_batch(self.n_batch), optimize=False)

                if (step * self.n_batch) % train.num_examples == 0 and step > 0:
                    train_error = self.n_batch / train.num_examples * loss_train
                    val_error = self.n_batch / train.num_examples * loss_val
                    self.learning_curve['train'] += [-train_error]
                    self.learning_curve['val'] += [-val_error]
                    loss_train = 0.
                    loss_val = 0.
                    print('epoch: {:2d}, step: {:5d}, training error: {:03.4f}, '
                          'validation error: {:03.4f}, time elapsed: {:4.0f} s'
                          .format(train.epochs_completed, step, train_error, val_error, time.time() - start))
        except KeyboardInterrupt:
            print('ending training')
        finally:
            self.saver.save(session, self.checkpoint_path)
            session.close()
            coord.request_stop()
            coord.join(threads)
            print('finished training')
        return self

    def decode(self, z):
        with tf.Session(graph=self.graph) as session:
            self._restore_model(session)
            y = session.run(self.y, {self.z: z})
        return y

    def encode(self, x):
        with tf.Session(graph=self.graph) as session:
            self._restore_model(session)
            mu = session.run(self.mu, {self.x: x})
        return mu

    def mosaic(self, grid_width=20):
        epsilon = norm.ppf(np.linspace(0, 1, grid_width + 2)[1:-1])
        z = np.dstack(np.meshgrid(epsilon, -epsilon)).reshape(-1, 2)
        image_width = int(np.sqrt(self.n_input))
        images = np.reshape(self.decode(z), [grid_width, grid_width, image_width, image_width])
        mos = 1 - np.bmat(images.tolist())
        return mos

if __name__ == '__main__':
    data = input_data.read_data_sets('data')
    vae = VAE()
    #vae.fit(data.train, data.validation)
    latent_space = vae.encode(data.test.images)
    mosaic = vae.mosaic()

    sns.set_color_codes()
    sns.set_style('white')
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))

    ax1.imshow(mosaic, cmap='gray')
    ax1.set_title('Learned MNIST manifold')
    ax1.axis('off')

    cmap = mpl.cm.get_cmap('tab10')
    for c in range(10):
        idx = (data.test.labels == c)
        ax2.scatter(latent_space[idx, 0], latent_space[idx, 1], c=cmap(c), marker='.', label=str(c), alpha=0.7)
    ax2.set_title('Latent representation')
    ax2.axis('square')
    ax2.legend(frameon=True)
    sns.despine()

    epochs = range(len(vae.learning_curve['train']))
    ax3.plot(epochs, vae.learning_curve['train'], 'b-', label='ELBO training')
    ax3.plot(epochs, vae.learning_curve['val'], 'r-', label='ELBO validation')
    ax3.set_title('Learning curve')
    ax3.set_xlabel('epoch')
    ax3.set_ylabel('ELBO')
    ax3.axis('square')
    ax3.legend(frameon=True)
    sns.despine()
    f.tight_layout()
    plt.savefig('vae.pdf', bbox_inches='tight')
    plt.show()
