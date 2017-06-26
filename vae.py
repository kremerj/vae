# Jan Kremer, 2017
# Tensorflow implementation of the variational autoencoder, Kingma & Welling, 2014, https://arxiv.org/pdf/1312.6114.pdf

import tensorflow as tf
import numpy as np
import time
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm

from tensorflow.contrib.keras import layers
from tensorflow.examples.tutorials.mnist import input_data


class VAE(object):
    """This class implements the variational autoencoder as described in the paper.

    The parameters are very similar to the ones mentioned there. This implementation deviates from the paper as follows.
    - No weight decay is used as it did not help training much in terms of error, but slowed the process.
    - Instead of tanh, we use relu as the activation function to speed up convergence.
    - We use ADAM instead of AdaGrad to stop the learning rate from decaying too much.

    The remaing architecture and algorithm is as close to the description in the paper as possible.
    """

    def __init__(self, n_batch=100, n_latent=2, n_hidden=500, learning_rate=0.001, n_epoch=100, seed=0,
                 checkpoint_path='checkpoints/model.ckpt'):
        """Initialize the object.

        Args:
            n_batch: Mini-batch size.
            n_latent: Dimensionality of the latent variables.
            n_hidden: Number of hidden neurons.
            learning_rate: The initial learning rate of the optimization algorithm.
            n_epoch: Number of epochs to train the autoencoder.
            seed: A random seed to create reproducible results.
            checkpoint_path: Path to the optimized model parameters.
        """
        self.n_batch = n_batch
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        self.seed = seed
        self.checkpoint_path = checkpoint_path

        self.learning_curve = {'train': [], 'val': []}

    def _create_graph(self, n_input):
        """Creates the computational graph of the model.

        Args:
            n_input: Feature dimension of input data.

        Returns:
            The computational graph.
        """
        with tf.Graph().as_default() as graph:
            tf.set_random_seed(self.seed)
            self.n_input = n_input
            self.loss = self._create_model()
            self.optimizer = self._create_optimizer(self.loss, self.learning_rate)
            self.initializer = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            graph.finalize()  # Finalize the graph to make sure it is not accidentally modified later.
        return graph

    def _restore_model(self, session):
        """Restores the model from the model's checkpoint path.

        Args:
            session: Current session which should hold the graph to which the model parameters are assigned.
        """
        self.saver.restore(session, self.checkpoint_path)

    def _create_encoder(self, x):
        """Creates the encoder network.

        The encoder network produces mean and standard deviation as the output of an MLP which takes an input vector x.
        Given these outputs we can formulate a parameterized random variable z whith density p(z|x) = N(z; mu, sigma).

        Args:
            x: Mini-batch of input vectors.

        Returns:
            mu, log_sigma, sigma_squared and sigma, the mean and different transformations of the std. deviation sigma.

        """
        h = layers.Dense(self.n_hidden, activation='relu')(x)
        mu = layers.Dense(self.n_latent)(h)
        log_sigma_squared = layers.Dense(self.n_latent)(h)

        sigma_squared = tf.exp(log_sigma_squared)
        sigma = tf.sqrt(sigma_squared)
        return mu, log_sigma_squared, sigma_squared, sigma

    def _create_decoder(self, z):
        """Creates the decoder network.

        The decoder produces a reconstruction in the input space y, given a latent code z.

        Args:
            z: Mini-batch of latent variables z to produce examples y in the input space.

        Returns:
            y_logit, the logit of the output y and y, the output variable in [0,1]^D.
        """
        h = layers.Dense(self.n_hidden, activation='relu')(z)
        y_logit = layers.Dense(self.n_input)(h)
        y = tf.sigmoid(y_logit)
        return y_logit, y

    def _create_model(self):
        """Creates the model consisting of encoder and decoder, and returns the loss.

        Returns:
            The loss of the model given examples x and normal noise epsilon.
        """

        self.x = tf.placeholder(tf.float32, [None, self.n_input])  # Holds the mini-batch of input examples.
        self.mu, log_sigma_squared, sigma_squared, sigma = self._create_encoder(self.x)

        self.epsilon = tf.placeholder(tf.float32, self.n_latent)  # Holds the normally distributed input noise.
        self.z = self.mu + sigma * self.epsilon
        y_logit, self.y = self._create_decoder(self.z)

        regularizer = -0.5 * tf.reduce_sum(1 + log_sigma_squared - tf.square(self.mu) - sigma_squared, 1)
        recon_error = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=self.x), 1)
        loss = tf.reduce_mean(regularizer + recon_error)
        return loss

    def _create_optimizer(self, loss, learning_rate):
        """Creates the optimizing operation for a given initial learning rate.

        Uses the ADAM optimizer to adaptively shrink the inital learning rate over time.

        Args:
            learning_rate: The initial learning rate for the optimization algorithm.

        Returns:
            An optimizer operation to minimize the error of the model.
        """
        step = tf.Variable(0, trainable=False)  # Variable to keep track of the current step.
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=step)
        return optimizer

    def _compute_loss(self, session, batch, optimize=True):
        """Computes the loss for a given session and input batch.

        Args:
            session: The session in which the loss should be computed.
            batch: The input batch on which the loss should be computed.
            optimize: Should the weights of the model also be optimized.

        Returns:
            The loss of the model given the supplied mini-batch of data.
        """
        epsilon = np.random.randn(self.n_latent).astype(np.float32)
        ops = [self.loss, self.optimizer] if optimize else [self.loss]
        loss = session.run(ops, {self.x: batch[0], self.epsilon: epsilon})[0]  # Ignore the label of the current batch.
        return loss

    def fit(self, train, validation):
        """Train the autoencoder on the given training data and validate it on the given validation data.

        Args:
            train: Dataset object which holds the training data and can supply the data in mini-batches.
            validation: Dataset object which holds the validation data.

        Returns:
            The optimized model itself.
        """
        self.graph = self._create_graph(n_input=train.images.shape[1])  # Create the computational graph.
        session = tf.Session(graph=self.graph)
        coord = tf.train.Coordinator()  # Lets tensorflow handle multiple threads.
        threads = tf.train.start_queue_runners(session, coord)
        session.run(self.initializer)  # Initialize the network.
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
                    # Return the negative error to allow monitoring for the ELBO.
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
            # If interrupted or stopped, store the progress of the model.
            self.saver.save(session, self.checkpoint_path)
            session.close()
            coord.request_stop()
            coord.join(threads)
            print('finished training')
        return self

    def decode(self, z):
        """Decodes using the decoder network.

        Args:
            z: Returns a mini-batch of vectors in the input space, as non-linear transformation of latent vectors z.

        Returns:
            y, a generated example in the input space.
        """
        with tf.Session(graph=self.graph) as session:
            self._restore_model(session)
            y = session.run(self.y, {self.z: z})
        return y

    def encode(self, x):
        """Encodes the input mini-batch x in a latent embedding.

        Args:
            x: Mini-batch of input variables.

        Returns:
            mu, the mean of the probabilistic output p(z|x).
        """
        with tf.Session(graph=self.graph) as session:
            self._restore_model(session)
            mu = session.run(self.mu, {self.x: x})
        return mu

    def mosaic(self, grid_width=20):
        """Creates a mosaic of the given number of grid points per dimension, where each image corresponds to a
        percentile of the prior distribution p(z).

        Args:
            grid_width: The number of images per mosaic dimension.

        Returns:
            mos, a mosaic of the decoded latent variables representing draws from the prior p(z).
        """
        epsilon = norm.ppf(np.linspace(0, 1, grid_width + 2)[1:-1])
        z = np.dstack(np.meshgrid(epsilon, -epsilon)).reshape(-1, 2)
        image_width = int(np.sqrt(self.n_input))
        images = np.reshape(self.decode(z), [grid_width, grid_width, image_width, image_width])
        mos = 1 - np.bmat(images.tolist())
        return mos

if __name__ == '__main__':
    # This sample code trains the variational autoencoder on MNIST data and outputs some visualizations, as well
    # as the learning curve of the training and validation process.
    data = input_data.read_data_sets('data')
    vae = VAE()
    vae.fit(data.train, data.validation)
    latent_space = vae.encode(data.test.images)  # Project test images into the latent space.
    mosaic = vae.mosaic()

    sns.set_color_codes()
    sns.set_style('white')
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))

    # Visualize the image manifold corresponding to draws z ~ N(z|I,0) from the latent space.
    ax1.imshow(mosaic, cmap='gray')
    ax1.set_title('Learned MNIST manifold')
    ax1.axis('off')

    # Visualize the projections of the test data into the latent space z.
    cmap = mpl.cm.get_cmap('tab10')
    for c in range(10):
        idx = (data.test.labels == c)
        ax2.scatter(latent_space[idx, 0], latent_space[idx, 1], c=cmap(c), marker='.', label=str(c), alpha=0.7)
    ax2.set_title('Latent representation')
    ax2.axis('square')
    ax2.legend(frameon=True)
    sns.despine()

    # Visualize the learning curves for the training and validation examples.
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
    plt.savefig('vae.png', bbox_inches='tight')
    plt.show()
