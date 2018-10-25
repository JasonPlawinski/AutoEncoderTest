# Dimensionality Reduction with Autoencoders

Comparing different Autoencoders with a bottleneck of 2 dimensions. Visualization and comparaison with PCA.  <br />

The PCA script is a notebook. The other ones are python scripts so the user can select what GPU to run on. <br />

The AutoEncoder used there are: A fully connected AutoEncoder with and without dropout. A Convolutional Autoencoder and an Adversarial Autoencoder (Gaussian Prior). <br />
The tests were done with between 50 and 100 epochs.<br />
The fully connected network and ConvNet both generate alright reconstruction but they use the latent space in a kind of wierd fashion.
The Adversarial AutoEncoder (AAE) has a quite evenly distributed latent space thanks to the Gaussian prior which make visualization easier but less separable.
<br />


![Screenshot](EncoderImage.png)