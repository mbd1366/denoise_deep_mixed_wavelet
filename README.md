# denoise_deep_mixed_wavelet
Enhancing quality and removing noise during preprocessing is one of the most critical steps in
image processing. X-ray images are created by photons colliding with atoms and the variation in scattered noise
absorption. This noise leads to a deterioration in the graph's medical quality and, at times, results in repetition, thereby
increasing the patient's effective dose. One of the most critical challenges in this area has consistently been lowering
the image noise. Techniques like BM3d, low-pass filters, and Autoencoder have taken this step. Owing to their structural
design and high rate of repetition, neural networks employing a variety of methods have, over the past decade, achieved
noise reduction with satisfactory outcomes, surpassing the traditional BM3D and low-pass filters. The combination of
the Hankel matrix with neural networks represents one of these configurations. The Hankel matrix aims to identify a local
circle by separating individual values into local and non-local components, utilizing a non-local matrix. A non-local
matrix can be created using the wave or DCT. This paper suggests integrating the waveform with the Daubechies (D4)
wavelet due to its higher energy concentration and employs the u-Net neural network architecture, which incorporates
the waveform exclusively at each stage. The outcomes were evaluated using the PSNR and SSIM criteria, and the
outcomes were verified by using various waves. The effectiveness of a one-wave network has increased from 0.5% to
1.2%, according to studies done on other datasets.


https%3A//doi.org/10.48550/arXiv.2302.10306
