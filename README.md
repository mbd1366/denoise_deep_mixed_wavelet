# denoise_deep_mixed_wavelet
Deep Convolutional Framelet Denoising for Panoramic by Mixed Wavelet Integration
Enhancing quality and removing noise during preprocessing is one of the most critical steps in image processing.
X-ray images are created by photons colliding with atoms and the variation in scattered noise absorption. 
This noise causes the graph's quality of medical to decline and, occasionally, causes it to repeat itself, causing an elevation in the patient's effective dose.
One of the most critical challenges in this area has consistently been lowering the image noise. Techniques like BM3d, low-pass filters, and Autoencoder 
have taken this step. Due to the algorithm's structure and high repetition rate, neural networks using various architectures have reduced noise 
with acceptable results over the past ten years compared to the traditional BM3D and low-pass filters. The Hankel matrix combined 
with neural networks is one of these configurations. The Hankel matrix seeks a local circle by splitting up individual values into local 
and non-local components using a non-local matrix. A non-local matrix can be created using the wave or DCT. 
This paper proposes combining the waveform with the Daubechies (D4) wavelength because it has more energy and uses the u-Net neural network structure,
which uses the waveform alone at each stage. 
The outcomes were evaluated using the PSNR and SSIM criteria, and the outcomes were verified by using various waves. 
The effectiveness of a one-wave network has increased from 0.5% to 1.2%, according to studies done on other datasets. 
