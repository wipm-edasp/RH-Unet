# RH-Unet

This repository contains the software implementation of RH-Unet, a method for utilizing deep learning to correct NMR spectra acquired in non-uniform magnetic fields. The detailed description and comparison with REFDCON can be found in the paper titled "Restore high-resolution NMR spectra from inhomogeneous magnetic fields using neural network".


## Features

The RH-Unet software provides a solution for restoring high-resolution NMR spectra by leveraging deep learning techniques. It addresses the issue of non-uniform magnetic fields during NMR spectroscopy and offers improved results compared to the REFDCON method. This implementation allows users to apply the RH-Unet algorithm to their own NMR data.

## Installation

Make sure you have the following tools and dependencies installed:

- torch - 1.7.1
- numpy - 1.22.4
- matplotlib - 3.5.1
- scipy - 1.7.3


## Usage

Provide detailed steps and examples on how to use.

To ensure a systematic approach for generating the specific data feature-label pairs for targeted NMR spectrum restoration, please follow the steps outlined below:
1.Exclude the water peak region from the spectrum.
2. Normalize the intensity of the singlet peak of the internal standard (reference) to 1.
3.Rescale the intensities of the spectrum, ensuring that the reference region falls within the range of 0 to 1.
4. Construct an ideal spectrum by generating a singlet peak at the same position as the reference peak.
5.Identify all the peaks present in the spectrum.
6.Replace the peaks in the rescaled spectrum with the actual singlet peak region of the internal standard to generate the data features. Additionally, replace the corresponding regions in the ideal spectrum with the ideal Lorentzian lineshape.
7.Rescale the feature-label data pair using the maximum value observed in both the features and labels.


## Notes

1. You should input the resonance frequency in MHz for the cal_t2 function.
2. You should define the water region for water_position cell.


## License

This project is licensed under the MIT License.

## Contact

You can reach me through the following channels:
- Email: xiaoxj007@outlook.com
