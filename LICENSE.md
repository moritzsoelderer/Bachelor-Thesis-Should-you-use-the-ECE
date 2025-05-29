# LICENSE


### My Code (MIT License)

MIT License

Copyright (c) Moritz Soelderer 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

### Third-Party Code and Licenses

This project includes code adapted from several sources.
Changes to the original version are marked by comments at the exact location of the change or
comments that reference changed line numbers.
Code was used from the following sources:


- **TCE: A Test-Based Approach to Measuring Calibration Error**  
    GitHub Repository: https://github.com/facebookresearch/tce  
    Licensed under the Creative Commons Attribution-NonCommercial (CC BY-NC) License.
    Adapted code is located in `src/metrics/tce.py`.


- **Calibration Error Estimation Using Fuzzy Binning**  
    GitHub Repository: https://github.com/bihani-g/fce
    Licensed under the MIT License.
    Adapted code is located in `src/metrics/fce.py`.

- **Calibration of Neural Networks using Splines**  
    GitHub Repository: https://github.com/kartikgupta-at-anu/spline-calibration
    Licensed under the MIT License.
    Adapted code is located in `src/metrics/ksce.py`.

- **Synthetic Data Generator**
Provided by co-supervisor Mohammad Hossein Shaker Ardakani. No License.
Adapted code is located in `src/experiments/optimal_ece_assessment/ml_approach/synthetic_data_generation.py`.

---

### Third-Party Libraries and Licenses
This project uses the following open-source libraries:

- **TensorFlow/Keras**  
  Copyright © Google Inc.  
  Licensed under the Apache License 2.0  
  https://www.tensorflow.org

- **PyTorch**  
  Copyright © Meta Platforms, Inc.  
  Licensed under the BSD 3-Clause License  
  https://pytorch.org

- **NumPy**  
  Copyright © NumPy Developers  
  Licensed under the BSD 3-Clause License  
  https://numpy.org

- **SciPy**  
  Copyright © SciPy Developers  
  Licensed under the BSD 3-Clause License  
  https://scipy.org

- **scikit-learn**  
  Copyright © scikit-learn Developers  
  Licensed under the BSD 3-Clause License  
  https://scikit-learn.org

These libraries are used under their respective open-source licenses. Their inclusion in this project does not imply endorsement.

