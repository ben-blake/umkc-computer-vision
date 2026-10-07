# umkc-computer-vision

Coursework for ECE/CS 5582 - Computer Vision at UMKC, Fall 2026.

| Assignment | Topic | Result |
| --- | --- | --- |
| [hw1-image-formation](hw1-image-formation/) | HSV color histograms for scene recognition, homography from SIFT matches by SVD | 60.5% on 15 NWPU classes against 6.7% chance; homography agrees with 84% of the matches |
| [hw2-log-keypoints](hw2-log-keypoints/) | Key points from scale space extrema of a LoG pyramid, compared with SIFT | 60% to 73% of non-edge LoG key points have a SIFT key point within 3 px against 7% to 14% by chance |

Everything runs in MATLAB R2025b and needs the Image Processing, Computer Vision, and Statistics and Machine Learning toolboxes. Each folder has its own README with the method, results and how to run it.
