# DeltaTrainer $5000 Computer Vision Challenge

My attempt at the above. It works for the simple test cases, but fails on more complex ones.

[Link to the challenge.](https://docs.google.com/document/d/12zMxu-BObYRD2Kgk5HbsTmekudC82s0fGJo_w8u4DFQ/edit?usp=sharing)

My most successful attempt required the user to highlight the logo; the program then tracks the logo based on keypoints and tries to inpaint using the shirt color. The problem with this is that, when the model is turned to the side and the logo "touches" the white background (or even if the logo touches the skin or any non-shirt material), that area tends to be blurred or inpainted as well. I was unable to find a pure computer vision solution for this problem. The only solution that I can see would be to train a CNN, but that then means one would need the corrected videos as training footage, which is obviously undesirable because that's what we're trying to automate. I am also unsure as to what degree of accuracy the CNN would be able to achieve (or even if this would be a good approach) because I have only basic knowledge of neural nets.

To use, run `remove_logo.py [path_to_test_video]`. The program prompts you to draw a box around the logo, then it inpaints the logo (to the best of its ability) for as many frames as possible. If it loses track of the logo, you will be prompted to draw a box around the logo again or press space to skip frames until the logo comes back into view.

Warning: the code was written as a stream of consciousness, without readability in mind, so it's very messy!