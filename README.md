
# Pybell (WIP)

### Extract FaceSpace
To use EigenFaces for facial recognition, we must have a basis of vectors representing principal faces. These "eigenfaces" are the eigenvalues of a mean-centered training set of face images.

A script has been provided to extract a facespace and mean_face from a folder of face images.

```
python extract_facespace.py --folder=FACE_FOLDER_PATH --max-image=MAX_NUMBER_IMGS --components=PRINCIPAL_COMPONENTS --out=OUTPUT_FILE_PATH
```

A pre-extracted `facespace.pkl` has been provided for convenience

### Run the program:
To run the program run the following command
```
python pybell.py
```

## Camera Calibration
--
**TODO**

## Face Detection
--
OpenCV's Pretrained Haar Cascades can be found (here)[https://github.com/opencv/opencv/tree/master/data/haarcascades].

Provided Face Space was extracted using the LFWcrop dataset [1]. It is based on 64x64
grayscale images.

## Facial Recognition
--
Facial Recognition is done using the EigenFaces [2] process


## Citations
[1] G.B. Huang, M. Ramesh, T. Berg, E. Learned-Miller.
    Labeled Faces in the Wild: A Database for Studying
    Face Recognition in Unconstrained Environments.
    University of Massachusetts, Amherst,
    Technical Report 07-49, 2007.

[2] Turk, Matthew, and Alex Pentland. "Eigenfaces for recognition."
    Journal of cognitive neuroscience 3.1 (1991): 71-86.
