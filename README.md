# AWS Lambda Model Functions

This folder contains local backups of three AWS Lambda container-image functions.

The functions were pulled from Amazon ECR and saved locally so the code and model files can be reviewed, backed up, or moved to GitHub.

## Quick task/model map

| Task                                           | Model / tool used                                            |
| ---------------------------------------------- | ------------------------------------------------------------ |
| Run submitted code                             | No ML model. Uses Python, Node.js, Java, C, and C++ runtimes |
| Detect faces in an image                       | `FaceProctoring/app/app/checkpoints/yolov11n-face.onnx`    |
| Create face embeddings for identity comparison | `FaceProctoring/app/app/checkpoints/embedding_model.onnx`  |
| Compare input face with reference faces        | `embedding_model.onnx` + cosine similarity                 |
| Detect objects during proctoring               | `FaceProctoring/app/app/checkpoints/yolov8n-obj.onnx`      |
| Detect face/iris landmarks for gaze tracking   | `gaze/app/src/gaze_tracking/face_landmarker.task`          |
| Decide gaze direction                          | MediaPipe landmarks + custom gaze-ratio logic in Python      |

## Folder overview

```text
aws-lambda-functions/
  compiler/
    app/
    compiler-image.tar
    compiler-rootfs.tar
    compiler-image-inspect.json

  FaceProctoring/
    app/
    FaceProctoring-image.tar
    FaceProctoring-rootfs.tar
    FaceProctoring-image-inspect.json

  gaze/
    app/
    gaze-image.tar
    gaze-rootfs.tar
    gaze-image-inspect.json
```

## 1. `compiler`

Purpose: runs code submitted through an API.

What it does:

- Accepts source code from an HTTP request.
- Saves the code into a temporary file.
- Runs or compiles it depending on the language.
- Returns the program output or error.

Supported languages:

- Python 3
- Node.js
- Java
- C
- C++

Main files:

- `compiler/app/main.py`
- `compiler/app/routes/code_execution.py`
- `compiler/app/utils.py`
- `compiler/app/configs.py`

Main endpoints:

- `POST /compile/`
- `POST /compile/multi`
- `POST /compile/bulk`

Model used: none.

Important note: this function runs user-submitted code, so it should be treated carefully. It needs strong sandboxing, timeouts, authentication, and resource limits before public production use.

## 2. `FaceProctoring`

Purpose: analyzes a camera image for face proctoring.

What it does:

- Detects faces.
- Optionally detects objects.
- Creates a face embedding from the detected face.
- Compares the input face against reference face images.
- Returns face count, face boxes, object detections, and similarity scores.

Models used:

| File                                                        | Used for                             |
| ----------------------------------------------------------- | ------------------------------------ |
| `FaceProctoring/app/app/checkpoints/yolov11n-face.onnx`   | Face detection                       |
| `FaceProctoring/app/app/checkpoints/embedding_model.onnx` | Face embedding / identity similarity |
| `FaceProctoring/app/app/checkpoints/yolov8n-obj.onnx`     | Object detection                     |

Main files:

- `FaceProctoring/app/main.py`
- `FaceProctoring/app/app/main.py`
- `FaceProctoring/app/app/image_processing.py`
- `FaceProctoring/app/app/utils.py`
- `FaceProctoring/app/app/model_loader.py`

Main endpoints:

- `POST /analyze`
- `POST /embedding`
- `POST /convert_to_base64`

Simple flow:

```text
base64 image
  -> YOLO face detection
  -> crop detected face
  -> embedding_model.onnx creates face vector
  -> compare with reference vectors using cosine similarity
  -> return match scores
```

If object detection is enabled:

```text
base64 image
  -> yolov8n-obj.onnx
  -> return detected object labels and boxes
```

## 3. `gaze`

Purpose: detects where the user is looking.

What it does:

- Accepts a base64 image.
- Uses MediaPipe FaceLandmarker to find face and eye landmarks.
- Locates iris/pupil positions.
- Calculates gaze ratios.
- Returns whether gaze is center, left, right, up, down, or no face.

Model used:

| File                                                | Used for                                   |
| --------------------------------------------------- | ------------------------------------------ |
| `gaze/app/src/gaze_tracking/face_landmarker.task` | MediaPipe face and iris landmark detection |

Main files:

- `gaze/app/main.py`
- `gaze/app/src/gaze_tracking/iris_tracker.py`
- `gaze/app/utils.py`

Main endpoints:

- `POST /vision`
- `POST /vision/`
- `GET /health`
- `GET /ping`

Simple flow:

```text
base64 image
  -> decode image
  -> MediaPipe FaceLandmarker finds face/eye/iris points
  -> custom Python logic calculates gaze direction
  -> return gaze response
```

Output includes:

- gaze status
- gaze direction
- pupil coordinates
- face bounding box
- eye details
- gaze metrics

## Backup files

Each function folder contains:

| File/folder              | Meaning                                                                               |
| ------------------------ | ------------------------------------------------------------------------------------- |
| `app/`                 | readable application code and model files copied from the container working directory |
| `*-image.tar`          | full Docker image backup                                                              |
| `*-rootfs.tar`         | full exported Linux container filesystem backup                                       |
| `*-image-inspect.json` | Docker image metadata                                                                 |

The `.tar` files are large. If moving this to GitHub, consider using Git LFS or leaving the `.tar` files out of the repository.

## GitHub cleanup suggestion

Before pushing to GitHub, consider excluding:

```text
*.tar
**/venv*/
**/__pycache__/
*.pyc
```

Also review the code for secrets, tokens, AWS account IDs, or environment variables before making the repository public.
